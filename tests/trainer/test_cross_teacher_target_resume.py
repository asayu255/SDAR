"""The target arm's resume: the scale survives, and the two arms stay apart.

The curriculum reads exactly one thing that crosses a step boundary -- the
diagonal of ``CumulativePolicyShiftRMS``, which ``nested_layers`` divides by to
decide how many teachers back a candidate. Everything else it consumes is the
batch or ``curriculum_rho``, a pure function of the driver's ``global_steps``.

That one thing used to be dropped on resume, on the argument that it re-warms
from live batches. It does not re-warm to where it was: sigma drifts DOWN as the
student moves, so an accumulation restarted at step k never contains the earlier
larger contributions. The run that died at step 59, resumed at 50 against its own
trace, held search at 0.763x for five straight steps and admitted 16-19% more
candidates into ``shared`` -- the very layer stage 1 distils. These tests pin the
restore, and pin that neither arm can load the other's file.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        CumulativePolicyShiftRMS,
        SIDECAR_VERSION,
        resume_identity,
    )
    from verl.trainer.ppo.cross_teacher_target import (
        TARGET_MECHANISM_ID,
        TARGET_SIDECAR_NAME,
        TARGET_SIDECAR_VERSION,
        load_target_sidecar_state,
        target_sidecar_state,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

TASKS = ["alfworld", "search", "webshop"]
SNAPSHOT = {
    "base_path": "Qwen/Qwen3-1.7B",
    "temperature": 1.0,
    "teacher_paths": {"alfworld": "/t/a", "search": "/t/s", "webshop": "/t/w"},
}


def _identity():
    return resume_identity(SNAPSHOT, TASKS)


def _fold(rms, g, scale=1.0):
    """One micro-batch of the shape :meth:`update` expects, then the reduce."""
    bs, resp, k, n_off = 2, 5, 4, 2
    rms.update(
        shifts={
            "on": torch.randn(bs, resp, k, generator=g) * scale,
            "off": torch.randn(bs, resp, k, n_off, generator=g) * scale,
            "tail_on": torch.randn(bs, resp, generator=g) * scale,
            "tail_off": torch.randn(bs, resp, n_off, generator=g) * scale,
        },
        # log-probs over the support: normalised so the tail mass is positive.
        student_logprob=torch.log_softmax(
            torch.randn(bs, resp, k + 1, generator=g), dim=-1
        )[..., :k],
        response_mask=torch.ones(bs, resp),
        task_ids=torch.tensor([0, 2]),
        off_plane_tasks=torch.tensor([[1, 2], [0, 1]]),
    )
    rms.all_reduce()


def _warmed(steps=4, seed=0, scale=1.0):
    """An RMS carrying several steps of history, as a mid-run one would."""
    g = torch.Generator().manual_seed(seed)
    rms = CumulativePolicyShiftRMS(n_tasks=3, device=torch.device("cpu"))
    for _ in range(steps):
        _fold(rms, g, scale=scale)
    return rms


def test_the_restored_scale_is_the_one_the_weight_divides_by():
    """``diagonal()`` after a round trip is what the uninterrupted run reads."""
    src = _warmed()
    want_diag, want_valid = src.diagonal()

    blob = target_sidecar_state(rms=src, step_index=50, identity=_identity())
    dst = CumulativePolicyShiftRMS(n_tasks=3, device=torch.device("cpu"))
    step = load_target_sidecar_state(blob, rms=dst, identity=_identity())

    got_diag, got_valid = dst.diagonal()
    torch.testing.assert_close(got_diag, want_diag, rtol=0, atol=0)
    assert torch.equal(got_valid, want_valid)
    assert step == 50


def test_a_cold_start_is_a_different_scale_and_not_a_small_one():
    """The failure the restore exists to prevent, as a number rather than a claim."""
    src = _warmed(steps=8)
    cold = CumulativePolicyShiftRMS(n_tasks=3, device=torch.device("cpu"))
    # One step of history, which is what a resumed run had at its first step --
    # and drawn at a smaller scale, which is the direction the live run showed
    # (search: sigma 3.42 at step 2, 2.76 at step 59).
    _fold(cold, torch.Generator().manual_seed(99), scale=0.5)
    warm_diag, _ = src.diagonal()
    cold_diag, _ = cold.diagonal()
    assert not torch.allclose(warm_diag, cold_diag, rtol=1e-2, atol=1e-2)
    # And the counts, which are what make the NEXT step a small correction
    # rather than a re-estimate: the cold accumulator carries a fraction.
    assert cold.snapshot()["n"].sum() < src.snapshot()["n"].sum()


def test_the_accumulated_counts_survive_not_just_the_ratio():
    """``n`` matters: it is what makes the next step a small correction."""
    src = _warmed(steps=6)
    blob = target_sidecar_state(rms=src, step_index=7, identity=_identity())
    dst = CumulativePolicyShiftRMS(n_tasks=3, device=torch.device("cpu"))
    load_target_sidecar_state(blob, rms=dst, identity=_identity())
    torch.testing.assert_close(dst.snapshot()["n"], src.snapshot()["n"], rtol=0, atol=0)


def test_the_step_index_rides_along_so_two_logs_dump_on_the_same_steps():
    blob = target_sidecar_state(rms=_warmed(), step_index=137, identity=_identity())
    dst = CumulativePolicyShiftRMS(n_tasks=3, device=torch.device("cpu"))
    assert load_target_sidecar_state(blob, rms=dst, identity=_identity()) == 137


@pytest.mark.parametrize(
    "key, value",
    [
        ("base_path", "Qwen/Qwen3-4B"),
        ("temperature", 0.7),
        ("teacher_paths", {"alfworld": "/t/a2", "search": "/t/s", "webshop": "/t/w"}),
    ],
)
def test_a_different_measurement_frame_is_refused(key, value):
    blob = target_sidecar_state(rms=_warmed(), step_index=1, identity=_identity())
    other = dict(SNAPSHOT)
    other[key] = value
    dst = CumulativePolicyShiftRMS(n_tasks=3, device=torch.device("cpu"))
    with pytest.raises(AssertionError, match=key):
        load_target_sidecar_state(blob, rms=dst, identity=resume_identity(other, TASKS))


def test_a_different_task_axis_is_refused():
    """Every matrix is indexed by this order; a permutation is not a relabel."""
    blob = target_sidecar_state(rms=_warmed(), step_index=1, identity=_identity())
    dst = CumulativePolicyShiftRMS(n_tasks=3, device=torch.device("cpu"))
    with pytest.raises(AssertionError, match="task_order"):
        load_target_sidecar_state(
            blob, rms=dst, identity=resume_identity(SNAPSHOT, ["search", "alfworld", "webshop"])
        )


def test_an_identity_key_the_checkpoint_never_recorded_is_refused():
    """Absent must not pass, or the check weakens with the checkpoint's age."""
    blob = target_sidecar_state(rms=_warmed(), step_index=1, identity=_identity())
    blob["identity"].pop("temperature")
    dst = CumulativePolicyShiftRMS(n_tasks=3, device=torch.device("cpu"))
    with pytest.raises(AssertionError, match="temperature"):
        load_target_sidecar_state(blob, rms=dst, identity=_identity())


def test_a_stale_version_is_refused():
    blob = target_sidecar_state(rms=_warmed(), step_index=1, identity=_identity())
    blob["version"] = TARGET_SIDECAR_VERSION + 1
    dst = CumulativePolicyShiftRMS(n_tasks=3, device=torch.device("cpu"))
    with pytest.raises(AssertionError, match="sidecar version"):
        load_target_sidecar_state(blob, rms=dst, identity=_identity())


# --------------------------------------------------------------------------- #
# the two arms
# --------------------------------------------------------------------------- #
def test_the_weighting_arm_s_file_is_refused_by_this_loader():
    """Same RMS, different rule. The mechanism id is what separates them."""
    from verl.trainer.ppo.cross_teacher_kl_weight import sidecar_state as klw_state
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        AdvantageReliabilityStats,
        PreviousStepTaskKLWeightedMean,
    )

    dev = torch.device("cpu")
    blob = klw_state(
        rms=_warmed(),
        mean=PreviousStepTaskKLWeightedMean(n_tasks=3, device=dev),
        adv=AdvantageReliabilityStats(n_tasks=3, device=dev, max_groups=8),
        alpha=torch.zeros((3, 3)),
        identity=_identity(),
    )
    dst = CumulativePolicyShiftRMS(n_tasks=3, device=dev)
    # Refused by whichever gate comes first -- today the version, since the two
    # files are at different vintages. What is pinned is that THIS loader is the
    # one refusing it, not that a particular gate caught it.
    with pytest.raises(AssertionError, match="cross_teacher_target resume"):
        load_target_sidecar_state(blob, rms=dst, identity=_identity())


def test_a_foreign_mechanism_at_this_version_is_refused():
    """The gate that will still be there when the versions happen to coincide."""
    blob = target_sidecar_state(rms=_warmed(), step_index=1, identity=_identity())
    blob["mechanism"] = "source_similarity_v1"
    dst = CumulativePolicyShiftRMS(n_tasks=3, device=torch.device("cpu"))
    with pytest.raises(AssertionError, match="mechanism"):
        load_target_sidecar_state(blob, rms=dst, identity=_identity())


def test_this_arm_s_file_is_refused_by_the_weighting_arm_s_loader():
    from verl.trainer.ppo.cross_teacher_kl_weight import load_sidecar_state as klw_load
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        AdvantageReliabilityStats,
        PreviousStepTaskKLWeightedMean,
    )

    dev = torch.device("cpu")
    blob = target_sidecar_state(rms=_warmed(), step_index=1, identity=_identity())
    with pytest.raises(AssertionError):
        klw_load(
            blob,
            rms=CumulativePolicyShiftRMS(n_tasks=3, device=dev),
            mean=PreviousStepTaskKLWeightedMean(n_tasks=3, device=dev),
            adv=AdvantageReliabilityStats(n_tasks=3, device=dev, max_groups=8),
            identity=_identity(),
        )


def test_the_two_files_do_not_share_a_name():
    from verl.trainer.ppo.cross_teacher_kl_weight import SIDECAR_NAME

    assert TARGET_SIDECAR_NAME != SIDECAR_NAME
    assert TARGET_MECHANISM_ID != "source_similarity_v1"
    # Version numbers are per file, so they are allowed to coincide; the name
    # and the mechanism are what keep the two apart.
    assert isinstance(SIDECAR_VERSION, int)
