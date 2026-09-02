# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The role-masked cross-teacher weighting arms.

Two arms, ``role_mask=content`` and ``role_mask=structural``, restrict the
reallocation to one half of the response. They exist to separate the two
surviving explanations for the corrected 150-step result, and the property that
makes them able to do that is narrow: they must differ from the unmasked arm and
from each other in WHERE the reallocation lands and in nothing else. In
particular the effective distillation strength has to stay put, which is the
``kl_scale == 1`` invariant, and that only holds if the normaliser's mean is
taken over the same positions the weight was applied to.

So the tests here are mostly one test written from several directions: the mask
reaches the weight, the normaliser and the attribution consistently, or the arm
silently becomes "less distillation" and answers neither question.
"""
import os as _os

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        POSITION_TERMS,
        ROLE_GROUPS,
        PreviousStepTaskKLWeightedMean,
        build_position_weight,
        compute_raw_policy_shifts,
        position_terms,
        position_weight_metrics,
        role_keep_mask,
    )
    from verl.trainer.ppo.sign_weights import (
        ScopeTermStats,
        ROLE_ENV_ACTION,
        ROLE_ENV_OBS,
        ROLE_FORMAT,
        ROLE_NAMES,
        ROLE_REASONING,
        ROLE_TAG,
        ROLE_TOOL_CALL,
    )
except Exception as e:  # pragma: no cover
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

TASKS = ("alfworld", "search", "webshop")


def _lp(bs, resp, k, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.log_softmax(torch.randn(bs, resp, k), dim=-1)


def _inputs(bs=6, resp=8, k=5, n_off=2, seed=7):
    torch.manual_seed(seed)
    on, base = _lp(bs, resp, k), _lp(bs, resp, k)
    off = torch.stack([_lp(bs, resp, k) for _ in range(n_off)], dim=-1)
    task_ids = torch.arange(bs) % 3
    alpha = torch.full((3, 3), 0.5)
    alpha.fill_diagonal_(0.0)
    return {
        "shifts": compute_raw_policy_shifts(
            on_task_logprob=on, off_task_logprobs=off, base_logprob=base
        ),
        "on_task_logprob": on,
        "student_logprob": _lp(bs, resp, k, seed=seed + 1),
        "task_ids": task_ids,
        "off_plane_tasks": torch.stack(
            [(task_ids + 1) % 3, (task_ids + 2) % 3], dim=-1
        )[:, :n_off],
        "diag": torch.ones(3),
        "diag_valid": torch.ones(3, dtype=torch.bool),
        "alpha_table": alpha,
        "response_mask": torch.ones(bs, resp),
    }


def _roles(bs=6, resp=8, seed=3):
    """A role code per position, every group represented in every row."""
    torch.manual_seed(seed)
    codes = torch.tensor(
        [ROLE_TAG, ROLE_FORMAT, ROLE_REASONING, ROLE_ENV_ACTION, ROLE_TOOL_CALL, ROLE_ENV_OBS]
    )
    return codes[torch.randint(0, codes.numel(), (bs, resp))]


def _build(kw, *, role_keep=None, normalizer=None):
    return build_position_weight(**kw, role_keep=role_keep, normalizer=normalizer)


def _snapshot(kw, built, kl, *, role_keep=None):
    mean = PreviousStepTaskKLWeightedMean(n_tasks=3, device="cpu")
    mean.update(
        pre_weight=built["pre_weight"], teacher_kl=kl,
        response_mask=kw["response_mask"], task_ids=kw["task_ids"],
        role_keep=role_keep,
    )
    return mean.snapshot()


def _kl_scale(kw, built, kl):
    stats = ScopeTermStats(names=POSITION_TERMS, n_tasks=3, device="cpu")
    stats.update(position_terms(built, kl), response_mask=kw["response_mask"],
                 task_ids=kw["task_ids"])
    return position_weight_metrics(stats.sums(task_names=TASKS))


# ------------------------------------------------------------------ the groups
def test_the_two_groups_partition_the_six_roles_with_nothing_left_over():
    """A role in neither group would be silently masked off in both arms, so the
    pair would not be a partition of the response and "content-only plus
    structural-only equals the unmasked arm" would stop being true."""
    st, co = set(ROLE_GROUPS["structural"]), set(ROLE_GROUPS["content"])
    assert st.isdisjoint(co)
    assert st | co == set(ROLE_NAMES)


def test_the_groups_are_named_after_what_they_keep():
    assert set(ROLE_GROUPS["structural"]) == {ROLE_TAG, ROLE_FORMAT}
    assert set(ROLE_GROUPS["content"]) == {
        ROLE_REASONING, ROLE_ENV_ACTION, ROLE_TOOL_CALL, ROLE_ENV_OBS
    }


def test_the_mask_keeps_exactly_its_group():
    roles = _roles()
    for group, codes in ROLE_GROUPS.items():
        keep = role_keep_mask(roles=roles, group=group)
        want = torch.zeros_like(roles, dtype=torch.bool)
        for c in codes:
            want |= roles == c
        assert torch.equal(keep, want)


def test_the_two_masks_are_complementary():
    roles = _roles()
    a = role_keep_mask(roles=roles, group="structural")
    b = role_keep_mask(roles=roles, group="content")
    assert torch.equal(a, ~b)


def test_an_unknown_role_code_is_masked_off_rather_than_kept():
    """Built by inclusion, not by exclusion. A span type added to the prompts
    later must not join the acted set without the mask being told about it."""
    roles = torch.full((1, 3), 99, dtype=torch.long)
    for group in ROLE_GROUPS:
        assert not role_keep_mask(roles=roles, group=group).any()


def test_an_unknown_group_is_rejected_rather_than_treated_as_no_mask():
    with pytest.raises(ValueError, match="unknown role group"):
        role_keep_mask(roles=_roles(), group="everything")


# ------------------------------------------------------- the weight it produces
def test_a_masked_position_is_exactly_one_and_not_one_over_mu():
    """THE placement regression. The mask goes after ``pre / mu``, so a masked
    position contributes its own unweighted KL. Applied to ``pre`` before the
    division instead it would come out at ``1/mu`` -- a uniform rescale of
    whichever half the arm claims not to touch, which is precisely the confound
    the normaliser exists to remove."""
    kw = _inputs()
    keep = role_keep_mask(roles=_roles(), group="content")
    norm = {"mean": torch.full((3,), 1.4), "valid": torch.ones(3, dtype=torch.bool)}
    got = _build(kw, role_keep=keep, normalizer=norm)
    masked = got["weight"][~keep]
    assert torch.equal(masked, torch.ones_like(masked))
    # and 1/mu is a real number here, so the wrong placement would be visible
    assert abs(1.0 / 1.4 - 1.0) > 0.2


def test_a_retained_position_is_bit_identical_to_the_unmasked_arm():
    kw = _inputs()
    keep = role_keep_mask(roles=_roles(), group="content")
    norm = {"mean": torch.full((3,), 1.4), "valid": torch.ones(3, dtype=torch.bool)}
    plain = _build(kw, normalizer=norm)
    got = _build(kw, role_keep=keep, normalizer=norm)
    assert torch.equal(got["weight"][keep], plain["weight"][keep])


def test_no_mask_is_bit_identical_to_the_arm_that_has_been_running():
    """``role_mask=""`` must be the same run, not a nearly-same one: the control
    comparison for both masked arms is the unmasked arm's existing 150 steps."""
    kw = _inputs()
    norm = {"mean": torch.full((3,), 1.2), "valid": torch.ones(3, dtype=torch.bool)}
    a = build_position_weight(**kw, normalizer=norm)
    b = _build(kw, role_keep=None, normalizer=norm)
    assert torch.equal(a["weight"], b["weight"])
    assert a["role_keep"] is None and b["role_keep"] is None


def test_the_mask_is_returned_so_the_normaliser_reads_the_same_tensor():
    """Rebuilding it from the roles at the second call site is how the two come
    to disagree; the build hands its own back instead."""
    kw = _inputs()
    keep = role_keep_mask(roles=_roles(), group="structural")
    got = _build(kw, role_keep=keep,
                 normalizer={"mean": torch.full((3,), 1.1), "valid": torch.ones(3, dtype=torch.bool)})
    assert got["role_keep"] is not None
    assert torch.equal(got["role_keep"], keep)


def test_the_masked_half_still_carries_its_own_teacher_kl():
    """W = 1 is not W = 0. The arm reallocates within one half and leaves the
    other half distilling exactly as the control does -- if the masked half lost
    its KL the arm would be an ablation of the teacher, not of the mechanism."""
    kw = _inputs()
    keep = role_keep_mask(roles=_roles(), group="content")
    kl = torch.rand(*kw["response_mask"].shape) + 0.1
    got = _build(kw, role_keep=keep,
                 normalizer={"mean": torch.full((3,), 1.3), "valid": torch.ones(3, dtype=torch.bool)})
    applied = got["weight"] * kl
    assert torch.allclose(applied[~keep], kl[~keep])


# ------------------------------------------------------------- the invariant
def test_kl_scale_stays_one_when_the_weight_and_the_normaliser_share_the_mask():
    """The property the arms are built on. ``sum W*D == sum D`` over the WHOLE
    response: the retained half reallocates around its own KL-weighted mean and
    the masked half sits at 1, so the total is untouched and the arm differs
    from its control in placement alone."""
    kw = _inputs(bs=6, resp=16, seed=11)
    keep = role_keep_mask(roles=_roles(bs=6, resp=16, seed=12), group="content")
    kl = torch.rand(6, 16) + 0.1
    first = _build(kw, role_keep=keep)
    snap = _snapshot(kw, first, kl, role_keep=keep)
    again = _build(kw, role_keep=keep, normalizer=snap)
    m = _kl_scale(kw, again, kl)
    assert m["kl_weight/effect/kl_scale"] == pytest.approx(1.0, abs=1e-5)
    for task in TASKS:
        assert m[f"kl_weight/{task}/effect/kl_scale"] == pytest.approx(1.0, abs=1e-5)


def test_kl_scale_leaves_one_if_only_the_weight_carries_the_mask():
    """The failure this wiring exists to prevent, asserted so it cannot come
    back quietly. With ``mu`` taken over every position while ``W`` applies to a
    subset, the arm distils a different total amount than its control -- and
    "less distillation" is one of the two explanations these arms are supposed
    to be able to rule out."""
    kw = _inputs(bs=6, resp=16, seed=11)
    keep = role_keep_mask(roles=_roles(bs=6, resp=16, seed=12), group="content")
    kl = torch.rand(6, 16) + 0.1
    first = _build(kw, role_keep=keep)
    # the mask withheld from the normaliser, which is the bug
    wrong = _snapshot(kw, first, kl, role_keep=None)
    again = _build(kw, role_keep=keep, normalizer=wrong)
    m = _kl_scale(kw, again, kl)
    assert abs(m["kl_weight/effect/kl_scale"] - 1.0) > 1e-3


def test_the_normaliser_ignores_the_masked_positions_entirely():
    """Not merely down-weights them: a masked position's ``W~`` must not move
    ``mu`` at all, or the retained half's reallocation is centred on evidence
    from positions the arm never acts on."""
    pre = torch.tensor([[2.0, 9.0]])
    kl = torch.ones(1, 2)
    ids = torch.zeros(1, dtype=torch.long)
    keep = torch.tensor([[True, False]])
    masked = PreviousStepTaskKLWeightedMean(n_tasks=1, device="cpu")
    masked.update(pre_weight=pre, teacher_kl=kl, response_mask=torch.ones(1, 2),
                  task_ids=ids, role_keep=keep)
    only = PreviousStepTaskKLWeightedMean(n_tasks=1, device="cpu")
    only.update(pre_weight=pre[:, :1], teacher_kl=kl[:, :1],
                response_mask=torch.ones(1, 1), task_ids=ids)
    assert float(masked.snapshot()["mean"][0]) == pytest.approx(
        float(only.snapshot()["mean"][0]), abs=1e-12
    )
    assert float(masked.snapshot()["mean"][0]) == pytest.approx(2.0, abs=1e-12)


# ---------------------------------------------------------- the attribution
def test_the_push_decomposition_is_still_an_identity_under_the_mask():
    """``W - 1`` splits three ways, and the split has to follow the mask: at a
    masked position ``W - 1`` is zero, so all three terms must be too. Left
    ungated, the attribution would charge corroboration evidence at positions
    the loss never saw it at."""
    kw = _inputs()
    keep = role_keep_mask(roles=_roles(), group="structural")
    got = _build(kw, role_keep=keep,
                 normalizer={"mean": torch.full((3,), 1.25), "valid": torch.ones(3, dtype=torch.bool)})
    total = got["push_shared"] + got["push_by_source"].sum(dim=-1) + got["push_normalizer"]
    assert torch.allclose(total, got["weight"] - 1.0, atol=1e-5)
    for name in ("push_shared", "push_normalizer"):
        vals = got[name][~keep]
        assert torch.equal(vals, torch.zeros_like(vals)), name
    src = got["push_by_source"][~keep]
    assert torch.equal(src, torch.zeros_like(src))


def test_the_normaliser_offset_still_reaches_the_retained_half():
    """The gate zeroes the masked half and must not zero the retained one: the
    ``1/mu - 1`` term is the arm's own whole-task divisor showing up at
    positions with no evidence, and it is real wherever the arm acts."""
    kw = _inputs()
    keep = role_keep_mask(roles=_roles(), group="content")
    got = _build(kw, role_keep=keep,
                 normalizer={"mean": torch.full((3,), 1.25), "valid": torch.ones(3, dtype=torch.bool)})
    assert got["push_normalizer"][keep].abs().max() > 0


# ------------------------------------------------------------------- the arm
yaml = pytest.importorskip("yaml")
pytest.importorskip("hydra")

from tests.trainer.test_signweight_arms_match_the_control import (  # noqa: E402
    IDENTITY,
    _differing,
    _effective,
)

REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
CONTENT = "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_content_qwen3.sh"
UNMASKED = "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_qwen3.sh"
CONTROL = "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_control_qwen3.sh"
# The composed config carries the algorithm-side key only: main_opd copies the
# block onto the actor at startup, AFTER Hydra. The intent lock is validated
# after that copy, so it pins both -- and pinning both is what catches a copy
# that silently stopped happening.
COMPOSED_MASK_KEY = "algorithm.opd.cross_teacher_kl_weight.role_mask"
LOCK_MASK_KEYS = (
    COMPOSED_MASK_KEY,
    "actor_rollout_ref.actor.cross_teacher_kl_weight.role_mask",
)


def test_the_content_arm_differs_from_the_unmasked_arm_in_the_mask_alone():
    """The whole design. Its comparison run is the unmasked arm's own control,
    which has already run to 298 steps, so a second difference would make the
    three-way reading meaningless."""
    differing = _differing(_effective(UNMASKED), _effective(CONTENT)) - IDENTITY
    assert differing == {COMPOSED_MASK_KEY}, sorted(differing)
    assert _effective(CONTENT)[COMPOSED_MASK_KEY] == "content"


def test_the_content_arm_differs_from_the_control_only_inside_the_mechanism():
    """Against the control the difference is the mechanism plus the mask, and
    nothing about the environments, the data, the optimiser or the OPD
    coefficient."""
    differing = _differing(_effective(CONTROL), _effective(CONTENT)) - IDENTITY
    assert differing, "the control must differ somewhere -- it is the same script otherwise"
    assert all("cross_teacher_kl_weight" in k for k in differing), sorted(differing)


def test_the_unmasked_arm_is_still_unmasked():
    """The 150 steps this comparison rests on were run without a mask, so the
    key has to read as absent there rather than as some default group."""
    assert _effective(UNMASKED).get(COMPOSED_MASK_KEY) in (None, "")


def test_the_arm_pins_its_mask_in_the_intent_lock():
    """A run-script-only edit must refuse to start. An unmasked run under this
    arm's name is the one failure no metric catches: kl_scale would be 1, every
    gate would pass, and the chart would claim a mask that was not applied."""
    arm = _effective(CONTENT)
    lock = yaml.safe_load(open(_os.path.join(REPO, str(arm["trainer.expected_config"]))))
    for key in LOCK_MASK_KEYS:
        assert lock.get(key) == "content", key


def test_the_two_arms_do_not_share_a_directory():
    """Same reason as every other pair on this branch: the token dumps and the
    validation logs are read by path, and two arms writing one directory is a
    silent merge of two experiments."""
    a, b = _effective(UNMASKED), _effective(CONTENT)
    for key in ("trainer.default_local_dir", "trainer.val_instance_log_dir",
                "trainer.sign_token_dump_dir", "trainer.experiment_name",
                "trainer.project_name"):
        assert a[key] != b[key], key


def test_the_state_partition_follows_the_mask_not_just_the_weight():
    """``state_shift_terms`` and ``per_candidate_shift`` are built from ``mu``
    and the evidence rather than from ``W``, so the mask has to be handed to
    them. At a masked position ``(W - 1) D`` is zero while ``(1/mu - 1) D`` is
    not, and an ungated offset column would report a shift the loss never took
    -- in the ONE table the per-state, per-token and source attributions all
    read."""
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        STATE_TERMS, per_candidate_shift, state_shift_terms,
    )

    kw = _inputs()
    keep = role_keep_mask(roles=_roles(), group="content")
    got = _build(kw, role_keep=keep,
                 normalizer={"mean": torch.full((3,), 1.35),
                             "valid": torch.ones(3, dtype=torch.bool)})
    kl = torch.rand(*got["weight"].shape) + 0.1
    terms = state_shift_terms(got, kl)
    total = sum(terms[t] for t in STATE_TERMS)
    assert torch.allclose(total, (got["weight"] - 1.0) * kl, atol=1e-5)
    for name, col in terms.items():
        vals = col[~keep]
        assert torch.equal(vals, torch.zeros_like(vals)), name
    cand = per_candidate_shift(got, kl)[~keep]
    assert torch.equal(cand, torch.zeros_like(cand))
    # and the retained half is not zeroed along with it
    assert sum(terms[t] for t in STATE_TERMS)[keep].abs().max() > 0


def test_a_built_without_the_key_is_treated_as_unmasked():
    """The probe and channel dicts are assembled at the call site. An unmasked
    arm's dict has no role_keep at all, and that must mean "no mask" rather than
    "mask everything off"."""
    from verl.trainer.ppo.cross_teacher_kl_weight import STATE_TERMS, state_shift_terms

    kw = _inputs()
    got = _build(kw, normalizer={"mean": torch.full((3,), 1.35),
                                 "valid": torch.ones(3, dtype=torch.bool)})
    assert got["role_keep"] is None
    kl = torch.rand(*got["weight"].shape) + 0.1
    terms = state_shift_terms({k: v for k, v in got.items() if k != "role_keep"}, kl)
    total = sum(terms[t] for t in STATE_TERMS)
    assert torch.allclose(total, (got["weight"] - 1.0) * kl, atol=1e-5)
