"""The student-indexed top-k arm of off-policy KD, at the seams the port touches.

``tests/trainer/test_teacher_hidden_cache.py`` covers the cache itself: that
``h @ W[ids].T - lse`` reproduces a full log-softmax, that a row answered from
another row's entry raises, that ownership is unique. What is left, and what this
file pins, is everything that is specific to doing it OFF-policy:

1. **The flag is a gate, not a hint.** Off, Stage 2 loads no teacher and reads the
   pool's precomputed top-k -- the arm this trainer was written for. On, it must
   refuse to start without the teachers, because the pool's top-k was chosen
   before the student existed and cannot answer at ids the student picks.
2. **The keys survive the batch.** Ids are assigned after ``adjust_batch`` has
   padded and ``_balance_batch`` has reordered, and every row -- padding copies
   included -- gets one. A padding row left at -1 would be trained against a zero
   teacher target rather than skipped.
3. **The oracle the off-policy arm alone has.** Stage 1 recorded the teacher's own
   top-k for these exact rows. The Stage-2 teacher, resolved from cached hidden
   states at those same recorded ids, must reproduce the recorded values. The
   on-policy arm can only check the cache against a forward it just ran; here the
   reference is on disk, written by a different process on a different day.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.workers.teacher_cache import (
        TeacherHiddenCache,
        exchange_teacher_logprobs,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


H, VOCAB, K = 48, 400, 6


def _stage1(seed=0, rows=12, positions=3):
    """What Stage 1 would have written, plus what Stage 2 would recompute.

    Returns the teacher's projection, the hidden states and normaliser of every
    (row, position), and the top-k columns the pool carries.
    """
    g = torch.Generator().manual_seed(seed)
    W = torch.randn((VOCAB, H), generator=g) / H**0.5
    h = torch.randn((rows, positions, H), generator=g) / 2
    logits = h @ W.T
    lse = torch.logsumexp(logits, dim=-1)
    topk_logprobs, topk_ids = torch.topk(torch.log_softmax(logits, dim=-1), K, dim=-1)
    return W, h, lse, topk_logprobs, topk_ids


# --------------------------------------------------------------------------- #
# 1. the flag is a gate
# --------------------------------------------------------------------------- #


class _Cfg(dict):
    """Enough of an OmegaConf node for the trainer's __init__ reads."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def get(self, key, default=None):
        return dict.get(self, key, default)


def _trainer_init(student_indexed_topk, teacher_paths=None):
    """Run OffPolicyOPDRayTrainer.__init__'s own body against a stub config.

    Called unbound on a bare instance so the base RayPPOTrainer's constructor --
    which wants Ray, datasets and a resource pool -- stays out of it. The point
    is the branch this arm added, not the plumbing under it.
    """
    from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer

    opd = _Cfg(teacher_data_dir="/pool", student_indexed_topk=student_indexed_topk)
    if teacher_paths is not None:
        opd["teacher_paths"] = teacher_paths
    return _init(
        opd,
        _Cfg(teacher_kl_loss_type="topk_kl", teacher_kl_topk=20,
             student_indexed_topk=student_indexed_topk),
    )


def _init(opd, actor):
    """Run OffPolicyOPDRayTrainer.__init__ with its super() call stubbed out.

    The base RayPPOTrainer constructor wants a live Ray cluster, datasets and a
    resource pool; none of that is what this file is about.
    """
    from unittest.mock import patch

    from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = OffPolicyOPDRayTrainer.__new__(OffPolicyOPDRayTrainer)
    trainer.use_reference_policy = False
    trainer.config = _Cfg(
        algorithm=_Cfg(opd=opd),
        actor_rollout_ref=_Cfg(actor=actor),
        data=_Cfg(task_balance=_Cfg(per_task_batch_size=15)),
        env=_Cfg(rollout=_Cfg(n=8)),
    )
    trainer._load_offpolicy_data = lambda: None
    with patch.object(RayPPOTrainer, "__init__", lambda self, *a, **k: None):
        OffPolicyOPDRayTrainer.__init__(trainer)
    return trainer


def test_the_default_arm_loads_no_teacher():
    """Off is the arm the pool was built for: the top-k is on disk and Stage 2
    runs no teacher at all. Nothing about this port may change that."""
    trainer = _trainer_init(student_indexed_topk=False)
    assert trainer.student_indexed_topk is False
    assert trainer.teacher_wg == {}
    assert not hasattr(trainer, "teacher_paths")


def test_turning_it_on_without_teachers_refuses_to_start():
    """The failure this prevents is a run that starts, reads a cache nobody
    filled, and trains against zeros for forty hours."""
    with pytest.raises(AssertionError, match="teacher_paths"):
        _trainer_init(student_indexed_topk=True)


def test_turning_it_on_with_teachers_registers_them_by_task():
    trainer = _trainer_init(
        student_indexed_topk=True,
        teacher_paths={"alfworld": "/ckpt/a", "search": "/ckpt/s", "webshop": "/ckpt/w"},
    )
    assert trainer.student_indexed_topk is True
    assert sorted(trainer.teacher_paths) == ["alfworld", "search", "webshop"]


def test_a_single_token_kl_type_leaves_the_arm_off():
    """student_indexed_topk only means anything for the dense top-k KL: there is
    no support to choose when the estimator reads one sampled token. Silently
    loading three teachers for a loss that cannot use them would be the worst of
    both outcomes."""
    trainer = _init(
        _Cfg(teacher_data_dir="/pool", student_indexed_topk=True),
        _Cfg(teacher_kl_loss_type="low_var_kl", teacher_kl_topk=20, student_indexed_topk=True),
    )
    assert trainer.student_indexed_topk is False
    assert trainer.teacher_wg == {}


def test_the_injection_sets_the_ref_flag_and_does_not_leave_it_to_the_interpolation():
    """The bug this pins killed every run at step 1.

    ppo_trainer.yaml spells ref.student_indexed_topk as an interpolation of the
    actor's value, but main_opd_offpolicy calls OmegaConf.resolve BEFORE its own
    injection block -- so the interpolation freezes at the actor's PRE-injection
    value. The teacher would then return a top-k nobody reads while the actor
    waited on a cache nobody filled. It fails loudly (every row unowned on the
    first mini-batch) rather than quietly, which is the only reason it is a bug
    and not a disaster.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "algorithm": {"opd": {"student_indexed_topk": True}},
            "actor_rollout_ref": {
                "actor": {"student_indexed_topk": False},
                "ref": {"student_indexed_topk": "${actor_rollout_ref.actor.student_indexed_topk}"},
            },
        }
    )
    # Exactly the order main_opd_offpolicy runs them in.
    OmegaConf.resolve(cfg)
    assert cfg.actor_rollout_ref.ref.student_indexed_topk is False, (
        "test premise: resolve() froze ref at the actor's pre-injection value"
    )

    want = bool(cfg.algorithm.opd.student_indexed_topk)
    cfg.actor_rollout_ref.actor.student_indexed_topk = want
    # Without the explicit second assignment the next line is False, and the run
    # dies at step 1.
    cfg.actor_rollout_ref.ref.student_indexed_topk = want
    assert cfg.actor_rollout_ref.ref.student_indexed_topk is True


def test_main_assigns_both_halves():
    """Source-level, because the failure is an omission: the injection block has
    to name ref explicitly, and an edit that deletes the line would otherwise
    only show up on a GPU."""
    import inspect

    from verl.trainer import main_opd_offpolicy

    src = inspect.getsource(main_opd_offpolicy)
    assert "config.actor_rollout_ref.actor.student_indexed_topk = student_indexed_topk" in src
    assert "config.actor_rollout_ref.ref.student_indexed_topk = student_indexed_topk" in src


def test_the_forward_refuses_return_lse_on_the_student_path():
    """Teacher mode returns a pair and the caching path unpacks one; student mode
    returns a single tensor. Asking for both would silently drop the student's
    values, so it is refused instead."""
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.use_remove_padding = True
    actor.use_fused_kernels = False
    actor.device_name = "cpu"

    with pytest.raises(NotImplementedError, match="topk_ids"):
        DataParallelPPOActor._forward_micro_batch(
            actor, micro_batch=None, temperature=1.0, topk_ids=torch.zeros(1, 1, K, dtype=torch.long),
            return_lse=True,
        )


# --------------------------------------------------------------------------- #
# 2. the oracle: Stage 1's recorded top-k
# --------------------------------------------------------------------------- #


def test_the_cache_reproduces_the_top_k_stage_1_recorded():
    """The off-policy arm's own end-to-end check.

    Stage 1 wrote ``teacher_topk_logprobs`` at ``teacher_topk_ids`` for exactly
    these rows. Stage 2's teacher caches hidden states instead, and the actor
    resolves them at whatever ids it is handed. Hand it the RECORDED ids and the
    answer must be the recorded values -- which pins the whole route (packing,
    keying, the exchange, the narrow GEMM) against a reference this session did
    not produce.
    """
    W, h, lse, want_lp, want_ids = _stage1(seed=11, rows=10, positions=4)

    cache = TeacherHiddenCache()
    cache.register_lm_head("alfworld", W)
    keys = torch.arange(10, dtype=torch.long)
    cache.put(keys, "alfworld", h, lse)

    got = exchange_teacher_logprobs(cache, keys, want_ids, world_size=1)
    torch.testing.assert_close(got, want_lp, rtol=0, atol=1e-5)


def test_the_oracle_fails_if_the_rows_are_shifted_against_their_keys():
    """The same check, with the pool's rows off by one against the cache: a
    plausible number for the wrong sample is exactly the failure the fingerprint
    and the witness exist to make loud, so the oracle has to notice it too."""
    W, h, lse, want_lp, want_ids = _stage1(seed=12, rows=10, positions=4)

    cache = TeacherHiddenCache()
    cache.register_lm_head("alfworld", W)
    keys = torch.arange(10, dtype=torch.long)
    cache.put(keys, "alfworld", h, lse)

    shifted = torch.roll(want_ids, shifts=1, dims=0)
    got = exchange_teacher_logprobs(cache, keys, shifted, world_size=1)
    assert (got - torch.roll(want_lp, shifts=1, dims=0)).abs().max() > 1.0


def test_rows_of_different_teachers_each_reproduce_their_own_recording():
    """A step's batch mixes tasks -- _balance_batch sorts by token count, not by
    teacher -- so one lookup answers rows belonging to three different models.
    Each must come back with ITS teacher's recorded values."""
    Wa, ha, lsea, lpa, idsa = _stage1(seed=1, rows=5, positions=3)
    Wb, hb, lseb, lpb, idsb = _stage1(seed=2, rows=5, positions=3)

    cache = TeacherHiddenCache()
    cache.register_lm_head("alfworld", Wa, slot=0, n_tasks=2)
    cache.register_lm_head("webshop", Wb, slot=1, n_tasks=2)
    cache.put(torch.arange(0, 5), "alfworld", ha, lsea)
    cache.put(torch.arange(5, 10), "webshop", hb, lseb)

    keys = torch.arange(10, dtype=torch.long)
    got = exchange_teacher_logprobs(cache, keys, torch.cat([idsa, idsb]), world_size=1)
    torch.testing.assert_close(got, torch.cat([lpa, lpb]), rtol=0, atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. what the two supports actually are
# --------------------------------------------------------------------------- #


def _full_reverse_kl(s_logsm, t_logsm):
    """The exact per-token KL(p_s || p_t) over the whole vocabulary."""
    return (s_logsm.exp() * (s_logsm - t_logsm)).sum(-1)


def test_both_supports_are_lower_bounds_on_the_same_full_kl():
    """The identity the arm rests on, checked against an exact reference.

    ``KL_full - KL_A = tail_s * KL(p_s|A-bar || p_t|A-bar) >= 0`` holds for ANY
    support A, so switching which top-k defines A cannot turn the objective into
    something that is not a bound on the same quantity. That -- not a claim that
    one is always numerically larger -- is what makes this a tighter bound rather
    than a different loss.
    """
    from verl.trainer.ppo.core_algos import topk_kl_per_token

    for seed in range(8):
        W, h, _, _, _ = _stage1(seed=100 + seed, rows=16, positions=1)
        g = torch.Generator().manual_seed(seed)
        t_logsm = torch.log_softmax((h @ W.T).squeeze(1), dim=-1)
        s_logsm = torch.log_softmax(
            (h @ W.T).squeeze(1) + 2.0 * torch.randn((16, VOCAB), generator=g), dim=-1
        )
        t_lp, t_ids = torch.topk(t_logsm, K, dim=-1)
        s_lp, s_ids = torch.topk(s_logsm, K, dim=-1)

        teacher_indexed = topk_kl_per_token(
            s_logsm.gather(-1, t_ids).unsqueeze(1), t_lp.unsqueeze(1)
        ).squeeze(1)
        student_indexed = topk_kl_per_token(
            s_lp.unsqueeze(1), t_logsm.gather(-1, s_ids).unsqueeze(1)
        ).squeeze(1)
        full = _full_reverse_kl(s_logsm, t_logsm)

        assert torch.all(teacher_indexed >= -1e-5)
        assert torch.all(student_indexed >= -1e-5)
        assert torch.all(teacher_indexed <= full + 1e-4)
        assert torch.all(student_indexed <= full + 1e-4)


def test_the_student_indexed_support_covers_more_of_the_student_s_mass():
    """The mechanism, stated as the thing that IS exactly true.

    What the bound drops is the residual weighted by ``tail_s``, the student mass
    outside the support. The student's own top-k maximises the covered student
    mass by construction, so ``tail_s`` can only shrink. That is the whole reason
    to prefer it -- and it is a different statement from "the resulting KL is
    always larger", which does not follow: the residual is ``tail_s`` times a
    conditional KL, and only the first factor is controlled here.
    """
    for seed in range(8):
        W, h, _, _, _ = _stage1(seed=200 + seed, rows=16, positions=1)
        g = torch.Generator().manual_seed(seed)
        t_logsm = torch.log_softmax((h @ W.T).squeeze(1), dim=-1)
        s_logsm = torch.log_softmax(
            (h @ W.T).squeeze(1) + 2.0 * torch.randn((16, VOCAB), generator=g), dim=-1
        )
        _, t_ids = torch.topk(t_logsm, K, dim=-1)
        _, s_ids = torch.topk(s_logsm, K, dim=-1)

        covered_by_teacher = s_logsm.exp().gather(-1, t_ids).sum(-1)
        covered_by_student = s_logsm.exp().gather(-1, s_ids).sum(-1)
        assert torch.all(covered_by_student >= covered_by_teacher - 1e-6)
    # And on this mixture it is a real difference, not a tie.
    assert float((covered_by_student - covered_by_teacher).max()) > 1e-3


def test_the_two_supports_genuinely_differ_on_disagreeing_models():
    """If they coincided, the arm would be a no-op with three teachers attached."""
    W, h, _, _, _ = _stage1(seed=7, rows=16, positions=1)
    g = torch.Generator().manual_seed(3)
    t_logsm = torch.log_softmax((h @ W.T).squeeze(1), dim=-1)
    s_logsm = torch.log_softmax(
        (h @ W.T).squeeze(1) + 3.0 * torch.randn((16, VOCAB), generator=g), dim=-1
    )
    _, t_ids = torch.topk(t_logsm, K, dim=-1)
    _, s_ids = torch.topk(s_logsm, K, dim=-1)
    overlap = (t_ids.unsqueeze(-1) == s_ids.unsqueeze(-2)).any(-1).float().mean()
    assert overlap < 0.5


# --------------------------------------------------------------------------- #
# 4. the pool columns this arm stops reading
# --------------------------------------------------------------------------- #


def test_the_student_indexed_arm_drops_the_recorded_top_k_at_load():
    """The largest thing in a row, kept only by the arm that trains on it.

    teacher_topk_logprobs + teacher_topk_ids are ~82 KB of the ~123 KB a row costs
    resident -- about 105 GB across the pool, on a box the Stage-2 profile measured
    at 494/503 GB. The teacher-indexed arm's loss IS those two columns. The
    student-indexed arm never selects them (the support is the student's own
    top-k), so carrying them buys nothing.
    """
    from verl.trainer.ppo.opd_offpolicy_ray_trainer import (
        _KD_TARGET_TENSOR_KEYS,
        OffPolicyOPDRayTrainer,
    )

    control = _trainer_init(student_indexed_topk=False)
    arm = _trainer_init(
        student_indexed_topk=True,
        teacher_paths={"alfworld": "/ckpt/a", "search": "/ckpt/s", "webshop": "/ckpt/w"},
    )

    kept = control._resolve_drop_tensor_keys()
    dropped = arm._resolve_drop_tensor_keys()
    for key in _KD_TARGET_TENSOR_KEYS:
        assert key not in kept, "the teacher-indexed arm trains on this column"
        assert key in dropped
    # Nothing else moved: the class-level dead set is still the floor for both.
    assert set(OffPolicyOPDRayTrainer._drop_tensor_keys) <= set(kept)
    assert set(kept) < set(dropped)


def test_the_dropped_columns_are_exactly_the_ones_update_policy_stops_selecting():
    """The two decisions have to agree, and they are made in different files.

    Dropping a column the loss then selects dies at the first micro-batch; keeping
    one it does not costs 105 GB for nothing. Both are driven by
    student_indexed_topk, so pin that they read the same names.
    """
    import inspect

    from verl.trainer.ppo.opd_offpolicy_ray_trainer import _KD_TARGET_TENSOR_KEYS
    from verl.workers.actor import dp_actor

    # The module, not update_policy: GPUMemoryLogger wraps that method without
    # functools.wraps, so getsource on it returns the decorator's body.
    src = inspect.getsource(dp_actor)
    # One place selects the recorded top-k, and it is the teacher-indexed branch;
    # the student-indexed branch selects the cache key instead. A read added
    # outside that branch changes this count, and the drop above stops being safe.
    selected = 'select_keys += ["teacher_topk_logprobs", "teacher_topk_ids"]'
    assert src.count(selected) == 1
    assert src.count('select_keys.append("teacher_cache_ids")') == 1
    for key in _KD_TARGET_TENSOR_KEYS:
        assert key in selected
    # And the two remaining uses are both inside the teacher-indexed else-branch.
    assert src.count('data["teacher_topk_ids"]') == 1
    assert src.count('data["teacher_topk_logprobs"]') == 1


def test_keeping_them_stays_available_for_the_cross_check():
    """The retain flag is the only reason to hold them in this arm, and the reason
    is the oracle above -- so it has to actually be reachable."""
    from verl.trainer.ppo import opd_offpolicy_ray_trainer as mod

    arm = _trainer_init(
        student_indexed_topk=True,
        teacher_paths={"alfworld": "/ckpt/a", "search": "/ckpt/s", "webshop": "/ckpt/w"},
    )
    original = mod._KEEP_KD_TARGETS
    try:
        mod._KEEP_KD_TARGETS = True
        assert set(arm._resolve_drop_tensor_keys()) == set(arm._drop_tensor_keys)
    finally:
        mod._KEEP_KD_TARGETS = original


def test_the_loader_still_defaults_to_the_class_attribute():
    """The subclass extension point the SFT arm established keeps working: a
    caller that says nothing gets cls._drop_tensor_keys, unchanged."""
    import inspect

    from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer

    sig = inspect.signature(OffPolicyOPDRayTrainer._load_offpolicy_file)
    assert sig.parameters["drop_keys"].default is None
    src = inspect.getsource(OffPolicyOPDRayTrainer._load_offpolicy_file)
    assert "drop_keys = cls._drop_tensor_keys" in src
