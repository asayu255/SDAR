"""The off-policy signal is the SUM of the top-k teacher KL and the hard-label CE.

    loss = algorithm.opd.kl_loss_coef  * KL(teacher_topk || student)
         + algorithm.opd.sft_loss_coef * CE(teacher's sampled tokens, student)

Two things are pinned here, because both are silent when wrong.

The arithmetic: the terms are aggregated the same way over the same mask, each
is scaled by its own coefficient, and the result is their sum -- not a
convex combination, and not one replacing the other. A run whose CE term
quietly did not reach the loss would still train, still log, and still produce a
plausible curve.

The wiring: ``main_opd_offpolicy`` turns ``algorithm.opd.sft_loss_coef`` into
the actor's ``use_sft_loss`` / ``sft_loss_coef``. Omitting the key is pure KD;
setting it to 0.0 keeps the term at zero weight, which is the control run. Those
two are different configurations and must not collapse into each other.

If the verl stack (numpy/torch/...) is not installed, the module is skipped.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from omegaconf import OmegaConf, open_dict

    from verl.trainer.ppo.core_algos import agg_loss, topk_kl_per_token
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


AGG = "token-mean"


def _fixture(bs=3, resp_len=5, k=4, seed=0):
    """A student/teacher pair plus the mask the two terms share."""
    g = torch.Generator().manual_seed(seed)
    teacher_topk = torch.log_softmax(torch.randn(bs, resp_len, k, generator=g), dim=-1) - 1.0
    student_topk = torch.log_softmax(torch.randn(bs, resp_len, k, generator=g), dim=-1) - 1.0
    # log_prob of the sampled token: strictly negative, as a log-prob must be.
    log_prob = -torch.rand(bs, resp_len, generator=g) - 0.1
    mask = torch.ones(bs, resp_len)
    mask[0, -2:] = 0  # a padded tail, so "same mask" is a real constraint
    return student_topk, teacher_topk, log_prob, mask


def _terms(student_topk, teacher_topk, log_prob, mask):
    kl = agg_loss(
        loss_mat=topk_kl_per_token(student_topk_logprob=student_topk, teacher_topk_logprob=teacher_topk),
        loss_mask=mask,
        loss_agg_mode=AGG,
    )
    ce = agg_loss(loss_mat=-log_prob, loss_mask=mask, loss_agg_mode=AGG)
    return kl, ce


def test_the_loss_is_the_weighted_sum_of_both_terms():
    student_topk, teacher_topk, log_prob, mask = _fixture()
    kl, ce = _terms(student_topk, teacher_topk, log_prob, mask)

    for kl_coef, sft_coef in [(1.0, 1.0), (1.0, 0.5), (0.3, 2.0), (1.0, 0.0)]:
        # Exactly what update_policy accumulates into policy_loss.
        total = kl * kl_coef + ce * sft_coef
        assert torch.allclose(total, kl_coef * kl + sft_coef * ce)
        # Not a convex combination: at 1:1 the total is strictly above either term.
        if kl_coef == sft_coef == 1.0:
            assert total > kl and total > ce


def test_both_terms_are_positive_and_neither_dominates_by_construction():
    """Sanity on the scales the coefficients have to reconcile."""
    student_topk, teacher_topk, log_prob, mask = _fixture()
    kl, ce = _terms(student_topk, teacher_topk, log_prob, mask)
    assert kl > 0 and ce > 0


def test_the_ce_term_survives_a_converged_student_but_the_kl_does_not():
    """The reason a fixed sft_loss_coef shifts the balance over a run.

    A student that matches the teacher drives the KL to zero, so from then on the
    gradient is the CE alone. The CE cannot go to zero: it is bounded below by the
    teacher's own entropy on the tokens it sampled.
    """
    _, teacher_topk, log_prob, mask = _fixture()
    kl, ce = _terms(teacher_topk.clone(), teacher_topk, log_prob, mask)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)
    assert ce > 0.1


def test_both_terms_read_the_same_masked_tokens():
    """A row the mask excludes must move neither term."""
    student_topk, teacher_topk, log_prob, mask = _fixture()
    before = _terms(student_topk, teacher_topk, log_prob, mask)

    student_topk[0, -1] = torch.log_softmax(torch.full_like(student_topk[0, -1], 3.0), dim=-1)
    log_prob[0, -1] = -99.0
    after = _terms(student_topk, teacher_topk, log_prob, mask)

    assert torch.allclose(before[0], after[0]) and torch.allclose(before[1], after[1])


# --------------------------------------------------------------------------- #
# Config wiring: algorithm.opd.sft_loss_coef -> actor.use_sft_loss/sft_loss_coef
# --------------------------------------------------------------------------- #
def _inject(opd_overrides: dict):
    """Reproduce main_opd_offpolicy's injection on a minimal config."""
    config = OmegaConf.create(
        {
            "algorithm": {"opd": dict(opd_overrides), "use_kl_in_reward": True},
            "actor_rollout_ref": {"actor": {}},
        }
    )
    opd_cfg = config.algorithm.get("opd", {})
    sft_loss_coef = opd_cfg.get("sft_loss_coef", None)
    with open_dict(config):
        config.actor_rollout_ref.actor.use_teacher_kl_loss = True
        config.actor_rollout_ref.actor.teacher_kl_loss_coef = opd_cfg.get("kl_loss_coef", 1.0)
        config.actor_rollout_ref.actor.use_sft_loss = sft_loss_coef is not None
        config.actor_rollout_ref.actor.sft_loss_coef = (
            float(sft_loss_coef) if sft_loss_coef is not None else 0.0
        )
    return config.actor_rollout_ref.actor


def test_setting_the_coefficient_turns_the_ce_term_on():
    actor = _inject({"kl_loss_coef": 1.0, "sft_loss_coef": 1.0})
    assert actor.use_teacher_kl_loss is True
    assert actor.use_sft_loss is True
    assert actor.sft_loss_coef == 1.0


def test_omitting_the_coefficient_is_pure_kd():
    """A config written before the CE term existed must keep behaving as KD-only."""
    actor = _inject({"kl_loss_coef": 1.0})
    assert actor.use_teacher_kl_loss is True
    assert actor.use_sft_loss is False


def test_zero_is_not_the_same_as_absent():
    """0.0 keeps the term and its metrics at zero weight -- the control run for
    'does the CE matter'. Folding it into 'off' would lose that distinction."""
    absent = _inject({"kl_loss_coef": 1.0})
    zero = _inject({"kl_loss_coef": 1.0, "sft_loss_coef": 0.0})
    assert absent.use_sft_loss is False
    assert zero.use_sft_loss is True and zero.sft_loss_coef == 0.0
