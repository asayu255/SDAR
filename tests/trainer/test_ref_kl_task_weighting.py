"""The reference KL (actor.use_kl_loss) under per-task loss normalisation (2026-09-26).

SDAR's recipe carries a reference KL (low_var_kl, 0.001); the OPD+GRPO recipe with per-task
normalisation refused it, because its branch aggregated the term by the plain token-mean while
every other term is a row-weighted sum. What has to hold now:
* the per-task check accepts use_kl_loss (and still refuses use_sdl_loss);
* update_policy aggregates the per-token KL by the same row weights (_task_agg), scaled by the
  configured coefficient, or by a per-row coefficient when the batch carries one;
* the row-weighted sum of the KL is what agg_loss_by_task_weights computes;
* without per-task weights the old token-mean path is unchanged.
"""
import ast
import os

import torch
from omegaconf import OmegaConf


def _cfg(**mutation):
    return OmegaConf.create({"use_dynamic_bsz": False, "ppo_epochs": 1, "loss_agg_mode": "token-mean",
                             "policy_loss": {"loss_mode": "vanilla"}, "use_kl_loss": False,
                             "use_sdl_loss": False, "use_sdar_loss": False, **mutation})


def test_per_task_weighting_accepts_the_reference_kl():
    from verl.workers.actor.dp_actor import check_task_weighting_supported
    check_task_weighting_supported(_cfg(use_kl_loss=True), use_teacher_kl_loss=True, ulysses_sequence_parallel_size=1)


def test_update_policy_aggregates_the_reference_kl_by_the_row_weights():
    from verl.workers.actor import dp_actor
    tree = ast.parse(open(dp_actor.__file__).read())
    (fn,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "update_policy"]
    text = ast.unparse(fn)
    assert "kl_term = _task_agg(kld) * self.config.kl_loss_coef" in text
    assert "kl_term = _task_agg(kld * kl_loss_coef.reshape(-1, 1).to(kld.dtype))" in text
    assert "policy_loss = policy_loss + kl_term" in text
    # the old token-mean path is still there for runs without per-task weights
    assert "policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef" in text


def test_the_weighted_kl_is_the_row_weighted_sum():
    from verl.trainer.ppo.core_algos import agg_loss_by_task_weights, kl_penalty
    torch.manual_seed(0)
    lp, ref = torch.randn(4, 6), torch.randn(4, 6)
    mask = (torch.rand(4, 6) > 0.3).float()
    w = torch.tensor([0.1, 0.2, 0.3, 0.4])
    kld = kl_penalty(logprob=lp, ref_logprob=ref, kl_penalty="low_var_kl")
    expected = float(((kld * mask).sum(-1) * w).sum())
    assert abs(float(agg_loss_by_task_weights(kld, mask, w)) - expected) < 1e-5
