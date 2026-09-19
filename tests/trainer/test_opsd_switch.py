"""The OPD+GRPO trainer's self-distillation switch (algorithm.opsd).

An OPD+GRPO run can replace its external teacher with the SDAR teacher -- the
model itself with the task's skill documents in the prompt -- by setting the
external coefficient to 0 and algorithm.opsd.enable=True. What has to hold:

* the per-token SDAR term is the one the SDAR runs trained on (sdar_gated_kl is
  compute_sdar_loss before aggregation);
* under per-task normalisation it is aggregated by the same row weights as the
  policy gradient, so the actor may accept use_sdar_loss there (and still refuses
  the terms it cannot weight);
* the switch reaches the actor through inject_opd_grpo_config and nothing turns
  it on by default;
* the trainer builds the teacher batch from real prompts and real skill files.
"""

import ast
import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_the_gated_term_is_the_sdar_loss_before_aggregation():
    from verl.trainer.ppo.core_algos import agg_loss
    from verl.trainer.ppo.sdar_utils import compute_sdar_loss, sdar_gated_kl

    torch.manual_seed(0)
    student = torch.randn(4, 7, requires_grad=True)
    teacher = torch.randn(4, 7)
    mask = (torch.rand(4, 7) > 0.3).float()
    loss, metrics = compute_sdar_loss(student, teacher, mask, gate_beta=5.0, loss_agg_mode="token-mean")
    gated, gate, delta = sdar_gated_kl(student, teacher, 5.0)
    assert torch.allclose(loss, agg_loss(loss_mat=gated, loss_mask=mask, loss_agg_mode="token-mean"))
    assert torch.allclose(delta, teacher - student.detach())
    assert not gate.requires_grad and not delta.requires_grad
    # The gradient reaches the student only, and is -gate per token.
    (gated * mask).sum().backward()
    assert torch.allclose(student.grad, -gate * mask)


def test_the_weighted_path_is_the_row_weighted_sum_of_the_same_term():
    from verl.trainer.ppo.core_algos import agg_loss_by_task_weights
    from verl.trainer.ppo.sdar_utils import sdar_gated_kl

    torch.manual_seed(1)
    student, teacher = torch.randn(5, 6), torch.randn(5, 6)
    mask = (torch.rand(5, 6) > 0.2).float()
    w = torch.rand(5)
    gated, _, _ = sdar_gated_kl(student, teacher, 5.0)
    expected = sum(float(w[i]) * float((gated[i] * mask[i]).sum()) for i in range(5))
    assert abs(float(agg_loss_by_task_weights(gated, mask, w)) - expected) < 1e-5


def _task_weighting_cfg(**mutation):
    return OmegaConf.create(
        {
            "use_dynamic_bsz": False,
            "ppo_epochs": 1,
            "loss_agg_mode": "token-mean",
            "policy_loss": {"loss_mode": "vanilla"},
            "use_kl_loss": False,
            "use_sdl_loss": False,
            "use_sdar_loss": False,
            **mutation,
        }
    )


def test_per_task_weighting_accepts_the_sdar_term():
    from verl.workers.actor.dp_actor import check_task_weighting_supported

    check_task_weighting_supported(
        _task_weighting_cfg(use_sdar_loss=True), use_teacher_kl_loss=True, ulysses_sequence_parallel_size=1
    )


@pytest.mark.parametrize("key", ["use_kl_loss", "use_sdl_loss"])
def test_per_task_weighting_still_refuses_the_terms_it_does_not_weight(key):
    from verl.workers.actor.dp_actor import check_task_weighting_supported

    with pytest.raises(AssertionError, match=key):
        check_task_weighting_supported(
            _task_weighting_cfg(**{key: True}), use_teacher_kl_loss=True, ulysses_sequence_parallel_size=1
        )


def test_update_policy_aggregates_the_sdar_term_by_the_row_weights():
    """The branch that makes accepting use_sdar_loss safe: without it the SDAR
    term would be a token-mean added to a row-weighted sum."""
    from verl.workers.actor import dp_actor

    # update_policy is wrapped by a memory logger, so inspect.getsource would
    # return the wrapper; read the definition out of the module file instead.
    tree = ast.parse(open(dp_actor.__file__).read())
    (fn,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "update_policy"]
    text = ast.unparse(fn)
    assert "sdar_gated_kl(log_prob, teacher_log_probs, _sdar_beta)" in text
    assert "sdar_term = _task_agg(_sdar_tok)" in text
    assert "policy_loss = policy_loss + sdar_term * sdar_coef" in text


def test_the_switch_reaches_the_actor_and_is_off_by_default():
    from hydra import compose, initialize_config_dir

    from verl.trainer.main_opd_grpo import inject_opd_grpo_config

    base = [
        "+algorithm.opd.kl_loss_coef=0.01",
        "+algorithm.opd.kl_loss_type=topk_kl",
        "+algorithm.opd.topk=20",
        "actor_rollout_ref.actor.pg_loss_coef=1.0",
    ]
    with initialize_config_dir(config_dir=os.path.join(REPO, "verl/trainer/config"), version_base=None):
        off = compose(config_name="ppo_trainer", overrides=base)
        on = compose(config_name="ppo_trainer", overrides=base + ["algorithm.opsd.enable=True"])
    inject_opd_grpo_config(off)
    inject_opd_grpo_config(on)
    assert off.actor_rollout_ref.actor.use_sdar_loss is False
    assert on.actor_rollout_ref.actor.use_sdar_loss is True
    assert on.actor_rollout_ref.actor.sdar_loss_coef == 0.01
    assert on.actor_rollout_ref.actor.sdar_gate_beta == 5.0
    # The external teacher is untouched: its coefficient is the run's choice.
    assert on.actor_rollout_ref.actor.teacher_kl_loss_coef == 0.01
    assert on.actor_rollout_ref.actor.use_teacher_kl_loss is True


def _fake_batch(tokenizer, prompts, tasks, response_length=8, max_prompt_length=4096):
    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask

    rows_ids, rows_mask = [], []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        pad = max_prompt_length - len(ids)
        rows_ids.append([tokenizer.pad_token_id] * pad + ids + [tokenizer.eos_token_id] * response_length)
        rows_mask.append([0] * pad + [1] * len(ids) + [1] * response_length)
    input_ids = torch.tensor(rows_ids)
    attention_mask = torch.tensor(rows_mask)
    batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "responses": input_ids[:, -response_length:],
        },
        non_tensors={
            "task_name": np.array(tasks, dtype=object),
            "data_source": np.array(tasks, dtype=object),
            "gamefile": np.array([""] * len(tasks), dtype=object),
        },
    )
    return batch


def test_the_trainer_builds_the_skill_prefixed_teacher_from_the_real_skill_files():
    from transformers import AutoTokenizer

    from verl import DataProto
    from verl.trainer.ppo.opd_grpo_ray_trainer import OPDGRPORayTrainer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    prompts = [
        "<|im_start|>user\nYour task is to: put a clean mug in coffeemachine.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nYou are an expert autonomous agent operating in the WebShop e-commerce environment. "
        "Your task is to: Find me a red dress.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nYour question: who wrote hamlet?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nYour task is to: heat some egg and put it in countertop.<|im_end|>\n<|im_start|>assistant\n",
    ]
    tasks = ["alfworld", "webshop", "search", "alfworld"]
    batch = _fake_batch(tok, prompts, tasks)
    seen = {}

    class _WG:
        def compute_log_prob(self, data):
            seen["batch"] = data
            n, r = data.batch["responses"].shape
            return DataProto.from_dict(tensors={"old_log_probs": torch.full((n, r), -1.5), "entropys": torch.zeros(n, r)})

    class _Self:
        pass

    me = _Self()
    me.config = OmegaConf.create({
        "algorithm": {"opd": {"kl_loss_type": "topk_kl"}},
        "data": {"max_prompt_length": 4096, "truncation": "left"},
    })
    me.tokenizer = tok
    me.actor_rollout_wg = _WG()
    cfg = OmegaConf.create({
        "enable": True, "skills_dir": os.path.join(REPO, "skills/alfworld"), "skill_all": False,
        "skills_dirs": {t: os.path.join(REPO, f"skills/{t}") for t in ("alfworld", "webshop", "search")},
    })
    metrics = {}
    out = OPDGRPORayTrainer._compute_self_teacher_log_probs(me, batch, cfg, metrics)
    assert out.shape == batch.batch["responses"].shape and torch.all(out == -1.5)
    tb = seen["batch"]
    # Same responses, longer prompts carrying the skill header, nothing capped.
    assert torch.equal(tb.batch["responses"], batch.batch["responses"])
    assert metrics["opsd/prompt_tokens_added/mean"] > 100
    assert metrics["opsd/prompt_capped_ratio"] == 0.0
    first = tb.batch["input_ids"][0][tb.batch["attention_mask"][0].bool()]
    assert tok.decode(first).startswith("[Privileged Skill Information]")


def test_the_trainer_refuses_the_single_token_opd_estimator():
    from verl.trainer.ppo.opd_grpo_ray_trainer import OPDGRPORayTrainer

    class _Self:
        pass

    me = _Self()
    me.config = OmegaConf.create({"algorithm": {"opd": {"kl_loss_type": "low_var_kl"}}})
    with pytest.raises(AssertionError, match="top-k"):
        OPDGRPORayTrainer._compute_self_teacher_log_probs(me, None, OmegaConf.create({}), {})
