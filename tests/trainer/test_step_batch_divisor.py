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
"""The step batch stops being padded for a phase this arm does not run.

``batch_size_divisor`` is the lcm over every worker call the ON-POLICY pipeline
makes with a batch, ``compute_log_prob`` included. The off-policy loop makes none
of those except ``update_actor``: it has no ``old_log_prob`` phase, and validation
sets ``recompute_log_prob=False``. On the 3-GPU multitask run
``rollout.log_prob_micro_batch_size_per_gpu=16`` therefore does exactly one thing
-- it puts 48 into that lcm and drags the divisor to 240 -- and adjust_batch
duplicates rows up to the next multiple: a measured 122.4 a step against 4260.8
real ones, 2.77% of the batch, which the SFT loss then weights at zero (wandb
x7g9r7bx).

The weighted SFT path can pad to the micro batch instead, because a short final
mini-batch is already correct for it. The unweighted path cannot, so this is an
SFT-arm change and these tests pin that boundary.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from omegaconf import OmegaConf

    from agent_system.multi_turn_rollout import adjust_batch, batch_size_divisor
    from verl import DataProto
    from verl.trainer.ppo.sft_multitask_ray_trainer import MultiTaskSFTTrainer
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _config(use_sft_loss=True, micro=5, log_prob_micro=16, n_gpus=3, use_dynamic_bsz=False):
    """The knobs the divisor is derived from, at the run script's values."""
    return OmegaConf.create({
        "actor_rollout_ref": {
            "actor": {
                "use_sft_loss": use_sft_loss,
                "ppo_micro_batch_size_per_gpu": micro,
                "ppo_mini_batch_size": 60,
                "use_dynamic_bsz": use_dynamic_bsz,
                "use_kl_loss": False,
            },
            "rollout": {"log_prob_micro_batch_size_per_gpu": log_prob_micro},
            "ref": {"log_prob_micro_batch_size_per_gpu": log_prob_micro},
        },
        "algorithm": {"use_kl_in_reward": False},
        "trainer": {"n_gpus_per_node": n_gpus, "nnodes": 1},
    })


def _divisor(**kwargs):
    trainer = object.__new__(MultiTaskSFTTrainer)
    trainer.config = _config(**kwargs)
    return MultiTaskSFTTrainer._step_batch_divisor(trainer)


def test_the_sft_arm_pads_to_whole_micro_batches():
    """5 rows per GPU across 3 GPUs, not the 240 the on-policy lcm asks for."""
    assert _divisor() == 15
    # What it replaces, from the same config.
    assert batch_size_divisor(_config()) == 240


def test_the_unweighted_path_keeps_the_full_divisor():
    """Without the per-row weights there is no cancellation: update_policy divides
    a short mini-batch by the CONFIGURED gradient_accumulation and nothing
    multiplies it back, so that mini-batch's gradient really would be mis-scaled."""
    assert _divisor(use_sft_loss=False) is None


def test_token_sized_micro_batches_keep_the_full_divisor():
    """use_dynamic_bsz forms micro batches by token count, so a row-count divisor
    says nothing about how they land."""
    assert _divisor(use_dynamic_bsz=True) is None


def test_a_missing_per_gpu_micro_size_keeps_the_full_divisor():
    assert _divisor(micro=None) is None


def test_the_divisor_follows_the_gpu_count():
    assert _divisor(n_gpus=2) == 10
    assert _divisor(n_gpus=3, micro=10) == 30


def test_it_still_divides_by_the_dp_world_size():
    """_balance_batch partitions with equal_size=True and the dispatch chunks by
    the world size; whatever this returns has to stay a multiple of it."""
    for n_gpus in (1, 2, 3, 4, 8):
        for micro in (1, 2, 5, 10):
            assert _divisor(n_gpus=n_gpus, micro=micro) % n_gpus == 0


def _batch(n_rows):
    return DataProto.from_dict(
        tensors={"input_ids": torch.zeros((n_rows, 4), dtype=torch.long)},
        non_tensors={"task_name": np.array(["alfworld"] * n_rows, dtype=object)},
    )


@pytest.mark.parametrize("n_real", [3702, 4260, 4383, 4774])
def test_padding_drops_to_the_micro_batch_remainder(n_real):
    """Measured row counts from the completed run. The padding a step pays is the
    remainder against the divisor, so shrinking the divisor is the whole saving."""
    config = _config()
    np.random.seed(0)
    old = adjust_batch(config, _batch(n_real))
    np.random.seed(0)
    new = adjust_batch(config, _batch(n_real), size_divisor=_divisor())

    assert len(old) % 240 == 0
    assert len(new) % 15 == 0
    assert len(new) - n_real < 15
    assert len(new) <= len(old)
    # Every real row is still there, in order: padding is appended copies.
    assert len(new) >= n_real


def test_an_explicit_divisor_that_already_divides_is_a_no_op():
    config = _config()
    batch = _batch(30)

    out = adjust_batch(config, batch, size_divisor=15)

    assert out is batch  # returned untouched, not copied


def test_the_default_divisor_is_unchanged_when_nothing_is_passed():
    """Every other caller of adjust_batch must see exactly what it saw before."""
    config = _config()
    np.random.seed(0)
    explicit = adjust_batch(config, _batch(1000), size_divisor=batch_size_divisor(config))
    np.random.seed(0)
    default = adjust_batch(config, _batch(1000))

    assert len(explicit) == len(default) == 1200
