"""batch_size_divisor is the number a pool's format hinges on, so it has a test."""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from omegaconf import OmegaConf
    from verl import DataProto
    from agent_system.multi_turn_rollout import adjust_batch, batch_size_divisor
except Exception as e:  # pragma: no cover
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _cfg(log_prob_micro, ppo_micro, n_gpus=3, nnodes=1, use_kl_loss=False, ref_micro=16):
    return OmegaConf.create({
        "trainer": {"n_gpus_per_node": n_gpus, "nnodes": nnodes},
        "algorithm": {"use_kl_in_reward": False},
        "actor_rollout_ref": {
            "rollout": {"log_prob_micro_batch_size_per_gpu": log_prob_micro},
            "ref": {"log_prob_micro_batch_size_per_gpu": ref_micro},
            "actor": {
                "use_kl_loss": use_kl_loss,
                "ppo_micro_batch_size_per_gpu": ppo_micro,
                "ppo_mini_batch_size": 60,
            },
        },
    })


@pytest.mark.parametrize(
    "log_prob_micro, ppo_micro, expected",
    [
        (16, 10, 240),   # the KD script's defaults
        (20, 10, 60),    # what reproduces the pool on disk
        (4, 10, 60),     # ...and so does this
        (10, 10, 30),
    ],
)
def test_divisor_is_the_lcm_over_the_world(log_prob_micro, ppo_micro, expected):
    assert batch_size_divisor(_cfg(log_prob_micro, ppo_micro)) == expected


def test_reference_micro_batch_only_counts_when_a_kl_term_reads_it():
    assert batch_size_divisor(_cfg(10, 10, use_kl_loss=False, ref_micro=7)) == 30
    assert batch_size_divisor(_cfg(10, 10, use_kl_loss=True, ref_micro=7)) == 210


def test_adjust_batch_pads_up_to_that_divisor():
    """The split-out helper has to be the number adjust_batch actually uses.

    Sizes stay above the divisor: adjust_batch draws its copies with
    replace=False, so it cannot pad a batch smaller than the number of rows it
    needs to add. A generation step is thousands of rows against a divisor of
    60, so that limit is never reached in practice.
    """
    config = _cfg(20, 10)  # divisor 60
    divisor = batch_size_divisor(config)
    for n in (61, 119, 120, 1019, 3093):
        data = DataProto.from_dict(
            tensors={"input_ids": torch.arange(n * 4).reshape(n, 4)},
            non_tensors={"traj_uid": np.array([f"t{i}" for i in range(n)], dtype=object)},
        )
        padded = adjust_batch(config, data)
        assert len(padded) % divisor == 0, n
        assert len(padded) == n if n % divisor == 0 else len(padded) > n
        # Padding only ever appends; the real rows keep their positions.
        assert padded.batch["input_ids"][:n].tolist() == data.batch["input_ids"].tolist()
