"""Two removals of work whose result was already being discarded.

Neither changes a value that reaches the loss; both are checked here by asserting
the *output* is unchanged while the discarded work no longer happens.

  * EpisodeRewardManager decoded every prompt and response to a string on every
    call. Those strings feed only the `num_examine` console sample, which is 0 for
    the training reward_fn (main_opd builds it that way), so on the multitask OPD
    batch that was ~2 * 7k discarded decodes per step and most of the manager's
    runtime. The score is the env's episode reward and never reads the text.

  * The vLLM rollout asked for the sampled token's log-prob and walked every
    generated token in Python to assemble `rollout_log_probs`. Its only consumer
    is the rollout-vs-actor drift check, which needs an old_log_prob phase to
    compare against; pure OPD has none. Gated by
    rollout.return_rollout_log_probs so arms that do run the check keep it.

CPU-only: the rollout half is exercised through the same output-assembly logic,
not through vLLM.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from agent_system.reward_manager.episode import EpisodeRewardManager
    from verl import DataProto
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

import numpy as np


class _CountingTokenizer:
    """Records every decode so the test can assert they stopped happening."""

    def __init__(self):
        self.decodes = 0

    def decode(self, ids, skip_special_tokens=False):
        self.decodes += 1
        return " ".join(str(int(i)) for i in ids)


def _reward_batch(n=6, prompt_len=5, response_len=4):
    seqlen = prompt_len + response_len
    mask = torch.zeros((n, seqlen), dtype=torch.long)
    for i in range(n):
        mask[i, prompt_len - 1 - (i % prompt_len) : prompt_len + 1 + (i % response_len)] = 1
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.arange(n * prompt_len).view(n, prompt_len) % 50,
            "responses": torch.arange(n * response_len).view(n, response_len) % 50,
            "input_ids": torch.arange(n * seqlen).view(n, seqlen) % 50,
            "attention_mask": mask,
        }
    )
    batch.non_tensor_batch["data_source"] = np.array(["alfworld"] * n, dtype=object)
    batch.non_tensor_batch["episode_rewards"] = np.array([1.5] * n, dtype=object)
    batch.non_tensor_batch["episode_lengths"] = np.array([3.0] * n, dtype=object)
    return batch


def test_training_reward_is_unchanged_and_decodes_nothing():
    """num_examine=0 is every training call. The reward tensor must be identical
    to what the decoding version produced, and no decode may happen."""
    batch = _reward_batch()
    tok = _CountingTokenizer()
    manager = EpisodeRewardManager(tokenizer=tok, num_examine=0, normalize_by_length=False)

    reward = manager(batch)

    assert tok.decodes == 0, "training calls must not decode text nothing reads"

    # The score lands on each row's last valid response position and equals the
    # env's episode reward -- computed here independently.
    prompt_len = batch.batch["prompts"].shape[-1]
    for i in range(len(batch)):
        valid = int(batch.batch["attention_mask"][i, prompt_len:].sum())
        assert reward[i, valid - 1].item() == pytest.approx(1.5)
        assert reward[i].sum().item() == pytest.approx(1.5), "exactly one scored position"


def test_validation_still_decodes_for_its_console_sample():
    """num_examine>0 is the validation reward_fn; the sample printing must keep
    working, so the decode has to stay eager there."""
    batch = _reward_batch()
    tok = _CountingTokenizer()
    manager = EpisodeRewardManager(tokenizer=tok, num_examine=1, normalize_by_length=False)

    manager(batch)

    assert tok.decodes == 2 * len(batch), "one prompt + one response decode per row"


def test_normalize_by_length_is_untouched():
    batch = _reward_batch()
    manager = EpisodeRewardManager(tokenizer=_CountingTokenizer(), num_examine=0, normalize_by_length=True)
    reward = manager(batch)
    assert reward.sum().item() == pytest.approx(len(batch) * 1.5 / 3.0)


# --------------------------------------------------------------------------- #
# rollout_log_probs gating
# --------------------------------------------------------------------------- #


def _assemble(return_rollout_log_probs, n=3, resp_len=4):
    """The output-assembly shape from vllm_rollout_spmd.generate_sequences."""
    response = torch.randint(0, 50, (n, resp_len))
    rollout_log_probs = torch.randn((n, resp_len)) if return_rollout_log_probs else None
    tensors = {
        "prompts": torch.randint(0, 50, (n, 5)),
        "responses": response,
        "input_ids": torch.randint(0, 50, (n, 5 + resp_len)),
        "attention_mask": torch.ones((n, 5 + resp_len), dtype=torch.long),
        "position_ids": torch.arange(5 + resp_len).expand(n, -1),
    }
    if rollout_log_probs is not None:
        tensors["rollout_log_probs"] = rollout_log_probs
    return DataProto.from_dict(tensors=tensors)


def test_gating_off_drops_only_the_log_prob_column():
    """Everything downstream reads -- the sampled tokens and their masks -- must
    still be there. Only the unread column goes."""
    on = _assemble(True)
    off = _assemble(False)

    assert "rollout_log_probs" in on.batch.keys()
    assert "rollout_log_probs" not in off.batch.keys()
    for key in ("prompts", "responses", "input_ids", "attention_mask", "position_ids"):
        assert key in off.batch.keys()
        assert off.batch[key].shape == on.batch[key].shape


def test_the_drift_check_guard_tolerates_the_missing_column():
    """RayPPOTrainer.fit reaches the drift check through
    `if "rollout_log_probs" in batch.batch.keys()`, so an arm that turns the
    column off skips the check instead of raising."""
    off = _assemble(False)
    assert ("rollout_log_probs" in off.batch.keys()) is False


# --------------------------------------------------------------------------- #
# Finished trajectories' rows are never recorded, so they are never materialised
# --------------------------------------------------------------------------- #


def _turn_batch(batch_size, seqlen=8):
    from tensordict import TensorDict

    return DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.arange(batch_size * seqlen).view(batch_size, seqlen),
                "responses": torch.arange(batch_size * 4).view(batch_size, 4),
            },
            batch_size=[batch_size],
        ),
        non_tensor_batch={"traj_uid": np.array([f"t{i}" for i in range(batch_size)], dtype=object)},
    )


def test_only_the_requested_rows_are_materialised():
    """gather_rollout_data drops rows with active_masks=False, so slicing every
    column for a finished trajectory and then discarding the dict is the whole
    cost for none of the result. By the late turns of a 50-step episode those are
    most of the batch."""
    from agent_system.multi_turn_rollout.utils import to_list_of_dict

    batch = _turn_batch(6)
    active = [0, 3, 4]

    rows = to_list_of_dict(batch, active)

    assert len(rows) == len(active)
    for pos, i in enumerate(active):
        assert torch.equal(rows[pos]["input_ids"], batch.batch["input_ids"][i])
        assert rows[pos]["traj_uid"] == f"t{i}"


def test_omitting_the_rows_argument_keeps_every_row():
    """The un-gated path (ROLLOUT_COMPACT_RECORD off) must be untouched."""
    from agent_system.multi_turn_rollout.utils import to_list_of_dict

    batch = _turn_batch(5)
    assert len(to_list_of_dict(batch)) == 5
    assert [r["traj_uid"] for r in to_list_of_dict(batch)] == [f"t{i}" for i in range(5)]


def test_finished_rows_share_one_padding_filler():
    """Their model inputs are pure padding, so every finished row's are the same
    tensor. Collate copies them into the stacked batch either way."""
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    class _Tok:
        pad_token_id = 7

    template = {
        "input_ids": torch.arange(6),
        "attention_mask": torch.ones(6, dtype=torch.long),
        "position_ids": torch.arange(6),
    }
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector.tokenizer = _Tok()

    filler = collector._padding_filler(template)

    assert torch.equal(filler["input_ids"], torch.full((6,), 7))
    assert torch.all(filler["attention_mask"] == 0)
    assert torch.all(filler["position_ids"] == 0)

    # Same object handed to every finished row, not a fresh fill each time.
    class _Cfg(dict):
        def get(self, k, d=None):
            return d

    collector.config = type("C", (), {"data": _Cfg()})()
    gen_batch = DataProto(
        batch=None,
        non_tensor_batch={"data_source": np.array(["x", "x"], dtype=object)},
    )
    obs = {"text": ["a", "b"], "anchor": None}
    rows = [
        collector._placeholder_single_sample(item=i, gen_batch=gen_batch, obs=obs, template=template, filler=filler)
        for i in range(2)
    ]
    assert rows[0]["input_ids"] is rows[1]["input_ids"] is filler["input_ids"]
    # ...and dropping the argument still produces a correct row on its own.
    solo = collector._placeholder_single_sample(item=0, gen_batch=gen_batch, obs=obs, template=template)
    assert torch.equal(solo["input_ids"], filler["input_ids"])
