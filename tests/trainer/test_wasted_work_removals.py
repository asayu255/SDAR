"""Two removals of work whose result was already being discarded.

Neither changes a value that reaches the loss; both are checked here by asserting
the *output* is unchanged while the discarded work no longer happens.

  * EpisodeRewardManager decoded every prompt and response to a string on every
    call. Those strings feed only the `num_examine` console sample, which is 0 for
    the training reward_fn (main_opd_grpo builds it that way), so on the multitask
    batch that was ~2 * 7k discarded decodes per step and most of the manager's
    runtime. The score is the env's episode reward and never reads the text.

  * generate_sequences called empty_cache once per turn even inside a rollout
    session, forcing a device synchronize (~50x per rollout) to hand back blocks
    the allocator immediately re-acquired. It cannot free vLLM's KV cache either --
    the engine owns that, and empty_cache only returns *unused* cached blocks.

Deliberately NOT ported from pure OPD: rollout.return_rollout_log_probs=False.
That arm drops the column because nothing reads it; this recipe runs the
rollout-vs-actor drift check in RayPPOTrainer.fit, which is its only consumer, so
the column has to keep being built. The last test pins that the consumer is still
there, since it is the whole reason for the divergence.

CPU-only.
"""

import ast
import pathlib

import pytest

torch = pytest.importorskip("torch")

try:
    from agent_system.reward_manager.episode import EpisodeRewardManager
    from verl import DataProto
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[2]


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
# empty_cache inside a rollout session
# --------------------------------------------------------------------------- #
def _function_def(path, class_name, func_name):
    tree = ast.parse(pathlib.Path(path).read_text())
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == class_name:
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == func_name:
                    return fn
    raise AssertionError(f"{class_name}.{func_name} not found in {path}")


def _empty_cache_calls(node):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "empty_cache"
    ]


def test_generate_sequences_only_empties_the_cache_outside_a_session():
    """Read structurally rather than by running it: standing up the worker needs
    FSDP, vLLM and a GPU, and the invariant being protected is a static one -- no
    reachable empty_cache in this method may sit outside the session guard. An
    unguarded one costs a forced synchronize per turn and frees nothing, because
    the vLLM engine owns its KV cache.
    """
    fn = _function_def(_REPO / "verl/workers/fsdp_workers.py", "ActorRolloutRefWorker", "generate_sequences")

    all_calls = _empty_cache_calls(fn)
    assert all_calls, "the call vanished entirely; the non-session path still needs it"

    guarded = []
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and "_rollout_session_active" in ast.dump(node.test):
            guarded.extend(_empty_cache_calls(node))

    assert len(guarded) == len(all_calls), (
        "every empty_cache in generate_sequences must sit under a "
        "_rollout_session_active guard; found an unguarded one"
    )


def test_the_session_boundary_still_empties_the_cache():
    """The call is moved to the wake/sleep boundary, not deleted: end_rollout_session
    exits the sharding manager, whose __exit__ is what empties it."""
    fn = _function_def(_REPO / "verl/workers/fsdp_workers.py", "ActorRolloutRefWorker", "end_rollout_session")
    exits = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "__exit__"
    ]
    assert exits, "end_rollout_session must still close the sharding manager"


# --------------------------------------------------------------------------- #
# the divergence from pure OPD
# --------------------------------------------------------------------------- #
def test_this_arm_still_has_a_consumer_for_rollout_log_probs():
    """Why return_rollout_log_probs=False was not ported.

    Pure OPD drops the column because its thin loop has no old_log_prob phase to
    compare against. This recipe runs the drift check, so the column is read every
    step -- if this guard ever disappears, dropping the column becomes portable
    here too, and this test is where that gets noticed.
    """
    src = (_REPO / "verl/trainer/ppo/ray_trainer.py").read_text()
    assert 'if "rollout_log_probs" in batch.batch.keys():' in src
    assert "rollout_probs_diff" in src
