"""actor.pg_loss_norm: the policy gradient normalised per token (b) or per trajectory (a), on CPU.

(b) "token", the default: every term, the policy gradient included, is aggregated by the per-task
token weights num_mini_batches / (D * T_d) (task_loss_weights.py), T_d the task's response tokens
in the step. (a) "trajectory": the policy-gradient term alone takes num_mini_batches /
(D * N_d * L_d), N_d the task's real trajectories and L_d a fixed reference length, so the mean
optimizer step's policy gradient is

    g = (1/D) sum_d 1/(N_d L_d) sum_{i in d} sum_t A_it sum_u grad log pi(y_itu)

and the step's summed update num_mini_batches * g, num_mini_batches = ceil(rows / mini-batch) moving
with the sampled turn counts -- the part of the update neither normalisation fixes.

WHAT IS CHECKED, THROUGH THE PRODUCTION CODE. attach_task_loss_weights on the driver side and
DataParallelPPOActor.update_policy itself on the worker side -- select_keys, the mini/micro-batch
split, the per-term aggregation, the dp_world * grad_accum factor, the loss division by the
configured accumulation, the backward and the clip are all update_policy's own. Two things are
replaced: the model forward, by a table of per-token log-probs (so the gradient update_policy leaves
on the table IS the gradient w.r.t. each per-token log-prob), and FSDP's average of the ranks'
gradients, done here by hand on the per-step gradients the optimizer is handed. The table starts
at the rollout's log-probs and the stand-in optimizer never moves it, so every mini-batch sits at
ratio 1: the on-policy, pre-clip gradient the formula describes.

  * the mean optimizer-step policy gradient equals -(1/D) A_it / (N_d L_d) at every token under
    (a) and -(1/D) A_it / T_d under (b), to 1e-6, with adjust_batch's padding copies at exactly 0,
    uneven turn counts and response lengths, rows reordered after the weights are attached, one
    and two DP ranks and a short final mini-batch; the aggregated loss update_policy reports
    agrees with the same formula;
  * the step's SUM is num_mini_batches times the formula, in both modes, on layouts whose turn
    counts make the batch 2, 4 and 6 optimizer steps (task_loss/optimizer_steps reports the count
    under (a)); and in a one-task toy whose objective has gradient 0, enumerated exactly, (a)'s mean
    optimizer step has expectation 0 -- the guarantee -- while its step sum does not (towards the
    episodes with more turns), and (b)'s mean optimizer step does not either (the ratio estimator);
  * switching (b) -> (a) moves the policy-gradient term and nothing else: with the advantages at
    zero the teacher-KL, reference-KL and entropy gradients are bitwise identical, and with them
    live the difference is exactly the two formulas' difference;
  * token mode never reads the policy-gradient column: a batch carrying it trains bit for bit as
    one without;
  * the refusals (the launch's among them: inject_opd_grpo_config runs the config check), the
    geometry diagnostics' ratio, and the 2x2 launchers against their locks, every lock pinning its
    column.
"""

import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hydra")

from omegaconf import OmegaConf  # noqa: E402

try:
    from verl import DataProto
    from verl.trainer.ppo.task_loss_weights import (
        TASK_LOSS_WEIGHT_KEY,
        TASK_PG_LOSS_WEIGHT_KEY,
        attach_task_loss_weights,
        check_pg_loss_norm_config,
        pg_loss_norm_kwargs,
    )
    from verl.utils.debug import performance as _perf
    from verl.workers.actor import dp_actor
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO, "verl", "trainer", "config")

# --------------------------------------------------------------------------- #
# The step: three tasks, uneven turns and lengths, adjust_batch padding
# --------------------------------------------------------------------------- #
RESP, PROMPT = 6, 3
# Turns per trajectory, in rollout order: uneven within and across tasks (one search episode is a
# single turn).
TURNS = {"alfworld": [5, 3, 4, 2], "webshop": [3, 4, 2, 3], "search": [2, 3, 1, 2, 3]}
# Response tokens per turn row, drawn from these: alfworld's longest, search's shortest.
TOKEN_RANGE = {"alfworld": (3, 6), "webshop": (2, 5), "search": (1, 3)}
# L_d. Deliberately not the step's own mean tokens per trajectory, so (a) and (b) differ per task.
REF_TOKENS = {"alfworld": 17.0, "webshop": 9.0, "search": 3.5}
TASK_ID_NAMES = ["alfworld", "search", "webshop"]  # sorted, as _attach_task_ids numbers them
N_PAD = 3
# Rows per optimizer step counted globally (the driver's mini_batch_size), and per GPU per micro.
# 37 real + 3 padding = 40 rows: 4 optimizer steps of 12 + 12 + 12 + 4, i.e. a short final one,
# and 40 is a multiple of MICRO * world for world 1 and 2, as adjust_batch makes it.
MINI_GLOBAL = 12
MICRO = 2


def _n_real(turns=None):
    return sum(sum(v) for v in (TURNS if turns is None else turns).values())


N_REAL = _n_real()
N_STEPS = math.ceil((N_REAL + N_PAD) / MINI_GLOBAL)
# The same trajectories per task (N_d is fixed by the batch design, as on the cluster) with other turn
# counts, so the batch becomes another number of optimizer steps: 17 + 3 = 20 rows are 2 steps, the
# default 37 + 3 = 40 are 4, 64 + 4 = 68 are 6 (each total a multiple of MICRO * world for world 1, 2).
LAYOUTS = {
    "short": ({"alfworld": [2, 1, 3, 1], "webshop": [1, 2, 1, 1], "search": [1, 1, 1, 1, 1]}, 3),
    "default": (TURNS, N_PAD),
    "long": ({"alfworld": [9, 7, 8, 6], "webshop": [5, 6, 4, 5], "search": [3, 3, 2, 3, 3]}, 4),
}


def _step_batch(seed=0, *, zero_advantages=False, turns=None, n_pad=N_PAD):
    """One step's batch after adjust_batch: real turn rows, then padding copies of ``n_pad`` of them.

    ``input_ids[:, 0]`` carries a row id, which is how the stand-in forward finds a row's
    log-probs after the batch has been reordered and split. A padding copy has its own id (its own
    table row, with its original's values), so its gradient can be seen to be exactly zero.
    ``turns`` is the layout (TURNS unless given).
    """
    turns = TURNS if turns is None else turns
    n_real = _n_real(turns)
    rng = np.random.default_rng(seed)
    task, uid, tokens = [], [], []
    for name in ("alfworld", "webshop", "search"):
        lo, hi = TOKEN_RANGE[name]
        for j, n_turns in enumerate(turns[name]):
            for _ in range(n_turns):
                task.append(name)
                uid.append(f"{name}-{j}")
                tokens.append(int(rng.integers(lo, hi + 1)))
    assert len(task) == n_real
    src = list(range(n_real)) + sorted(rng.choice(n_real, n_pad, replace=False).tolist())
    bs = len(src)

    g = torch.Generator().manual_seed(seed)
    response_mask = torch.zeros(bs, RESP, dtype=torch.long)
    for i, s in enumerate(src):
        response_mask[i, : tokens[s]] = 1
    real_lp = -0.2 - torch.rand(n_real, RESP, generator=g, dtype=torch.float64)
    real_adv = torch.zeros(n_real, dtype=torch.float64) if zero_advantages else torch.randn(
        n_real, generator=g, dtype=torch.float64)
    real_teacher = -0.2 - torch.rand(n_real, RESP, generator=g, dtype=torch.float64)
    real_ref = -0.2 - torch.rand(n_real, RESP, generator=g, dtype=torch.float64)
    real_resp = torch.randint(0, 50, (n_real, RESP), generator=g)
    idx = torch.as_tensor(src)

    input_ids = torch.zeros(bs, PROMPT + RESP, dtype=torch.long)
    input_ids[:, 0] = torch.arange(bs)
    input_ids[:, PROMPT:] = real_resp[idx]
    attention_mask = torch.cat([torch.ones(bs, PROMPT, dtype=torch.long), response_mask], dim=1)
    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": torch.arange(PROMPT + RESP).repeat(bs, 1),
        "responses": real_resp[idx],
        "response_mask": response_mask,
        "old_log_probs": real_lp[idx],
        "advantages": real_adv[idx].unsqueeze(-1) * response_mask,
        "teacher_log_probs": real_teacher[idx],
        "ref_log_prob": real_ref[idx],
        "task_ids": torch.tensor([TASK_ID_NAMES.index(task[s]) for s in src], dtype=torch.long),
    }
    non_tensors = {
        "task_name": np.array([task[s] for s in src], dtype=object),
        "traj_uid": np.array([uid[s] for s in src], dtype=object),
    }
    meta = {"temperature": 1.0, "task_id_names": list(TASK_ID_NAMES), "multi_turn": False}
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta)


def _attach(batch, mode, metrics=None, n_real=N_REAL):
    """The driver's call, with the two arguments read off an actor config as the trainer would."""
    actor_cfg = OmegaConf.create({"pg_loss_norm": mode, "pg_ref_tokens": dict(REF_TOKENS)})
    attach_task_loss_weights(
        batch, n_real=n_real, mini_batch_size=MINI_GLOBAL,
        metrics={} if metrics is None else metrics, **pg_loss_norm_kwargs(actor_cfg),
    )
    return batch


def _balance(batch, seed=7):
    """_balance_batch's reorder: after the weights are attached, which move with their rows."""
    perm = torch.as_tensor(np.random.default_rng(seed).permutation(len(batch)))
    batch.reorder(perm)
    return batch


# --------------------------------------------------------------------------- #
# The worker: update_policy with a table for a model
# --------------------------------------------------------------------------- #
class _LogProbTable(torch.nn.Module):
    """The model: one free log-prob per (row id, response position)."""

    def __init__(self, init):
        super().__init__()
        self.theta = torch.nn.Parameter(init.clone())


class _StepRecorder:
    """The optimizer: records the gradient each optimizer step is handed and moves nothing.

    Nothing moving is what keeps every mini-batch at ratio 1 -- the update the formula is about --
    and makes the recorded gradients the ones the real optimizer would be handed.
    """

    def __init__(self, module):
        self.module = module
        self.grads = []

    def zero_grad(self, set_to_none=True):
        self.module.theta.grad = None

    def step(self):
        g = self.module.theta.grad
        self.grads.append(torch.zeros_like(self.module.theta) if g is None else g.detach().clone())


def _table_forward(module):
    def forward(micro_batch, temperature, calculate_entropy=False, topk_ids=None, topk_k=None,
                need_log_prob=True):
        rows = micro_batch["input_ids"][:, 0].long()
        log_prob = module.theta[rows]
        # Any differentiable function of the table stands in for the entropy; what is checked is
        # which weights aggregate it, not its value.
        entropy = 0.5 * module.theta[rows] ** 2 if calculate_entropy else None
        return entropy, log_prob, None

    return forward


def _actor_cfg(world, mode, *, teacher_coef=0.0, entropy=0.0, ref_kl=None, mini=MINI_GLOBAL, micro=MICRO,
               ref_tokens=None, **extra):
    """The OPD+GRPO actor settings the path needs; per-task weighting on, top-k off."""
    return OmegaConf.create({
        "strategy": "fsdp",
        # The worker divides the global (prompt-count x n) mini-batch by the DP size.
        "ppo_mini_batch_size": mini // world,
        "ppo_micro_batch_size_per_gpu": micro,
        "use_dynamic_bsz": False,
        "ppo_max_token_len_per_gpu": 16384,
        "ppo_epochs": 1,
        "shuffle": False,
        # Far above any norm here: the clip multiplies by exactly 1.
        "grad_clip": 1.0e12,
        "clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.2, "clip_ratio_c": 3.0,
        "loss_agg_mode": "token-mean",
        "entropy_coeff": entropy,
        "use_kl_loss": ref_kl is not None,
        "kl_loss_coef": 0.0 if ref_kl is None else ref_kl,
        "kl_loss_type": "low_var_kl",
        "pg_loss_coef": 1.0,
        "use_teacher_kl_loss": True,
        "teacher_kl_loss_type": "low_var_kl",
        "teacher_kl_loss_coef": teacher_coef,
        "normalize_loss_by_task": True,
        "pg_loss_norm": mode,
        "pg_ref_tokens": dict(REF_TOKENS if ref_tokens is None else ref_tokens),
        "ulysses_sequence_parallel_size": 1,
        "use_torch_compile": False,
        "use_remove_padding": False,
        "policy_loss": {"loss_mode": "vanilla"},
        **extra,
    })


def _update(batch, cfg, world, monkeypatch, n_steps=N_STEPS):
    """One training step on ``world`` DP ranks: every optimizer step's FSDP-averaged gradient.

    Each rank gets its contiguous chunk of the batch (the dp dispatch) and runs update_policy with
    torch.distributed reporting ``world`` ranks, so the actor multiplies the weights by
    world * grad_accum exactly as on the cluster; the ranks' per-step gradients are then averaged,
    which is FSDP's part. ``n_steps``: the optimizer steps every rank must have taken. Returns
    (per-step gradients over the table, per-rank metrics).
    """
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a, **k: world)
    # GPUMemoryLogger reads device memory around update_policy; there is none here.
    monkeypatch.setattr(_perf, "_get_current_mem_info", lambda *a, **k: ("0", "0", "0", "0"))
    ids = batch.batch["input_ids"][:, 0].long()
    table = torch.empty_like(batch.batch["old_log_probs"])
    table[ids] = batch.batch["old_log_probs"]
    per_rank, metrics = [], []
    for shard in batch.chunk(world):
        module = _LogProbTable(table)
        opt = _StepRecorder(module)
        actor = dp_actor.DataParallelPPOActor(cfg, module, opt)
        actor._forward_micro_batch = _table_forward(module)
        metrics.append(actor.update_policy(shard))
        per_rank.append(opt.grads)
    assert {len(g) for g in per_rank} == {n_steps}, [len(g) for g in per_rank]
    steps = [sum(g[m] for g in per_rank) / world for m in range(n_steps)]
    return steps, metrics


def _by_row(batch, table_values):
    """A (row id, position) table read back in the batch's own row order."""
    return table_values[batch.batch["input_ids"][:, 0].long()]


# --------------------------------------------------------------------------- #
# The formula, from the step's own description (not from any weight column)
# --------------------------------------------------------------------------- #
def _formula_row_weights(batch, mode, turns=None):
    """w_i = 1 / (D N_d L_d) under (a), 1 / (D T_d) under (b), 0 on padding -- in the batch's order."""
    turns = TURNS if turns is None else turns
    task = batch.non_tensor_batch["task_name"]
    uid = batch.non_tensor_batch["traj_uid"]
    real = ~batch.batch["is_padding_row"].numpy()
    tokens = batch.batch["response_mask"].sum(-1).to(torch.float64).numpy()
    D = len(turns)
    w = np.zeros(len(batch))
    for name in turns:
        rows = (task == name) & real
        if mode == "trajectory":
            n_traj = len(turns[name])
            assert len(set(uid[rows].tolist())) == n_traj
            w[rows] = 1.0 / (D * n_traj * REF_TOKENS[name])
        else:
            w[rows] = 1.0 / (D * tokens[rows].sum())
    return torch.as_tensor(w)


def _expected_pg_step_gradient(batch, mode, turns=None):
    """d(mean step loss)/d log pi(y_itu) = -w_i A_it on every generated token, in the batch's order."""
    w = _formula_row_weights(batch, mode, turns)
    return -(w.unsqueeze(-1) * batch.batch["advantages"] * batch.batch["response_mask"])


def _mark_padding(batch, n_real=N_REAL):
    """The test's own record of which rows are the copies (rows >= n_real before the reorder)."""
    batch.batch["is_padding_row"] = batch.batch["input_ids"][:, 0] >= n_real
    return batch


def _prepared_marked(mode, *, seed=0, zero_advantages=False, turns=None, n_pad=N_PAD, metrics=None):
    b = _step_batch(seed, zero_advantages=zero_advantages, turns=turns, n_pad=n_pad)
    n_real = _n_real(turns)
    _mark_padding(b, n_real)
    return _balance(_attach(b, mode, metrics, n_real=n_real))


# --------------------------------------------------------------------------- #
# The formula, through update_policy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("mode", ["trajectory", "token"])
def test_the_policy_gradient_is_the_formula(mode, world, monkeypatch):
    """PG alone (teacher-KL coefficient 0, no entropy, no reference KL): every token's gradient."""
    batch = _prepared_marked(mode)
    steps, metrics = _update(batch, _actor_cfg(world, mode), world, monkeypatch)

    mean_step = sum(steps) / len(steps)
    got = _by_row(batch, mean_step)
    want = _expected_pg_step_gradient(batch, mode)
    pad = batch.batch["is_padding_row"]
    # adjust_batch's copies contribute nothing, exactly.
    assert torch.count_nonzero(got[pad]) == 0
    # The weights are float32 (a relative 6e-8); everything else here is float64.
    assert torch.allclose(got, want, rtol=1e-6, atol=0.0), float((got - want).abs().max())
    # Not a vacuous agreement: every real row with a nonzero advantage moved.
    live = (~pad) & (batch.batch["advantages"].abs().sum(-1) > 0)
    assert bool((got[live].abs().sum(-1) > 0).all())
    # ...and it moved in exactly one optimizer step, the one its mini-batch became, carrying the
    # whole num_mini_batches factor there (the short final step included).
    touched = torch.stack([_by_row(batch, s).abs().sum(-1) > 0 for s in steps])
    assert bool((touched[:, live].sum(0) == 1).all())
    assert bool(touched[-1].any()), "the short final mini-batch trained nothing"


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("mode", ["trajectory", "token"])
def test_the_reported_policy_loss_is_the_formula(mode, world, monkeypatch):
    """actor/pg_loss_weighted: the mean over a rank's micro-batches of pg_term, which carries the
    world * grad_accum factor. Undone here, summed over ranks and divided by the number of
    optimizer steps it is the step's objective: -sum_i w_i A_i n_i at ratio 1."""
    batch = _prepared_marked(mode)
    cfg = _actor_cfg(world, mode)
    _, metrics = _update(batch, cfg, world, monkeypatch)
    mini_per_rank = MINI_GLOBAL // world
    grad_accum = mini_per_rank // MICRO
    rows_per_rank = len(batch) // world
    sizes = [min(mini_per_rank, rows_per_rank - s) for s in range(0, rows_per_rank, mini_per_rank)]
    n_micro = sum(math.ceil(s / MICRO) for s in sizes)
    total = sum(m["actor/pg_loss_weighted"] * n_micro for m in metrics) / (world * grad_accum)
    got = total / N_STEPS
    w = _formula_row_weights(batch, mode)
    n_tok = batch.batch["response_mask"].sum(-1).to(torch.float64)
    a_row = batch.batch["advantages"].sum(-1) / n_tok.clamp(min=1)
    want = float(-(w * a_row * n_tok).sum())
    assert got == pytest.approx(want, rel=1e-6)


# --------------------------------------------------------------------------- #
# The step's sum: num_mini_batches times the formula, and num_mini_batches moves
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("mode", ["trajectory", "token"])
def test_the_step_sum_is_the_formula_times_the_optimizer_steps(mode, world, monkeypatch):
    """The same trajectories per task with other turn counts: 20, 40 and 68 rows become 2, 4 and 6
    optimizer steps. The mean optimizer step is the formula every time; the step's summed gradient is
    that many times the formula, in both modes -- a factor set by the sampled turn counts, which
    task_loss_weights' docstring says the normalisation does not fix. task_loss/optimizer_steps
    reports it under (a) and is not written under (b)."""
    seen = set()
    for name, (turns, n_pad) in LAYOUTS.items():
        n_steps = math.ceil((_n_real(turns) + n_pad) / MINI_GLOBAL)
        metrics = {}
        batch = _prepared_marked(mode, seed=5, turns=turns, n_pad=n_pad, metrics=metrics)
        steps, _ = _update(batch, _actor_cfg(world, mode), world, monkeypatch, n_steps=n_steps)
        want = _expected_pg_step_gradient(batch, mode, turns)
        assert float(want.abs().max()) > 0
        step_sum = _by_row(batch, sum(steps))
        assert torch.allclose(step_sum, n_steps * want, rtol=1e-6, atol=0.0), (
            name, float((step_sum - n_steps * want).abs().max()))
        assert torch.allclose(_by_row(batch, sum(steps) / n_steps), want, rtol=1e-6, atol=0.0), name
        if mode == "trajectory":
            assert metrics["task_loss/optimizer_steps"] == n_steps
        else:
            assert "task_loss/optimizer_steps" not in metrics
        seen.add(n_steps)
    assert seen == {2, 4, 6}


# One task, one group of three: turn 0 picks a reply, "long" with probability sigma(phi) (phi = 0),
# a 3-turn episode whose turns 1-2 are forced (log-prob 0), or "short", a 1-turn one. Every episode
# wins, so the success objective does not depend on phi: its gradient is 0.
TOY_L = {"alfworld": 2.0}
# d log pi(reply) / d phi at phi = 0
TOY_DPHI = {"long": 0.5, "short": -0.5}


def _toy_batch(choices):
    """The group's turn rows, one generated token each, A = 1/2 on every turn (R = 1 against a frozen
    V = 1/2 at gamma = lambda = 1). Returns (batch, the row id of each trajectory's first turn)."""
    rows = [(i, t) for i, c in enumerate(choices) for t in range(3 if c == "long" else 1)]
    bs = len(rows)
    input_ids = torch.zeros(bs, 2, dtype=torch.long)
    input_ids[:, 0] = torch.arange(bs)
    mask = torch.ones(bs, 1, dtype=torch.long)
    old_lp = torch.tensor([[math.log(0.5) if t == 0 else 0.0] for _, t in rows], dtype=torch.float64)
    tensors = {
        "input_ids": input_ids,
        "attention_mask": torch.ones(bs, 2, dtype=torch.long),
        "position_ids": torch.arange(2).repeat(bs, 1),
        "responses": torch.zeros(bs, 1, dtype=torch.long),
        "response_mask": mask,
        "old_log_probs": old_lp,
        "advantages": 0.5 * mask.to(torch.float64),
        "teacher_log_probs": old_lp.clone(),
        "task_ids": torch.zeros(bs, dtype=torch.long),
    }
    non_tensors = {"task_name": np.array(["alfworld"] * bs, dtype=object),
                   "traj_uid": np.array([f"toy-{i}" for i, _ in rows], dtype=object)}
    meta = {"temperature": 1.0, "task_id_names": ["alfworld"], "multi_turn": False}
    first = {i: r for r, (i, t) in enumerate(rows) if t == 0}
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta), first


def _toy_expectations(mode, mini, monkeypatch):
    """E[d loss / d phi] of the mean optimizer step and of the step's sum, over the 8 equally likely
    groups, each through attach_task_loss_weights and update_policy (at ratio 1: nothing moves)."""
    import itertools

    e_mean = e_sum = 0.0
    for choices in itertools.product(("short", "long"), repeat=3):
        batch, first = _toy_batch(choices)
        attach_task_loss_weights(batch, n_real=len(batch), mini_batch_size=mini, metrics={},
                                 pg_loss_norm=mode, pg_ref_tokens=dict(TOY_L))
        n_steps = math.ceil(len(batch) / mini)
        cfg = _actor_cfg(1, mode, mini=mini, micro=1, ref_tokens=TOY_L)
        steps, _ = _update(batch, cfg, 1, monkeypatch, n_steps=n_steps)
        # the chain rule to phi: only a turn-0 token depends on it
        d_sum = sum(float(s[first[i], 0]) * TOY_DPHI[c] for s in steps for i, c in enumerate(choices))
        e_sum += d_sum / 8
        e_mean += d_sum / n_steps / 8
    return e_mean, e_sum


@pytest.mark.parametrize("mini", [1, 2])
def test_the_guarantee_is_the_mean_optimizer_step(mini, monkeypatch):
    """Exact expectations in the toy above, through the production code. (a): the mean optimizer
    step's gradient is g = -(2n - 3) / 24 in phi (n the long replies), expectation 0 = the objective's
    gradient -- what the normalisation guarantees. The step's sum is num_mini_batches * g, and
    num_mini_batches = ceil((3 + 2n) / mini) grows with the long replies: expectation -1/8 at one row
    per mini-batch, -1/16 at two, i.e. the update raises the long reply's log-odds although the
    objective is flat -- the optimizer-step count the docstring leaves outside the guarantee.
    (b), for contrast: its mean optimizer step is the ratio estimator, -(2n - 3) / (4 (3 + 2n)),
    expectation 11/420 whatever the mini-batch -- the bias (a) removes."""
    e_mean, e_sum = _toy_expectations("trajectory", mini, monkeypatch)
    # the weights are float32 (5/6 and 7/6 round at 1e-8): zero to that, against terms of order 0.1
    assert e_mean == pytest.approx(0.0, abs=1e-7)
    assert e_sum == pytest.approx({1: -1 / 8, 2: -1 / 16}[mini], rel=1e-6)
    b_mean, _ = _toy_expectations("token", mini, monkeypatch)
    assert b_mean == pytest.approx(11 / 420, rel=1e-6)


def test_switching_moves_the_policy_gradient_and_nothing_else(monkeypatch):
    """Teacher KL, reference KL and entropy on, at the token weights in both modes."""
    world = 2
    full = dict(teacher_coef=0.3, entropy=0.01, ref_kl=0.05)

    # Advantages at zero: the step's gradient is the other terms' alone, and bitwise one thing.
    tok0 = _prepared_marked("token", seed=3, zero_advantages=True)
    trj0 = _prepared_marked("trajectory", seed=3, zero_advantages=True)
    assert torch.equal(tok0.batch[TASK_LOSS_WEIGHT_KEY], trj0.batch[TASK_LOSS_WEIGHT_KEY])
    s_tok0, _ = _update(tok0, _actor_cfg(world, "token", **full), world, monkeypatch)
    s_trj0, _ = _update(trj0, _actor_cfg(world, "trajectory", **full), world, monkeypatch)
    for a, b in zip(s_tok0, s_trj0):
        assert torch.equal(a, b)
    assert any(bool(s.abs().sum() > 0) for s in s_tok0), "the other terms produced no gradient"

    # Advantages live: the two modes differ by exactly the two formulas' difference.
    tok = _prepared_marked("token", seed=3)
    trj = _prepared_marked("trajectory", seed=3)
    s_tok, _ = _update(tok, _actor_cfg(world, "token", **full), world, monkeypatch)
    s_trj, _ = _update(trj, _actor_cfg(world, "trajectory", **full), world, monkeypatch)
    got = _by_row(trj, sum(s_trj) / N_STEPS) - _by_row(trj, sum(s_tok) / N_STEPS)
    want = _expected_pg_step_gradient(trj, "trajectory") - _expected_pg_step_gradient(trj, "token")
    scale = float(want.abs().max())
    assert scale > 0
    assert torch.allclose(got, want, rtol=1e-6, atol=1e-6 * scale), float((got - want).abs().max())


def test_token_mode_never_reads_the_policy_gradient_column(monkeypatch):
    """A batch carrying the trajectory weights trains, under token, exactly as one without them."""
    world = 2
    cfg = _actor_cfg(world, "token", teacher_coef=0.3, entropy=0.01, ref_kl=0.05)
    without = _prepared_marked("token", seed=11)
    with_col = _prepared_marked("trajectory", seed=11)
    assert TASK_PG_LOSS_WEIGHT_KEY not in without.batch.keys()
    assert TASK_PG_LOSS_WEIGHT_KEY in with_col.batch.keys()
    s_a, m_a = _update(without, cfg, world, monkeypatch)
    s_b, m_b = _update(with_col, cfg, world, monkeypatch)
    for a, b in zip(s_a, s_b):
        assert torch.equal(a, b)
    assert m_a == m_b


def test_trajectory_mode_refuses_a_batch_without_its_column(monkeypatch):
    """The token normalisation under the other one's name is the failure this refuses."""
    batch = _prepared_marked("token")
    with pytest.raises(AssertionError, match=TASK_PG_LOSS_WEIGHT_KEY):
        _update(batch, _actor_cfg(1, "trajectory"), 1, monkeypatch)


# --------------------------------------------------------------------------- #
# The driver side
# --------------------------------------------------------------------------- #
def test_the_trajectory_weights_count_trajectories_not_rows():
    metrics = {}
    batch = _mark_padding(_step_batch(1))
    _attach(batch, "trajectory", metrics)
    pg = batch.batch[TASK_PG_LOSS_WEIGHT_KEY]
    assert pg.dtype == torch.float32
    pad = batch.batch["is_padding_row"]
    assert torch.count_nonzero(pg[pad]) == 0
    task = batch.non_tensor_batch["task_name"]
    tokens = batch.batch["response_mask"].sum(-1).to(torch.float64)
    for name, turns in TURNS.items():
        rows = torch.from_numpy(task == name) & ~pad
        # N_d is the trajectories, whatever their turn counts, and a padding copy's traj_uid is
        # its original's: not a second trajectory.
        assert metrics[f"task_loss/pg_trajectories/{name}"] == len(turns)
        want = N_STEPS / (len(TURNS) * len(turns) * REF_TOKENS[name])
        assert torch.allclose(pg[rows].double(), torch.full((int(rows.sum()),), want, dtype=torch.float64), rtol=1e-7)
        # The (a) weight over the (b) weight is the step's mean tokens per trajectory over L_d.
        t_d = float(tokens[rows].sum())
        assert metrics[f"task_loss/pg_weight_ratio/{name}"] == pytest.approx(t_d / (len(turns) * REF_TOKENS[name]))
        tok_w = float(batch.batch[TASK_LOSS_WEIGHT_KEY][rows][0])
        assert float(pg[rows][0]) / tok_w == pytest.approx(metrics[f"task_loss/pg_weight_ratio/{name}"], rel=1e-6)


def test_token_mode_writes_what_it_always_wrote():
    tok_metrics, trj_metrics = {}, {}
    tok = _attach(_step_batch(2), "token", tok_metrics)
    trj = _attach(_step_batch(2), "trajectory", trj_metrics)
    assert TASK_PG_LOSS_WEIGHT_KEY not in tok.batch.keys()
    # The token weights do not depend on the mode.
    assert torch.equal(tok.batch[TASK_LOSS_WEIGHT_KEY], trj.batch[TASK_LOSS_WEIGHT_KEY])
    assert set(tok_metrics) == (
        {f"task_loss/token_share/{t}" for t in TURNS} | {f"task_loss/rows/{t}" for t in TURNS}
        | {"task_loss/padding_rows"}
    )
    assert set(trj_metrics) - set(tok_metrics) == (
        {f"task_loss/pg_trajectories/{t}" for t in TURNS} | {f"task_loss/pg_weight_ratio/{t}" for t in TURNS}
        | {"task_loss/optimizer_steps"}
    )
    assert trj_metrics["task_loss/optimizer_steps"] == N_STEPS
    assert {k: tok_metrics[k] for k in tok_metrics} == {k: trj_metrics[k] for k in tok_metrics}


@pytest.mark.parametrize("ref,match", [
    (None, "pg_ref_tokens is unset"),
    ({"alfworld": 17.0, "webshop": 9.0}, "no entry for \\['search'\\]"),
    ({"alfworld": 17.0, "webshop": 0.0, "search": 3.5}, "positive and finite"),
    ({"alfworld": float("nan"), "webshop": 9.0, "search": 3.5}, "positive and finite"),
])
def test_the_reference_lengths_are_required(ref, match):
    batch = _step_batch(0)
    with pytest.raises(AssertionError, match=match):
        attach_task_loss_weights(batch, n_real=N_REAL, mini_batch_size=MINI_GLOBAL, metrics={},
                                 pg_loss_norm="trajectory", pg_ref_tokens=ref)
    # Refused before anything was written.
    assert TASK_LOSS_WEIGHT_KEY not in batch.batch.keys()


def test_the_trainer_hands_the_actor_settings_to_the_weights():
    """attach_task_loss_weights can only write the trajectory column if the driver tells it the mode."""
    import ast

    src = open(os.path.join(REPO, "verl", "trainer", "ppo", "opd_ray_trainer.py"), encoding="utf-8").read()

    def _name(node):
        return getattr(node, "id", getattr(node, "attr", None))

    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and _name(n.func) == "attach_task_loss_weights"]
    assert calls
    for call in calls:
        named = {k.arg for k in call.keywords if k.arg is not None}
        spread = [k.value for k in call.keywords if k.arg is None]
        assert {"pg_loss_norm", "pg_ref_tokens"} <= named or any(
            isinstance(v, ast.Call) and _name(v.func) == "pg_loss_norm_kwargs" for v in spread
        ), ast.unparse(call)


def test_the_trajectory_count_needs_traj_uid():
    batch = _step_batch(0)
    batch.non_tensor_batch.pop("traj_uid")
    with pytest.raises(AssertionError, match="traj_uid"):
        _attach(batch, "trajectory")


def test_an_unknown_mode_is_refused():
    with pytest.raises(AssertionError, match="pg_loss_norm"):
        attach_task_loss_weights(_step_batch(0), n_real=N_REAL, mini_batch_size=MINI_GLOBAL,
                                 metrics={}, pg_loss_norm="sequence")


def test_the_launch_check():
    cfg = OmegaConf.create({"normalize_loss_by_task": True, "pg_loss_norm": "trajectory",
                            "pg_ref_tokens": dict(REF_TOKENS)})
    assert check_pg_loss_norm_config(cfg, tasks=list(TURNS)) == "trajectory"
    assert check_pg_loss_norm_config(OmegaConf.create({}), tasks=list(TURNS)) == "token"
    with pytest.raises(AssertionError, match="normalize_loss_by_task"):
        check_pg_loss_norm_config(OmegaConf.merge(cfg, {"normalize_loss_by_task": False}))
    with pytest.raises(AssertionError, match="no entry"):
        check_pg_loss_norm_config(cfg, tasks=list(TURNS) + ["sokoban"])
    with pytest.raises(AssertionError, match="unset"):
        check_pg_loss_norm_config(OmegaConf.merge(cfg, {"pg_ref_tokens": None}))
    with pytest.raises(AssertionError, match="expected one of"):
        check_pg_loss_norm_config(OmegaConf.merge(cfg, {"pg_loss_norm": "row"}))


# --------------------------------------------------------------------------- #
# The actor's refusals
# --------------------------------------------------------------------------- #
def _check(cfg_extra=None, **kw):
    cfg = OmegaConf.create({"pg_loss_norm": "trajectory", **(cfg_extra or {})})
    args = dict(task_weighted=True, has_pg_weights=True, pg_loss_coef=1.0, measure_terms=False)
    args.update(kw)
    return dp_actor.check_pg_loss_norm_supported(cfg, **args)


def test_the_actor_check_accepts_and_defaults():
    assert _check() is True
    assert dp_actor.check_pg_loss_norm_supported(
        OmegaConf.create({}), task_weighted=False, has_pg_weights=False, pg_loss_coef=0.0) is False
    # Token ignores everything the trajectory mode would refuse.
    assert dp_actor.check_pg_loss_norm_supported(
        OmegaConf.create({"pg_loss_norm": "token", "oci_slots": {"enable": True}}),
        task_weighted=False, has_pg_weights=True, pg_loss_coef=0.0, measure_terms=True) is False


@pytest.mark.parametrize("extra,kw,match", [
    (None, dict(task_weighted=False), "normalize_loss_by_task"),
    (None, dict(pg_loss_coef=0.0), "pg_loss_coef is 0"),
    (None, dict(has_pg_weights=False), TASK_PG_LOSS_WEIGHT_KEY),
    (None, dict(measure_terms=True), "mass"),
    ({"oci_sat": {"shaping": {"enable": True}}}, {}, "oci_sat.shaping"),
    ({"oci_slots": {"enable": True}}, {}, "oci_slots"),
    ({"teacher_kl_pushback": {"enable": True}}, {}, "teacher_kl_pushback"),
    ({"teacher_kl_cross_gate": {"enable": True}}, {}, "teacher_kl_cross_gate"),
    ({"logit_precision": {"enable": True}}, {}, "logit_precision"),
    ({"teacher_kl_task_diag": True}, {}, "teacher_kl_task_diag"),
    ({"pg_loss_norm": "sequence"}, {}, "expected one of"),
])
def test_the_actor_check_refuses(extra, kw, match):
    with pytest.raises(AssertionError, match=match):
        _check(extra, **kw)


def test_the_disabled_blocks_are_not_refused():
    assert _check({"oci_sat": {"shaping": {"enable": False}}, "oci_slots": {"enable": False},
                   "teacher_kl_pushback": {"enable": False}, "logit_precision": None,
                   "teacher_kl_task_diag": False}) is True


# --------------------------------------------------------------------------- #
# The geometry diagnostics under (a)
# --------------------------------------------------------------------------- #
def test_the_geometry_diagnostics_see_the_weight_the_loss_applied():
    """opd_attribution_terms puts ONE row weight on both halves. Handed the PG coefficient times
    pg_geometry_ratio, its columns are those of the objective under (a): the teacher side at the
    token weight, the policy-gradient side at its own."""
    from verl.trainer.ppo.core_algos import topk_kl_per_token
    from verl.trainer.ppo.cross_teacher_kl_weight import opd_attribution_terms

    g = torch.Generator().manual_seed(0)
    bs, T, k = 5, 4, 3
    s = torch.log_softmax(torch.randn(bs, T, k + 2, generator=g), -1)[..., :k]
    t = torch.log_softmax(torch.randn(bs, T, k + 2, generator=g), -1)[..., :k]
    kl = topk_kl_per_token(student_topk_logprob=s, teacher_topk_logprob=t)
    onehot = torch.nn.functional.one_hot(torch.randint(0, k, (bs, T), generator=g), k).to(torch.float32)
    coef = torch.randn(bs, T, generator=g)
    tok_w = torch.tensor([0.3, 0.1, 0.0, 0.7, 0.2])
    pg_w = torch.tensor([0.5, 0.05, 0.0, 0.2, 0.9])

    ratio = dp_actor.pg_geometry_ratio(pg_w, tok_w)
    assert ratio.shape == (bs, 1) and float(ratio[2]) == 0.0 and bool(torch.isfinite(ratio).all())
    base = opd_attribution_terms(student_logprob=s, teacher_logprob=t, teacher_kl=kl, pg_grad_coef=coef,
                                 sampled_onehot=onehot, coef=0.01, pg_coef=1.0, row_weight=None)
    got = opd_attribution_terms(student_logprob=s, teacher_logprob=t, teacher_kl=kl,
                                pg_grad_coef=coef * ratio.to(coef.dtype), sampled_onehot=onehot,
                                coef=0.01, pg_coef=1.0, row_weight=tok_w)
    tw, pw = tok_w.reshape(-1, 1), pg_w.reshape(-1, 1)
    want = {
        "g_opd_sq": base["g_opd_sq"] * tw * tw,
        "g_grpo_sq": base["g_grpo_sq"] * pw * pw,
        "g_dot": base["g_dot"] * tw * pw,
        "d_kl": base["d_kl"] * tw,
        "push_abs": base["push_abs"] * tw,
    }
    for name, value in want.items():
        assert torch.allclose(got[name], value, rtol=1e-5, atol=1e-12), name


# --------------------------------------------------------------------------- #
# The 2x2's launchers and locks (parsed and composed, never executed)
# --------------------------------------------------------------------------- #
CELLS = {
    ("grpo", "token"): "examples/opd_grpo_trainer/run_multitask_grpo_v2recipe_qwen3.sh",
    ("grpo", "trajectory"): "examples/opd_grpo_trainer/run_multitask_grpo_v2recipe_trajnorm_qwen3.sh",
    ("value", "token"): "examples/opd_grpo_trainer/run_multitask_progress_value_gae_qwen3.sh",
    ("value", "trajectory"): "examples/opd_grpo_trainer/run_multitask_progress_value_gae_trajnorm_qwen3.sh",
}
LOCKS = {
    ("grpo", "token", None): "expected_multitask_grpo_v2recipe_config.yaml",
    ("grpo", "trajectory", None): "expected_multitask_grpo_v2recipe_trajnorm_config.yaml",
    ("value", "token", "1.0"): "expected_multitask_progress_value_gae_lam1.0_config.yaml",
    ("value", "token", "0.9"): "expected_multitask_progress_value_gae_lam0.9_config.yaml",
    ("value", "trajectory", "1.0"): "expected_multitask_progress_value_gae_trajnorm_lam1.0_config.yaml",
    ("value", "trajectory", "0.9"): "expected_multitask_progress_value_gae_trajnorm_lam0.9_config.yaml",
}


def _injected(monkeypatch, cell, lam=None, extra=()):
    from hydra import compose, initialize_config_dir

    from tests.trainer.test_run_script_overrides_compose import _overrides
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config

    monkeypatch.setenv("RUN_TAG_SUFFIX", "")
    if lam is None:
        monkeypatch.delenv("LAM", raising=False)
    else:
        monkeypatch.setenv("LAM", lam)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="ppo_trainer", overrides=list(_overrides(CELLS[cell])) + list(extra))
    inject_opd_grpo_config(cfg)
    return cfg


@pytest.mark.parametrize("key", sorted(LOCKS, key=str), ids=lambda k: "-".join(str(x) for x in k))
def test_each_cell_satisfies_its_own_lock(monkeypatch, key):
    from verl.utils.expected_config import check_expected_config

    est, norm, lam = key
    cfg = _injected(monkeypatch, (est, norm), lam)
    assert cfg.trainer.expected_config.endswith(LOCKS[key])
    assert check_expected_config(cfg, os.path.join(REPO, cfg.trainer.expected_config)) == []
    actor, alg = cfg.actor_rollout_ref.actor, cfg.algorithm
    assert alg.adv_estimator == ("grpo" if est == "grpo" else "progress_value_gae")
    assert alg.lam == float(lam or "1.0") and alg.gamma == 1.0
    # The launch check the trainer's driver call would otherwise make only after the first rollout.
    assert check_pg_loss_norm_config(actor, tasks=list(cfg.data.task_balance.tasks)) == norm
    # ...and the actor's own check, on the actor config the workers get: nothing the trajectory
    # normalisation refuses (OCI shaping, the gates, logit precision, the task diag) is on here.
    assert dp_actor.check_pg_loss_norm_supported(
        actor, task_weighted=True, has_pg_weights=True, pg_loss_coef=float(actor.pg_loss_coef),
    ) is (norm == "trajectory")
    if norm == "trajectory":
        assert dict(actor.pg_ref_tokens) == {"alfworld": 4650, "webshop": 2320, "search": 310}
    # The recipe underneath: the value arm's records and counters, no (a)/sat, the teacher kept.
    assert alg.progress_value.enable and not alg.progress_value.allow_missing_table
    assert not alg.progress_rank.enable
    assert (alg.progress_rank.alfworld_k, alg.progress_rank.search_k, alg.progress_rank.webshop_k) == (
        "milestone_arrive", "evidence_answered", "session")
    assert actor.normalize_loss_by_task and alg.opd.retire.enable is False


def _knobs(cfg):
    """The effective config as {dotted key: value}, pg_ref_tokens' entries folded into one knob."""
    out = {}

    def walk(node, prefix):
        if isinstance(node, dict) and not prefix.endswith("pg_ref_tokens"):
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        else:
            out[prefix] = node

    walk(OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False), "")
    return out


def test_the_cells_differ_only_in_their_row_and_column(monkeypatch):
    """Each pair of neighbouring cells differs in its one knob (and its lock), nothing else."""
    knobs = {cell: _knobs(_injected(monkeypatch, cell)) for cell in CELLS}

    def diff(a, b):
        ka, kb = knobs[a], knobs[b]
        return {k for k in set(ka) | set(kb) if ka.get(k, "<absent>") != kb.get(k, "<absent>")}

    norm = {"actor_rollout_ref.actor.pg_loss_norm", "actor_rollout_ref.actor.pg_ref_tokens",
            "trainer.expected_config"}
    est = {"algorithm.adv_estimator", "trainer.expected_config"}
    assert diff(("value", "trajectory"), ("value", "token")) == norm
    assert diff(("grpo", "trajectory"), ("grpo", "token")) == norm
    assert diff(("grpo", "token"), ("value", "token")) == est
    assert diff(("grpo", "trajectory"), ("value", "trajectory")) == est


def test_each_lock_is_its_parent_plus_its_knob():
    """The derived locks are the value arm's with the cell's knob changed and nothing else, so a
    key later pinned in the parent and not carried over fails here rather than going unchecked."""
    from verl.utils.expected_config import load_expectations

    d = os.path.join(REPO, "examples", "opd_grpo_trainer")
    lock = {key: load_expectations(os.path.join(d, name)) for key, name in LOCKS.items()}
    traj = {
        "actor_rollout_ref.actor.pg_loss_norm": "trajectory",
        "actor_rollout_ref.actor.pg_ref_tokens.alfworld": 4650,
        "actor_rollout_ref.actor.pg_ref_tokens.webshop": 2320,
        "actor_rollout_ref.actor.pg_ref_tokens.search": 310,
    }
    for lam in ("1.0", "0.9"):
        assert lock[("value", "trajectory", lam)] == {**lock[("value", "token", lam)], **traj}
    assert lock[("grpo", "token", None)] == {**lock[("value", "token", "1.0")], "algorithm.adv_estimator": "grpo"}
    assert lock[("grpo", "trajectory", None)] == {**lock[("grpo", "token", None)], **traj}
    # Every cell pins its column, the value arm's (b) locks included: the one key that decides which
    # column a run is in can never be left to a default or a stray override.
    for (_, norm, _), pins in lock.items():
        assert pins["actor_rollout_ref.actor.pg_loss_norm"] == norm


def test_the_two_trajnorm_locks_differ_only_in_lam():
    from verl.utils.expected_config import load_expectations

    d = os.path.join(REPO, "examples", "opd_grpo_trainer")
    one = load_expectations(os.path.join(d, LOCKS[("value", "trajectory", "1.0")]))
    nine = load_expectations(os.path.join(d, LOCKS[("value", "trajectory", "0.9")]))
    assert {k for k in set(one) | set(nine) if one.get(k) != nine.get(k)} == {"algorithm.lam"}


@pytest.mark.parametrize("cell,override", [
    (("value", "trajectory"), "actor_rollout_ref.actor.pg_loss_norm=token"),
    # One task's L_d moved (v1's ALFWorld length): the lock pins each entry.
    (("value", "trajectory"), "actor_rollout_ref.actor.pg_ref_tokens={alfworld:4690,webshop:2320,search:310}"),
    (("value", "trajectory"), "algorithm.adv_estimator=grpo"),
    (("grpo", "token"), "actor_rollout_ref.actor.pg_loss_norm=trajectory"),
    (("grpo", "token"), "algorithm.adv_estimator=progress_value_gae"),
    (("grpo", "trajectory"), "actor_rollout_ref.actor.pg_loss_norm=token"),
    (("grpo", "trajectory"), "algorithm.progress_value.allow_missing_table=True"),
])
def test_the_locks_catch(monkeypatch, cell, override):
    from verl.utils.expected_config import check_expected_config

    extra = [override]
    if override.endswith("pg_loss_norm=trajectory"):
        extra.append("actor_rollout_ref.actor.pg_ref_tokens={alfworld:4650,webshop:2320,search:310}")
    cfg = _injected(monkeypatch, cell, extra=extra)
    miss = check_expected_config(cfg, os.path.join(REPO, cfg.trainer.expected_config))
    assert len(miss) == 1, miss


@pytest.mark.parametrize("lam", ["1.0", "0.9"])
def test_the_value_arm_token_locks_catch_the_other_column(monkeypatch, lam):
    """(a)'s two overrides reaching the value arm's (b) launcher -- copied from an (a) command line, or
    arriving through a supervisor's arguments -- are one mismatch at launch, not a run named
    progress_value_gae_lam* that trains under (a) for 300 steps."""
    from verl.utils.expected_config import check_expected_config

    cfg = _injected(monkeypatch, ("value", "token"), lam, extra=[
        "actor_rollout_ref.actor.pg_loss_norm=trajectory",
        "actor_rollout_ref.actor.pg_ref_tokens={alfworld:4650,webshop:2320,search:310}"])
    assert cfg.trainer.expected_config.endswith(LOCKS[("value", "token", lam)])
    miss = check_expected_config(cfg, os.path.join(REPO, cfg.trainer.expected_config))
    assert [m[0] for m in miss] == ["actor_rollout_ref.actor.pg_loss_norm"], miss


@pytest.mark.parametrize("extra,match", [
    (["actor_rollout_ref.actor.pg_loss_norm=trajectory"], "pg_ref_tokens is unset"),
    (["actor_rollout_ref.actor.pg_loss_norm=trajectory",
      "actor_rollout_ref.actor.pg_ref_tokens={alfworld:4650,webshop:2320}"], "no entry for \\['search'\\]"),
    (["actor_rollout_ref.actor.pg_loss_norm=trajectroy"], "expected one of"),
])
def test_the_launch_refuses_what_the_driver_would_refuse_after_a_rollout(monkeypatch, extra, match):
    """inject_opd_grpo_config runs check_pg_loss_norm_config: a trajectory run without every task's L_d
    (or a misspelt mode) stops while the config is composed, before the lock is even read -- not at
    the driver's first attach_task_loss_weights after a rollout, and again at every supervisor restart."""
    with pytest.raises(AssertionError, match=match):
        _injected(monkeypatch, ("value", "token"), extra=extra)


def test_each_cell_has_its_own_run_tag():
    """RUN_TAG names the checkpoint directory and the wandb run; the defaults must not collide."""
    want = {
        CELLS[("grpo", "token")]: 'export RUN_TAG="${RUN_TAG:-grpo_v2recipe}"',
        CELLS[("grpo", "trajectory")]: 'export RUN_TAG="${RUN_TAG:-grpo_v2recipe_trajnorm}"',
        CELLS[("value", "trajectory")]: 'export RUN_TAG="${RUN_TAG:-progress_value_gae_lam${LAM//./p}_trajnorm}"',
    }
    for path, line in want.items():
        assert line in open(os.path.join(REPO, path)).read(), path
