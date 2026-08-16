# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Teacher log-probs at ids the student picks, without re-running the teacher.

The distillation KL needs ``log p_t`` at a support set. Taking that support from
the *student's* top-k rather than the teacher's would normally force the teacher
to run after the student, because a top-k of teacher log-probs can only be read
at ids chosen in advance. It does not have to: with

    log p_t(v) = h · W_t[v] - lse_t

only the last gather depends on the ids, and it is ~1/42,000 of the teacher
forward (2·H·k against a full body plus vocabulary projection). Caching ``h`` and
``lse`` therefore makes the teacher's output id-agnostic, and the teacher can
keep running where it runs today -- inside the rollout's CPU glue.

What that buys costs a new hazard. Today the teacher's output goes back to the
driver and travels with its row, so nothing has to know which rank produced it.
A hidden-state cache lives in one rank's memory, and the rank that later computes
the student's top-k is a different one: between the two calls the rows are
regrouped by task, concatenated, padded by ``adjust_batch`` and then reordered by
``_balance_batch`` specifically to equalise tokens per rank. So ownership is real
and it is never the identity map.

This module keeps that hazard loud:

* ``exchange_teacher_logprobs`` all-gathers each rank's (cache id, student ids),
  lets whichever rank owns an entry answer, and sums the answers back. Ownership
  is unique, so the sum is exact -- ``x + 0.0 == x``. A companion count is summed
  the same way and must be exactly 1 everywhere: 0 means a row nobody had, 2
  means two ranks claimed it. Both raise.
* ``TeacherHiddenCache.check_witness`` recomputes log-probs at the teacher's OWN
  top-k -- which the teacher returns anyway, and which the trainer already ships
  -- and compares against the values it returned. Consistent h/lse/W reproduce
  them to a few ULP; a cache entry paired with the wrong row is off by orders of
  magnitude. That is the difference between a check that fires and one that does
  not.

The count guard answers "is this the row I asked for"; the witness answers "is
what you cached still what the teacher computed". Neither subsumes the other: a
self-consistent entry filed under the wrong id passes the witness, and a
corrupted entry filed correctly passes the count.
"""

from typing import Dict, Optional, Tuple

import torch


_PROCESS_CACHE: Optional["TeacherHiddenCache"] = None


def get_teacher_cache() -> "TeacherHiddenCache":
    """The one cache in this worker process.

    A module global rather than an attribute because the teachers and the actor
    are siblings inside one ``WorkerDict``, not nested: the actor's
    ``update_policy`` has no handle on the ref-role workers that filled the cache,
    but it does share their process. Ranks each get their own -- that is the
    ownership this module exists to police, not to hide.
    """
    global _PROCESS_CACHE
    if _PROCESS_CACHE is None:
        _PROCESS_CACHE = TeacherHiddenCache()
    return _PROCESS_CACHE


def teacher_logprobs_from_hidden(
    h: torch.Tensor,
    lse: torch.Tensor,
    lm_head_weight: torch.Tensor,
    ids: torch.Tensor,
) -> torch.Tensor:
    """``log p_t`` at ``ids``, from the teacher's hidden states and its lse.

    Equal to indexing a full ``log_softmax(h @ W.T)`` at the same ids, in exact
    arithmetic: the projection is per-position linear and the normaliser is the
    one the teacher already computed over the whole vocabulary. Only the GEMM
    shape differs (``h @ W[ids].T`` instead of ``h @ W.T``), which moves the last
    bits the way any repacking does.

    Args:
        h: (n, hidden) teacher hidden states at the scored positions.
        lse: (n,) or (n, 1) logsumexp over the FULL vocabulary, from the teacher
            forward that produced ``h``.
        lm_head_weight: (vocab, hidden) the teacher's output projection.
        ids: (n, k) token ids to evaluate.

    Returns:
        (n, k) float32 log-probs.
    """
    if lse.dim() == 1:
        lse = lse.unsqueeze(-1)
    # (n, k, hidden) gathered rows of the projection. Built per micro-batch and
    # dropped: at step scale this would be ~90 GB, at micro-batch scale ~90 MB.
    w_ids = lm_head_weight[ids]
    logits = torch.einsum("nh,nkh->nk", h.float(), w_ids.float())
    return logits - lse.float()


class TeacherHiddenCache:
    """Process-local store of one step's teacher hidden states.

    Keyed by an int64 the driver assigns when it queues a row for scoring, so the
    key survives every reordering between the teacher call and the actor update.
    One entry per row; ``owner`` is implicit -- whichever rank has the key.
    """

    def __init__(self):
        self._h: Dict[int, torch.Tensor] = {}
        self._lse: Dict[int, torch.Tensor] = {}
        self._task: Dict[int, str] = {}
        self._witness_ids: Dict[int, torch.Tensor] = {}
        self._witness_lp: Dict[int, torch.Tensor] = {}
        self._weights: Dict[str, torch.Tensor] = {}

    # -- registration ----------------------------------------------------- #

    def register_lm_head(self, task: str, weight: torch.Tensor):
        """Keep an unsharded copy of a teacher's output projection.

        Needed because the ref path reshards after every call, so by the time the
        actor update runs the sharded parameter cannot be indexed at arbitrary
        ids. One copy per teacher, held for the run.
        """
        self._weights[task] = weight

    def lm_head(self, task: str) -> torch.Tensor:
        if task not in self._weights:
            raise KeyError(f"no lm_head registered for teacher '{task}'; registered: {sorted(self._weights)}")
        return self._weights[task]

    # -- filling ---------------------------------------------------------- #

    def put(self, cache_ids, task: str, h, lse, witness_ids=None, witness_lp=None):
        """Store one call's rows. ``cache_ids`` is (n,), ``h`` is (n, hidden)."""
        ids_list = [int(c) for c in cache_ids]
        for i, key in enumerate(ids_list):
            if key < 0:
                continue
            self._h[key] = h[i]
            self._lse[key] = lse[i]
            self._task[key] = task
            if witness_ids is not None:
                self._witness_ids[key] = witness_ids[i]
                self._witness_lp[key] = witness_lp[i]

    def clear(self):
        self._h.clear()
        self._lse.clear()
        self._task.clear()
        self._witness_ids.clear()
        self._witness_lp.clear()

    def __len__(self):
        return len(self._h)

    def __contains__(self, key):
        return int(key) in self._h

    # -- reading ---------------------------------------------------------- #

    def logprobs_at(self, cache_ids: torch.Tensor, ids: torch.Tensor):
        """Answer for the entries this cache owns; leave the rest at zero.

        Args:
            cache_ids: (n,) int64 keys being asked about; -1 means "not scored".
            ids: (n, k) token ids to evaluate.

        Returns:
            values: (n, k) float32, zero where this cache does not own the key.
            found: (n,) int32, 1 where it does.
        """
        n, k = ids.shape
        values = torch.zeros((n, k), dtype=torch.float32, device=ids.device)
        found = torch.zeros((n,), dtype=torch.int32, device=ids.device)
        if not self._h:
            return values, found

        # Group by task: the projection differs per teacher, the arithmetic does not.
        by_task: Dict[str, list] = {}
        for pos in range(n):
            key = int(cache_ids[pos])
            if key < 0 or key not in self._h:
                continue
            by_task.setdefault(self._task[key], []).append((pos, key))

        for task, entries in by_task.items():
            rows = torch.tensor([p for p, _ in entries], dtype=torch.long, device=ids.device)
            h = torch.stack([self._h[key] for _, key in entries]).to(ids.device)
            lse = torch.stack([self._lse[key] for _, key in entries]).to(ids.device)
            out = teacher_logprobs_from_hidden(h, lse, self.lm_head(task), ids[rows])
            values[rows] = out
            found[rows] = 1
        return values, found

    def check_witness(self, atol: float = 1e-3) -> float:
        """Recompute at the teacher's own top-k and return the largest deviation.

        The teacher returns its top-k anyway (the trainer ships it today), so this
        costs one narrow GEMM over what is already cached. Consistent h/lse/W
        reproduce those values to a few ULP; an entry paired with the wrong row is
        off by whole nats. Raises past ``atol`` rather than reporting, because a
        cache that fails this has been feeding wrong targets to the loss.
        """
        worst = 0.0
        for key, w_ids in self._witness_ids.items():
            h = self._h[key].unsqueeze(0)
            lse = self._lse[key].reshape(1, -1)[:, :1]
            got = teacher_logprobs_from_hidden(h, lse, self.lm_head(self._task[key]), w_ids.unsqueeze(0).to(h.device))
            err = (got.squeeze(0) - self._witness_lp[key].to(got.device).float()).abs().max().item()
            worst = max(worst, err)
        if worst > atol:
            raise RuntimeError(
                f"teacher hidden-state cache failed its witness check: max deviation {worst:.3e} > {atol:.0e}. "
                f"The cached h/lse no longer reproduce the log-probs the teacher returned, so the ids being "
                f"scored do not belong to the rows they are filed under."
            )
        return worst


def exchange_teacher_logprobs(
    cache: TeacherHiddenCache,
    cache_ids: torch.Tensor,
    ids: torch.Tensor,
    group=None,
    world_size: Optional[int] = None,
) -> torch.Tensor:
    """Get ``log p_t`` at each rank's ids from whichever rank cached that row.

    Every rank broadcasts what it wants, every rank answers what it owns, and the
    answers are summed back. Ownership is unique so the sum is exact.

    Shapes must agree across ranks -- they do, because ``adjust_batch`` rounds the
    batch to a multiple of ``ppo_micro_batch_size_per_gpu * world_size``, so every
    rank runs the same number of micro-batches of the same size. That is also what
    makes calling a collective from inside the micro-batch loop safe.

    Args:
        cache_ids: (n,) int64 keys for this rank's rows; -1 for unscored rows.
        ids: (n, k) this rank's student-chosen token ids.

    Returns:
        (n, k) float32 teacher log-probs for this rank's ids.
    """
    import torch.distributed as dist

    if world_size is None:
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1

    if world_size == 1:
        values, found = cache.logprobs_at(cache_ids, ids)
        _assert_owned_once(found, cache_ids)
        return values

    n, k = ids.shape
    all_cache_ids = [torch.empty_like(cache_ids) for _ in range(world_size)]
    all_ids = [torch.empty_like(ids) for _ in range(world_size)]
    dist.all_gather(all_cache_ids, cache_ids.contiguous(), group=group)
    dist.all_gather(all_ids, ids.contiguous(), group=group)

    # Answer every rank's request from this rank's cache; zeros elsewhere.
    values = torch.zeros((world_size, n, k), dtype=torch.float32, device=ids.device)
    found = torch.zeros((world_size, n), dtype=torch.int32, device=ids.device)
    for r in range(world_size):
        v, f = cache.logprobs_at(all_cache_ids[r], all_ids[r])
        values[r] = v
        found[r] = f

    dist.all_reduce(values, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(found, op=dist.ReduceOp.SUM, group=group)

    rank = dist.get_rank(group) if dist.is_initialized() else 0
    _assert_owned_once(found[rank], cache_ids)
    return values[rank]


def _assert_owned_once(found: torch.Tensor, cache_ids: torch.Tensor):
    """Every scored row must have been answered by exactly one rank.

    0 is a row whose hidden states nobody kept -- the loss would silently take a
    zero target. 2 is two ranks claiming the same key, which means the ids are not
    unique and some row is being answered from another row's cache.
    """
    wanted = cache_ids >= 0
    if not torch.any(wanted):
        return
    got = found[wanted]
    if not torch.all(got == 1):
        missing = int((got == 0).sum())
        duplicated = int((got > 1).sum())
        raise RuntimeError(
            f"teacher hidden-state exchange did not resolve every row: {missing} unanswered, "
            f"{duplicated} answered more than once (of {int(wanted.sum())} scored). Unanswered rows would "
            f"train against a zero teacher target; duplicated ones mean cache ids are not unique."
        )
