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
    temperature=1.0,
) -> torch.Tensor:
    """``log p_t`` at ``ids``, from the teacher's hidden states and its lse.

    Equal to indexing a full ``log_softmax(h @ W.T / T)`` at the same ids, in
    exact arithmetic: the projection is per-position linear and the normaliser is
    the one the teacher already computed over the whole vocabulary. Only the GEMM
    shape differs (``h @ W[ids].T`` instead of ``h @ W.T``), which moves the last
    bits the way any repacking does.

    ``temperature`` must be the same one the forward applied before taking its
    logsumexp, because ``lse`` normalises the SCALED logits. ``h`` is cached raw,
    so the scaling has to be redone here; forgetting it leaves the two halves of
    ``logit - lse`` on different scales, which is silent at T=1 and wrong
    everywhere else.

    Args:
        h: (n, hidden) teacher hidden states at the scored positions.
        lse: (n,) or (n, 1) logsumexp over the FULL vocabulary, from the teacher
            forward that produced ``h``.
        lm_head_weight: (vocab, hidden) the teacher's output projection.
        ids: (n, k) token ids to evaluate.
        temperature: scalar, or (n,) / (n, 1) to allow a mixed batch.

    Returns:
        (n, k) float32 log-probs.
    """
    if lse.dim() == 1:
        lse = lse.unsqueeze(-1)
    # (n, k, hidden) gathered rows of the projection. Built per micro-batch and
    # dropped: at step scale this would be ~90 GB, at micro-batch scale ~90 MB.
    w_ids = lm_head_weight[ids]
    logits = torch.einsum("nh,nkh->nk", h.float(), w_ids.float())
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.to(logits.device).float().reshape(-1, 1)
        logits = logits / temperature
    elif temperature != 1.0:
        logits = logits / float(temperature)
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
        self._len: Dict[int, int] = {}
        self._task: Dict[int, str] = {}
        self._temperature: Dict[int, float] = {}
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

    def put(
        self, cache_ids, task: str, h, lse, witness_ids=None, witness_lp=None, temperature: float = 1.0,
        live_mask=None,
    ):
        """Store one call's rows, one entry per ROW, packed to the real positions.

        A row is scored once but the student picks a top-k at every response
        position, so an entry has to hold all of them: ``h`` arrives as
        (n, response_length, hidden) and ``lse`` as (n, response_length). Keying
        per position instead -- repeating the row's id across its positions --
        collapses the row to whichever position was written last, and does so
        silently, because the witness stored under the same key collapses with it
        and stays self-consistent.

        What is NOT kept is the padding. ``response_length`` is the cap (512 here)
        while a turn generates ~127 tokens, so three quarters of the padded form
        is memory the loss never reads. Rows are gathered down to their real
        prefix in one kernel and the entries are views into that; padding
        positions are reconstructed as zeros on the way out, which is the value
        they had.

        ``live_mask`` should be the attention mask over the same window. Without
        it the real positions are inferred from a non-zero normaliser, which is
        what ``pad_input`` leaves behind -- true, but inference rather than the
        fact. It is taken as a MASK and not as a length so that the prefix check
        below has something to disagree with.

        ``temperature`` travels with the entry because ``lse`` normalises the
        scaled logits while ``h`` is raw; reading the entry back has to apply the
        same scaling.
        """
        if h.dim() != 3 or lse.dim() != 2:
            raise ValueError(
                f"expected per-row (n, response_length, hidden) / (n, response_length), got {tuple(h.shape)} / "
                f"{tuple(lse.shape)}; a flattened cache silently keeps one position per row"
            )
        n, resp_len = lse.shape
        dev = h.device
        slot = torch.arange(resp_len, device=dev)

        real = (lse != 0) if live_mask is None else live_mask.to(dev).bool()
        if real.shape != lse.shape:
            raise ValueError(f"live_mask {tuple(real.shape)} does not match lse {tuple(lse.shape)}")
        lens_t = real.sum(-1)
        # Reconstruction assumes a prefix: the response is right-padded and the
        # window opens on the last prompt token, so the live slots run 0..len-1.
        # If that ever stops holding, the rows would be silently misaligned on the
        # way back out, so check rather than assume.
        if not torch.equal(real, slot.unsqueeze(0) < lens_t.unsqueeze(1)):
            raise RuntimeError(
                "teacher cache expects each row's live response positions to be a prefix (right-padded "
                "responses); this batch has holes, so packing them would misalign the reconstruction."
            )

        keys = [int(c) for c in cache_ids]
        kept = []
        for i, key in enumerate(keys):
            if key < 0:
                continue
            if key in self._h:
                raise RuntimeError(
                    f"teacher cache id {key} written twice on this rank. Ids are assigned once per row, so a "
                    f"repeat means a row was duplicated into this call -- most likely DP padding that was not "
                    f"marked -1."
                )
            kept.append(i)
        if not kept:
            return

        rows = torch.tensor(kept, dtype=torch.long, device=dev)
        lens = lens_t[rows]
        offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), lens.cumsum(0)])
        total = int(offsets[-1])
        if total == 0:
            # Every kept row is pure padding. Still register them: the exchange
            # requires an owner for each key it is asked about, and "owned, empty"
            # is a different answer from "nobody has it". Freshly allocated rather
            # than a zero-length slice, which would be a view and would pin the
            # whole padded input for the step.
            for i in kept:
                self._register(
                    keys[i], task, h.new_empty((0, h.shape[-1])), lse.new_empty((0,)), None, None, 0, temperature
                )
            return
        row_of = torch.repeat_interleave(torch.arange(len(kept), device=dev), lens)
        flat = rows[row_of] * resp_len + (torch.arange(total, device=dev) - offsets[row_of])

        # One gather each; the per-key entries below are views into these, so the
        # cache holds exactly the packed size and the padded input is free to go.
        h_packed = h.reshape(n * resp_len, -1)[flat]
        lse_packed = lse.reshape(n * resp_len)[flat]
        w_ids_packed = witness_ids.reshape(n * resp_len, -1)[flat] if witness_ids is not None else None
        w_lp_packed = witness_lp.reshape(n * resp_len, -1)[flat] if witness_lp is not None else None

        lens_l, off_l = lens.tolist(), offsets.tolist()
        for j, i in enumerate(kept):
            a, b = off_l[j], off_l[j + 1]
            self._register(
                keys[i], task, h_packed[a:b], lse_packed[a:b],
                None if w_ids_packed is None else w_ids_packed[a:b],
                None if w_lp_packed is None else w_lp_packed[a:b],
                lens_l[j], temperature,
            )

    def _register(self, key, task, h, lse, w_ids, w_lp, length, temperature):
        self._h[key] = h
        self._lse[key] = lse
        self._len[key] = int(length)
        self._task[key] = task
        self._temperature[key] = float(temperature)
        if w_ids is not None:
            self._witness_ids[key] = w_ids
            self._witness_lp[key] = w_lp

    def clear(self):
        self._h.clear()
        self._lse.clear()
        self._len.clear()
        self._task.clear()
        self._temperature.clear()
        self._witness_ids.clear()
        self._witness_lp.clear()

    def __len__(self):
        return len(self._h)

    def __contains__(self, key):
        return int(key) in self._h

    # -- reading ---------------------------------------------------------- #

    def logprobs_at(self, cache_ids: torch.Tensor, ids: torch.Tensor):
        """Answer for the rows this cache owns; leave the rest at zero.

        Args:
            cache_ids: (n,) int64 keys, one per ROW; -1 means "not scored here".
            ids: (n, response_length, k) token ids to evaluate.

        Returns:
            values: (n, response_length, k) float32, zero on rows this cache does
                not own.
            found: (n,) int32, 1 on the rows it does.
        """
        n, resp_len, k = ids.shape
        values = torch.zeros((n, resp_len, k), dtype=torch.float32, device=ids.device)
        found = torch.zeros((n,), dtype=torch.int32, device=ids.device)
        if not self._h:
            return values, found

        # One host round-trip for the whole request, not one per row: cache_ids
        # may be a device tensor and int() on it synchronises.
        wanted = cache_ids.tolist()
        by_task: Dict[str, list] = {}
        for row, key in enumerate(wanted):
            key = int(key)
            if key < 0 or key not in self._h:
                continue
            by_task.setdefault(self._task[key], []).append((row, key))

        dev = ids.device
        flat_values = values.view(n * resp_len, k)
        for task, entries in by_task.items():
            rows = torch.tensor([r for r, _ in entries], dtype=torch.long, device=dev)
            found[rows] = 1
            # Entries are packed to their real positions, so the work here is
            # packed too: padding slots keep the zero they were already given, and
            # the narrow GEMM never sees them.
            stored = [self._len[key] for _, key in entries]
            if max(stored) > resp_len:
                # The reconstruction indexes a flattened (n, resp_len) grid, so a
                # longer entry would spill into the next row rather than fail.
                raise RuntimeError(
                    f"teacher cache holds a row of {max(stored)} response positions but is being asked for "
                    f"{resp_len}; the teacher and the actor disagree on response_length."
                )
            lens = torch.tensor(stored, dtype=torch.long, device=dev)
            offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), lens.cumsum(0)])
            total = int(offsets[-1])
            if total == 0:
                continue
            row_of = torch.repeat_interleave(torch.arange(len(entries), device=dev), lens)
            flat = rows[row_of] * resp_len + (torch.arange(total, device=dev) - offsets[row_of])

            h = torch.cat([self._h[key] for _, key in entries]).to(dev)      # (T, H)
            lse = torch.cat([self._lse[key] for _, key in entries]).to(dev)  # (T,)
            temps = torch.tensor(
                [self._temperature.get(key, 1.0) for _, key in entries], dtype=torch.float32, device=dev
            )
            flat_values[flat] = teacher_logprobs_from_hidden(
                h, lse, self.lm_head(task), ids.reshape(n * resp_len, k)[flat],
                temperature=temps.repeat_interleave(lens),
            )
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
            # Every position of the row, not just one: a cache that keeps a single
            # position per row is exactly the failure this has to catch, and it is
            # self-consistent at whichever position survived. Entries are already
            # packed to the real positions, so this is all of them.
            h, lse = self._h[key], self._lse[key]          # (len, H), (len,)
            if h.shape[0] == 0:
                continue
            got = teacher_logprobs_from_hidden(
                h, lse, self.lm_head(self._task[key]), w_ids.to(h.device),
                temperature=self._temperature.get(key, 1.0),
            )
            err = (got - self._witness_lp[key].to(got.device).float()).abs().max().item()
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
        cache_ids: (n,) int64 keys, one per ROW; -1 for rows scored elsewhere.
        ids: (n, response_length, k) this rank's student-chosen token ids.

    Returns:
        (n, response_length, k) float32 teacher log-probs for this rank's ids.
    """
    import torch.distributed as dist

    if world_size is None:
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1

    if world_size == 1:
        values, found = cache.logprobs_at(cache_ids, ids)
        _assert_owned_once(found, cache_ids)
        return values

    n, resp_len, k = ids.shape
    all_cache_ids = [torch.empty_like(cache_ids) for _ in range(world_size)]
    all_ids = [torch.empty_like(ids) for _ in range(world_size)]
    dist.all_gather(all_cache_ids, cache_ids.contiguous(), group=group)
    dist.all_gather(all_ids, ids.contiguous(), group=group)

    # Answer every rank's request from this rank's cache; zeros elsewhere.
    values = torch.zeros((world_size, n, resp_len, k), dtype=torch.float32, device=ids.device)
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
