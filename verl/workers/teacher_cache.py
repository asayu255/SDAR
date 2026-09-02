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

import os
import torch


# WHERE THE STEP'S HIDDEN STATES LIVE WHILE THE ACTOR TRAINS.
#
# The cache is filled before update_actor and read all the way through it, so on
# the cross-teacher arm its 16.3 GB (four models a row, measured) sits on the card
# for the whole backward. That arm reached 99% of both cards during update_actor
# with ppo_micro_batch_size_per_gpu already halved to 5 -- the store is the single
# largest thing in the step that is not the model, and none of it is touched
# between micro batches.
#
# Offloaded, the store is pinned host memory and each micro batch pulls only the
# rows it asks for: (rows x response_length x hidden) bf16, about 10 MB at a
# micro batch of 5, against 16.3 GB resident. Over a step that is a few GB across
# PCIe, tenths of a second against update_actor's 261 s.
#
# BIT-IDENTICAL. A bf16 copy host-to-device is exact, and the gather and the GEMM
# still run on the device on the same values. What changes is only where the
# bytes wait.
_OFFLOAD = os.environ.get("TEACHER_CACHE_OFFLOAD", "1").strip().lower() not in ("0", "false", "no", "off")
# Ceiling on the projection gather inside teacher_logprobs_from_hidden; see
# _projection_rows. Above what the arm asks for today, so it changes nothing
# until the micro batch or the support grows.
_PROJ_CHUNK_BYTES = int(float(os.environ.get("TEACHER_PROJ_CHUNK_MB", "64")) * (1 << 20))
# WHETHER THE OFFLOADED STORE IS PAGE-LOCKED. On by default because that is what
# makes the per-micro-batch pull a DMA the copy engine can overlap, and the store
# is pulled from thousands of times a step.
#
# It is off-able because the page-locked cost is not the store's size. torch
# rounds a pinned request up to a power of two and never gives the block back, so
# a store that swings 7.0-10.3 GB a rank (teacher_cache/gb 14.0-20.7 over run
# n9zfny6m) settles into an 8.6 GiB block AND a 17.2 GiB one: ~24 GiB a rank,
# ~48 GB on a two-card box, for 9.6 GiB of data. On tamago that is a fifth of the
# node, and the node is what run e8x57zyu died on.
#
# Unpinned the store is ordinary host memory the OS takes back, and _read_packed's
# copies become staged and synchronous -- ~1-2% of a step by the shapes here.
# That is a trade to make when host RAM is the binding constraint, which is why it
# is a knob and not a rewrite.
_PIN_STORE = os.environ.get("TEACHER_CACHE_PIN_STORE", "1").strip().lower() not in ("0", "false", "no", "off")


def store_placement(offload: bool, read_device, cuda_available: Optional[bool] = None, pin: Optional[bool] = None):
    """``(store_device, pin_memory, index_device)`` for a finalized store.

    A function, and not three expressions inline in ``_finalize``, because a CPU
    test box cannot tell the two placements apart any other way: with the read
    device already ``cpu`` every branch below collapses to the same answer, and a
    test that only checks where the tensors ended up passes just as happily when
    the offload does nothing at all.

    The index device follows the STORE, not the read: indexing a host tensor with
    a cuda index is not allowed, and sending the index the other way would be a
    device-to-host sync per micro batch -- which is what this class's whole layout
    exists to avoid.

    Pinning is a property of the READ device: it only buys a DMA, and
    ``torch.empty(pin_memory=True)`` raises outright with no CUDA driver. It is
    also a property of how much host RAM there is to spend -- see ``_PIN_STORE``,
    which ``pin`` overrides for tests.
    """
    read = torch.device(read_device)
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if pin is None:
        pin = _PIN_STORE
    if not offload:
        return read, False, read
    host = torch.device("cpu")
    return host, bool(pin and read.type == "cuda" and cuda_available), host


_PROCESS_CACHE: Optional["TeacherHiddenCache"] = None


def packed_plan(lens: torch.Tensor, offs: torch.Tensor, resp_len: int):
    """Where each live position goes, for a read that skips the padding.

    HOST ONLY, in and out. ``lens`` and ``offs`` are the micro batch's per-row
    live length and store offset, and the caller is responsible for handing over
    the copies that already live on the host -- which offloaded they do, because
    :func:`store_placement` put the index there. That is what makes the
    ``.tolist()`` below a host read rather than the device-to-host sync this
    class's whole layout exists to keep out of the micro-batch loop. Kept pure
    and CPU-side so the arithmetic can be checked on a box with no driver, the
    way ``store_placement`` is: a plan that misplaces a row returns a real
    teacher log-prob for the wrong position, which is silent.

    Returns ``(dest, row, spans, m)`` or None when the batch owns no rows:

    ``dest``   (m,) where packed row j belongs in the flattened (n, resp_len)
               grid the caller gets back.
    ``row``    (m,) which of the n entries packed row j came from, for the
               per-row columns (vocab offset, temperature) the GEMM needs.
    ``spans``  ``[(store_offset, length)]`` per owned row. An entry's positions
               are contiguous in the store, so this is the whole pull: one slice
               copy each, straight out of pinned memory, rather than a gather.
    ``m``      total live positions.
    """
    m = int(lens.sum())
    if m == 0:
        return None
    row = torch.repeat_interleave(torch.arange(lens.numel(), dtype=torch.long), lens)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)[:-1]])
    dest = row * resp_len + (torch.arange(m, dtype=torch.long) - starts[row])
    spans = [(int(o), int(n)) for o, n in zip(offs.tolist(), lens.tolist()) if n]
    return dest, row, spans, m


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
    chunk_bytes: Optional[int] = None,
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
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.to(h.device).float().reshape(-1, 1)

    rows = ids.shape[0]
    step = _projection_rows(rows, ids.shape[-1], lm_head_weight, chunk_bytes)
    if step >= rows:
        return _project(h, lse, lm_head_weight, ids, temperature)
    # Row-independent, so the split is exact rather than approximate: each chunk
    # sees the same h, the same ids and the same normaliser it would have seen
    # whole. Only the peak differs.
    out = torch.empty((rows, ids.shape[-1]), dtype=torch.float32, device=h.device)
    for a in range(0, rows, step):
        b = min(a + step, rows)
        t = temperature[a:b] if isinstance(temperature, torch.Tensor) else temperature
        out[a:b] = _project(h[a:b], lse[a:b], lm_head_weight, ids[a:b], t)
    return out


def _project(h, lse, lm_head_weight, ids, temperature):
    # (n, k, hidden) gathered rows of the projection. Built per micro-batch and
    # dropped: at step scale this would be ~90 GB, at micro-batch scale ~90 MB.
    # Kept in the inputs' own dtype rather than widened to float32 -- it is by far
    # the largest tensor here, and widening bf16 that the forward already produced
    # doubles the traffic without recovering any precision: the products are exact
    # either way and both paths accumulate in float32.
    w_ids = lm_head_weight[ids]
    dtype = torch.promote_types(h.dtype, w_ids.dtype)
    logits = torch.einsum("nh,nkh->nk", h.to(dtype), w_ids.to(dtype)).float()
    if isinstance(temperature, torch.Tensor):
        logits = logits / temperature
    elif temperature != 1.0:
        logits = logits / float(temperature)
    return logits - lse.float()


def _projection_rows(rows: int, k: int, lm_head_weight: torch.Tensor, chunk_bytes=None) -> int:
    """How many rows to project at once, from a byte budget for the gather.

    ``lm_head_weight[ids]`` is (rows, k, hidden) and is the largest tensor the
    micro-batch loop allocates -- 52 MB packed at a micro batch of 5, k=20 and a
    2048-wide teacher, taken and returned eight times a micro batch. It scales
    with the micro batch while the activations it competes with do too, so it is
    what makes a larger micro batch cost more than the arithmetic says.

    The budget is a CEILING and the default (64 MB) is above what the arm asks
    for today: nothing chunks at the current settings, and the loop below stays a
    single call. It is what lets ppo_micro_batch_size_per_gpu go back up without
    the transient going up with it -- at 10 the gather would be 104 MB, and this
    holds it at two passes of 52. Set TEACHER_PROJ_CHUNK_MB=0 to disable.
    """
    if chunk_bytes is None:
        chunk_bytes = _PROJ_CHUNK_BYTES
    if chunk_bytes <= 0:
        return rows
    per_row = max(int(k) * lm_head_weight.shape[-1] * lm_head_weight.element_size(), 1)
    return max(1, int(chunk_bytes) // per_row)


def row_fingerprint(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """A cheap identity for a row, computed the same way on both sides.

    Masked so it does not depend on how far the row happens to be padded, and
    mixed with the token count so two rows that merely share a token multiset
    still differ. It is not a hash against an adversary -- it only has to make two
    DIFFERENT trajectory rows disagree, which a sum over a few thousand
    six-figure token ids does comfortably.
    """
    mask = attention_mask.to(torch.long)
    ids = input_ids.to(torch.long) * mask
    return ids.sum(-1) * 1000003 + mask.sum(-1)


class TeacherHiddenCache:
    """Process-local store of one step's teacher hidden states.

    Keyed by an int64 the driver assigns when it queues a row for scoring, so the
    key survives every reordering between the teacher call and the actor update.
    One entry per row; ``owner`` is implicit -- whichever rank has the key.
    """

    def __init__(self):
        self._offload = _OFFLOAD
        self._h: Dict[int, torch.Tensor] = {}
        self._lse: Dict[int, torch.Tensor] = {}
        self._len: Dict[int, int] = {}
        self._task: Dict[int, str] = {}
        self._temperature: Dict[int, float] = {}
        self._fingerprint: Dict[int, int] = {}
        self._witness_ids: Dict[int, torch.Tensor] = {}
        self._witness_lp: Dict[int, torch.Tensor] = {}
        # Entries are views into one packed tensor per put call. Held here too so
        # _finalize can release each source the moment its last row has been
        # copied into the contiguous store, instead of holding every source until
        # the concatenation is finished.
        self._chunks: Dict[int, list] = {}
        self._chunk_of: Dict[int, int] = {}
        self._chunk_refs: Dict[int, int] = {}
        self._next_chunk = 0
        self._weights: Dict[str, torch.Tensor] = {}
        self._stacked = None   # (n_tasks * vocab, hidden), built on first read
        self._voff: Dict[str, int] = {}
        self._final = None     # the read-side tensors; invalidated by put/clear

    # -- registration ----------------------------------------------------- #

    def register_lm_head(self, task: str, weight: torch.Tensor, slot=None, n_tasks=None):
        """Keep an unsharded copy of a teacher's output projection.

        Needed because the ref path reshards after every call, so by the time the
        actor update runs the sharded parameter cannot be indexed at arbitrary
        ids. One copy per teacher, held for the run.

        With ``slot``/``n_tasks`` the copy lands straight in its slice of the
        stacked projection. Registering separately and stacking later means both
        layouts exist at once -- ~1.9 GB of duplicate for a 1.7B teacher, and it
        peaks while the allocator is still deciding how much vLLM gets. The caller
        knows how many teachers there are, so it can say.
        """
        if n_tasks is not None:
            v, hidden = weight.shape
            if self._stacked is None:
                self._stacked = torch.empty((n_tasks * v, hidden), dtype=weight.dtype, device=weight.device)
            elif self._stacked.shape != (n_tasks * v, hidden):
                raise ValueError(
                    f"teacher lm_head {task} does not fit the stacked projection: expected "
                    f"{tuple(self._stacked.shape)} for {n_tasks} teachers, got vocab {v} x hidden {hidden}"
                )
            off = int(slot) * v
            self._stacked[off : off + v].copy_(weight)
            self._voff[task] = off
            self._weights[task] = self._stacked[off : off + v]
        else:
            if self._voff:
                raise ValueError("teachers must all register with a slot or all without; this mixes the two")
            self._weights[task] = weight
            # A new teacher shifts every other one's slice of the stacked
            # projection, so any offsets already baked in are no longer valid.
            self._stacked = None
        self._final = None

    def _stack_heads(self):
        """One (n_tasks * vocab, hidden) projection, teachers laid end to end.

        A micro-batch mixes tasks -- ``_balance_batch`` sorts by token count, not
        by teacher -- so answering it from three separate weights means grouping
        its rows by task, and grouping means a boolean mask, a ``nonzero``, and a
        device-to-host sync inside the micro-batch loop. Offsetting the ids by the
        teacher's slice instead makes the whole lookup one gather with no
        branching. The memory is the same 1.9 GB the three copies already cost;
        only the layout changes.

        The fallback path for callers that registered without a slot: it builds
        the same thing, but transiently holds both layouts.
        """
        if self._stacked is not None:
            return
        if not self._weights:
            raise KeyError("no teacher lm_head registered; see register_lm_head")
        tasks = sorted(self._weights)
        vocab = {int(self._weights[t].shape[0]) for t in tasks}
        if len(vocab) != 1:
            raise ValueError(f"teachers disagree on vocabulary size: {vocab}; the stacked projection assumes one")
        v = vocab.pop()
        self._voff = {t: i * v for i, t in enumerate(tasks)}
        self._stacked = torch.cat([self._weights[t] for t in tasks], dim=0)
        # The parts are now views' worth of duplicate memory; drop them so the
        # stacked copy is the only one held.
        self._weights = {t: self._stacked[self._voff[t] : self._voff[t] + v] for t in tasks}

    def lm_head(self, task: str) -> torch.Tensor:
        if task not in self._weights:
            raise KeyError(f"no lm_head registered for teacher '{task}'; registered: {sorted(self._weights)}")
        return self._weights[task]

    # -- filling ---------------------------------------------------------- #

    def put(
        self, cache_ids, task: str, h, lse, witness_ids=None, witness_lp=None, temperature: float = 1.0,
        live_mask=None, witness_rows=None, fingerprints=None,
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

        ``fingerprints`` is what the row WAS, independent of the key it is filed
        under. The witness proves the stored h/lse still reproduce the teacher's
        output; the ownership count proves exactly one rank answered. Neither
        proves the key on the asking side names the row the asker is training --
        a ``teacher_cache_ids`` column shifted against its batch passes both. The
        fingerprint closes that: it is derived from the row's own tokens, so a key
        pointing at another row disagrees, and it is an identity check rather than
        a numeric one so no tolerance enters.
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
                    keys[i], task, h.new_empty((0, h.shape[-1])), lse.new_empty((0,)), None, None, 0, temperature,
                    chunk=None, fingerprint=None if fingerprints is None else int(fingerprints[i]),
                )
            return
        row_of = torch.repeat_interleave(torch.arange(len(kept), device=dev), lens)
        flat = rows[row_of] * resp_len + (torch.arange(total, device=dev) - offsets[row_of])

        # One gather each; the per-key entries below are views into these, so the
        # cache holds exactly the packed size and the padded input is free to go.
        h_packed = h.reshape(n * resp_len, -1)[flat]
        lse_packed = lse.reshape(n * resp_len)[flat]
        # OFFLOADED MEANS OFFLOADED NOW, NOT AT THE FIRST READ. _finalize used to
        # be where the store crossed to the host, and _finalize runs on the first
        # read -- which is inside the actor update, long after the rollout has
        # ended. Everything cached during a rollout therefore sat in device memory
        # NEXT TO vLLM's KV pool for the whole rollout, and on the cross-teacher
        # arms that is four models a row rather than one (teacher_cache/gb 15.6
        # against pure OPD's 4.1). Copying here bounds the device side at one
        # chunk's transient instead, which is what makes prefetching the other
        # three planes affordable at all.
        #
        # Blocking, and deliberately: the destination is read later on the host
        # (check_witness, _finalize) with nothing in between that would order an
        # async copy against those reads. It runs on the prefetch thread, under
        # the rollout's CPU glue, which is the window this whole path exists for.
        if self._offload:
            h_packed, lse_packed = self._to_host(h_packed), self._to_host(lse_packed)
        chunk = self._next_chunk
        self._next_chunk += 1
        self._chunks[chunk] = [h_packed, lse_packed]
        self._chunk_refs[chunk] = 0

        # The witness is a SAMPLE of rows, not a column: it is the teacher's own
        # top-k, which nothing downstream reads, and building it costs a
        # full-vocabulary selection. ``witness_rows`` says which of this call's
        # rows carry it; without it, every row does (the old shape).
        w_of_row = None
        if witness_ids is not None:
            if witness_rows is None:
                w_of_row = {i: i for i in range(n)}
            else:
                w_of_row = {int(r): j for j, r in enumerate(witness_rows.tolist())}

        lens_l, off_l = lens.tolist(), offsets.tolist()
        for j, i in enumerate(kept):
            a, b = off_l[j], off_l[j + 1]
            w_ids = w_lp = None
            if w_of_row is not None and i in w_of_row:
                w = w_of_row[i]
                w_ids = witness_ids[w, : lens_l[j]]
                w_lp = witness_lp[w, : lens_l[j]]
            self._register(
                keys[i], task, h_packed[a:b], lse_packed[a:b], w_ids, w_lp, lens_l[j], temperature, chunk=chunk,
                fingerprint=None if fingerprints is None else int(fingerprints[i]),
            )

    def _to_host(self, t: torch.Tensor) -> torch.Tensor:
        """A host copy of ``t``. PAGEABLE, and that is the whole point.

        This chunk is written once and read twice: ``check_witness`` samples a
        row or two, and ``_finalize`` copies it into the contiguous store and
        drops it. Nothing else ever touches it -- ``_read_packed``, which runs
        once per micro-batch of every training step, pulls from ``final["h"]``,
        a SEPARATE tensor that is pinned on its own account. So pinning here
        would buy a DMA on one copy per chunk and nothing after that.

        What it costs is unbounded. torch's pinned allocator rounds every request
        up to the next power of two, keys its free list by that bucket, and NEVER
        returns a block to the OS -- on torch 2.8 (what setup.py pins) unconditionally,
        and on 2.14 unless ``pinned_max_cached_size`` is set, which defaults to
        SIZE_MAX. ``put`` runs many times a step -- once per teacher per prefetch
        window -- with a different row count each time, and every chunk stays live
        until ``_finalize`` drains it, so the pool has to hold all of them at once
        and grows to the worst arrangement any step has yet produced. Measured:
        host RAM went from +0.37 GB/step (run n9zfny6m, 148 steps, plateauing) to
        +5.06 GB/step (run e8x57zyu), which took the node from 207 GB at step 1 to
        Ray's 0.98 kill threshold at step 10.

        Pageable host memory goes through the ordinary allocator, which hands
        large blocks back at free. The copy is a staged transfer instead of a DMA:
        ~8 GB a rank a step at a few GB/s, on the prefetch thread inside the
        rollout's CPU glue, against a step of ~250 s.
        """
        out = torch.empty(t.shape, dtype=t.dtype, device="cpu")
        out.copy_(t)
        return out

    def _register(self, key, task, h, lse, w_ids, w_lp, length, temperature, chunk=None, fingerprint=None):
        self._h[key] = h
        self._lse[key] = lse
        self._len[key] = int(length)
        self._task[key] = task
        self._temperature[key] = float(temperature)
        self._fingerprint[key] = 0 if fingerprint is None else int(fingerprint)
        if w_ids is not None:
            self._witness_ids[key] = w_ids
            self._witness_lp[key] = w_lp
        if chunk is not None:
            self._chunk_of[key] = chunk
            self._chunk_refs[chunk] += 1
        self._final = None

    def clear(self):
        self._h.clear()
        self._lse.clear()
        self._len.clear()
        self._task.clear()
        self._temperature.clear()
        self._fingerprint.clear()
        self._witness_ids.clear()
        self._witness_lp.clear()
        self._chunks.clear()
        self._chunk_of.clear()
        self._chunk_refs.clear()
        self._next_chunk = 0
        self._final = None

    def __len__(self):
        return len(self._h)

    def __contains__(self, key):
        return int(key) in self._h

    def nbytes(self, device_only: bool = False) -> int:
        """What the step's entries hold, for the metric that watches it.

        ``device_only`` counts the CUDA-resident part alone. Since ``put`` copies
        to the host as it goes, an offloaded cache holds ~nothing on the device
        between calls, and reporting the total under a name that used to mean
        "device" would read as a regression that is in fact the fix. Both numbers
        are logged, as ``teacher_cache/gb`` and ``teacher_cache/device_gb``.

        Neither is the PAGE-LOCKED footprint, and they should not be read as it:
        torch rounds a pinned request up to a power of two and keeps the block, so
        the store's 9.6 GB is 17.2 GB of pinned pages. That number is
        ``perf/pinned_host_gb``.

        Worth a number rather than an estimate because the sign-weighting arms
        multiply it: they cache the base policy and each off-task teacher besides
        the on-task one, so a row carries four models' hidden states instead of
        one, on a box whose vLLM engine is already sized to 0.6 of the card. The
        entries are views into the packed sources, so those are what is summed --
        adding up the views would count the same memory once per row.
        """
        total = 0

        def _add(t):
            if device_only and t.device.type == "cpu":
                return 0
            return t.numel() * t.element_size()

        for tensors in self._chunks.values():
            for t in tensors:
                total += _add(t)
        if self._final is not None:
            for t in self._final.values():
                if hasattr(t, "numel"):
                    total += _add(t)
        return total

    # -- reading ---------------------------------------------------------- #

    def _finalize(self, device):
        """Lay the step's entries out as tensors, once, so reads can be branch-free.

        Everything below is per STEP -- the fill is finished by the time the actor
        update starts, and ``put``/``clear`` invalidate this. What matters is what
        is NOT here: ``logprobs_at`` runs inside the micro-batch loop, thousands of
        times a step, and a ``.tolist()`` or an ``int(t)` there is a
        device-to-host sync that stalls the pipeline the whole rollout-overlap
        effort exists to keep full.

        The store is filled row by row rather than concatenated, and each entry is
        re-pointed at its slice as it is copied. Two reasons, both about memory:
        a concatenation holds every source until it finishes, and leaving the
        entries pointing at the old sources would keep the WHOLE cache alive twice
        for the length of the actor update. Copying in key order drains one put's
        packed tensor at a time -- keys are handed out in order, so a source is
        released about as soon as it is exhausted -- and the peak is the store plus
        one chunk instead of the store plus all of them.
        """
        if self._final is not None and self._final["device"] == device:
            return self._final
        self._stack_heads()
        keys = sorted(self._h)
        if not keys:
            self._final = {"device": device, "empty": True}
            return self._final

        base = keys[0]
        _store_dev, _pin, index_dev = store_placement(self._offload, device)
        span = keys[-1] - base + 1
        key_to_slot = torch.full((span,), -1, dtype=torch.long)
        key_to_slot[torch.tensor(keys, dtype=torch.long) - base] = torch.arange(len(keys), dtype=torch.long)

        lens = torch.tensor([self._len[k] for k in keys], dtype=torch.long)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)[:-1]])
        total = int(lens.sum())
        probe = self._h[keys[0]]
        # Pinned host memory when offloading, because THIS is the tensor the
        # micro-batch loop pulls from -- once per micro batch of every training
        # step, which is where a DMA the copy engine can overlap is worth having.
        # (The put() chunks are not: they are read once and dropped. See _to_host.)
        store_dev, pin, _index_dev = store_placement(self._offload, device)
        store_h = torch.empty(
            (total, probe.shape[-1]), dtype=probe.dtype, device=store_dev, pin_memory=pin
        )
        store_lse = torch.empty(
            (total,), dtype=self._lse[keys[0]].dtype, device=store_dev, pin_memory=pin
        )
        off_l = offsets.tolist()
        for i, k in enumerate(keys):
            a, b = off_l[i], off_l[i] + self._len[k]
            store_h[a:b].copy_(self._h[k])
            store_lse[a:b].copy_(self._lse[k])
            self._h[k] = store_h[a:b]
            self._lse[k] = store_lse[a:b]
            chunk = self._chunk_of.pop(k, None)
            if chunk is not None:
                self._chunk_refs[chunk] -= 1
                if self._chunk_refs[chunk] == 0:
                    # Last row out of that put's packed tensor; nothing references
                    # it now, so the allocator gets it back before the next copy.
                    del self._chunks[chunk], self._chunk_refs[chunk]

        self._final = {
            "device": device,
            "empty": False,
            "base": base,
            # Offloaded, the row/slot arithmetic runs on the HOST -- indexing a
            # cpu tensor with a cuda index is not allowed, and moving the index
            # the other way would be a device-to-host sync per micro batch, which
            # is exactly what this class's layout exists to avoid. The metadata is
            # a few kB either way. The per-row columns the CALLER needs on the
            # device (temp, voff, fp) are kept on both sides.
            "key_to_slot": key_to_slot.to(index_dev),
            "slot_key": torch.tensor(keys, dtype=torch.long).to(index_dev),
            "slot_len": lens.to(index_dev),
            "slot_off": offsets.to(index_dev),
            "slot_voff": torch.tensor([self._voff[self._task[k]] for k in keys], dtype=torch.long).to(device),
            "slot_temp": torch.tensor([self._temperature[k] for k in keys], dtype=torch.float32).to(device),
            "slot_fp": torch.tensor([self._fingerprint.get(k, 0) for k in keys], dtype=torch.long).to(device),
            "h": store_h,
            "lse": store_lse,
            "max_len": int(lens.max()),
            "offloaded": bool(self._offload),
            "index_dev": index_dev,
        }
        return self._final

    @property
    def offloaded(self) -> bool:
        """Whether the finalized store lives in host memory. Read by
        ``exchange_teacher_logprobs`` to decide whether a host copy of the keys is
        worth making -- resident, it would be a sync bought for nothing."""
        return bool(self._offload)

    def logprobs_at(self, cache_ids: torch.Tensor, ids: torch.Tensor, cache_ids_cpu=None):
        """Answer for the rows this cache owns; leave the rest at zero.

        Sync-free either way, and no grouping by task -- the ids carry the
        teacher's offset into the stacked projection instead.

        Resident, it is also branch-free and has no per-row Python: a key lookup,
        a gather of the stored rows, one narrow GEMM, and a mask. Offloaded it
        packs first (see :meth:`_read_packed`), which costs a Python loop over
        the micro batch's rows and buys back the padding -- roughly four fifths
        of the bytes and of the projection gather. That loop reads lengths the
        offloaded index already holds on the HOST, so it is not the sync the
        resident path would pay for the same thing.

        Args:
            cache_ids: (n,) int64 keys, one per ROW; -1 means "not scored here".
            ids: (n, response_length, k) token ids to evaluate.

        Returns:
            values: (n, response_length, k) float32, zero on rows this cache does
                not own and on padding positions.
            found: (n,) int32, 1 on the rows it does.
            fingerprint: (n,) int64, what the owner recorded the row's tokens as;
                0 where it does not own the row. The caller compares it against
                the row it is actually training.
        """
        n, resp_len, k = ids.shape
        dev = ids.device
        final = self._finalize(dev)
        if final["empty"]:
            return (
                torch.zeros((n, resp_len, k), dtype=torch.float32, device=dev),
                torch.zeros((n,), dtype=torch.int32, device=dev),
                torch.zeros((n,), dtype=torch.long, device=dev),
            )
        if final["max_len"] > resp_len:
            # The reconstruction indexes a flattened (n, resp_len) grid, so a
            # longer entry would spill into the next row rather than fail.
            raise RuntimeError(
                f"teacher cache holds a row of {final['max_len']} response positions but is being asked for "
                f"{resp_len}; the teacher and the actor disagree on response_length."
            )

        # The slot arithmetic runs where the store's index has to be: the host
        # when offloaded, the device otherwise. cache_ids_cpu lets the caller hand
        # over a copy it already has -- four planes a micro batch share one on the
        # cross-teacher arm -- instead of each read paying its own sync.
        idev = final["index_dev"]
        if final["offloaded"]:
            cid = (cache_ids_cpu if cache_ids_cpu is not None else cache_ids.cpu()).to(torch.long)
        else:
            cid = cache_ids.to(dev, torch.long)
        q = cid - final["base"]
        span = final["key_to_slot"].numel()
        slot = torch.where(
            (q >= 0) & (q < span),
            final["key_to_slot"][q.clamp(0, span - 1)],
            torch.full_like(q, -1),
        )
        safe = slot.clamp(min=0)
        ok = (slot >= 0) & (final["slot_key"][safe] == cid)
        safe = torch.where(ok, safe, torch.zeros_like(safe))

        lens = torch.where(ok, final["slot_len"][safe], torch.zeros_like(safe))
        if final["offloaded"]:
            # The ragged plan is worked out on the host, where the index already
            # is, and only then do the row columns move to the device.
            plan = packed_plan(lens, final["slot_off"][safe], resp_len)
            safe = safe.to(dev)
            ok = ok.to(dev)
            values = self._read_packed(final, ids, safe, plan, dev)
        else:
            pos = torch.arange(resp_len, device=idev)
            live = pos.unsqueeze(0) < lens.unsqueeze(1)                   # (n, resp_len)
            src = final["slot_off"][safe].unsqueeze(1) + pos.unsqueeze(0)
            src = torch.where(live, src, torch.zeros_like(src)).reshape(-1)   # (n * resp_len,)
            h_rows, lse_rows = final["h"][src], final["lse"][src]
            flat_ids = (
                ids.reshape(-1, k)
                + final["slot_voff"][safe].unsqueeze(1).expand(n, resp_len).reshape(-1, 1)
            )
            out = teacher_logprobs_from_hidden(
                h_rows, lse_rows, self._stacked, flat_ids,
                temperature=final["slot_temp"][safe].unsqueeze(1).expand(n, resp_len).reshape(-1),
            )
            values = torch.where(live.reshape(-1, 1), out, out.new_zeros(())).view(n, resp_len, k)

        found = ok.to(torch.int32)
        fingerprint = torch.where(ok, final["slot_fp"][safe], torch.zeros_like(safe))
        return values, found, fingerprint

    def _read_packed(self, final, ids, safe, plan, dev):
        """The offloaded read, over the LIVE positions only. (n, resp_len, k).

        The padded form asks for ``resp_len`` positions a row where a turn
        generates about a quarter of them, and it asks for them as a scattered
        gather into a pageable tensor -- which torch then has to stage through its
        own bounce buffer, so the ``pin_memory`` on the store buys nothing and the
        copy blocks. Packing first fixes both: an entry's rows are CONTIGUOUS in
        the store, so the pull becomes one slice copy per row straight out of
        pinned memory, which is a DMA the copy engine can overlap.

        Three things shrink by ``resp_len / mean_len``, about 4x here: the bytes
        crossing the bus, the host-side work (measured 1.41 ms -> 0.09 ms for a
        micro batch of 5), and -- on the device, which is the point -- the
        projection gather inside :func:`teacher_logprobs_from_hidden`. That
        ``(rows, k, hidden)`` tensor is by far the largest thing the micro-batch
        loop allocates, 210 MB padded at k=20 against 52 MB packed, taken and
        returned eight times a micro batch. It is the churn ``alloc_retries``
        counts.

        Only available offloaded, and not by accident: the plan needs the lengths
        as Python ints, and offloaded they are already on the host. Resident they
        are on the device, where reading them is the device-to-host sync in the
        micro-batch loop this whole class is laid out to avoid.
        """
        n, resp_len, k = ids.shape
        if plan is None:
            return torch.zeros((n, resp_len, k), dtype=torch.float32, device=dev)
        dest, row, spans, m = plan
        dest = dest.to(dev, non_blocking=True)
        row = row.to(dev, non_blocking=True)

        h_pack = torch.empty((m, final["h"].shape[-1]), dtype=final["h"].dtype, device=dev)
        lse_pack = torch.empty((m,), dtype=final["lse"].dtype, device=dev)
        at = 0
        for off, length in spans:
            h_pack[at : at + length].copy_(final["h"][off : off + length], non_blocking=True)
            lse_pack[at : at + length].copy_(final["lse"][off : off + length], non_blocking=True)
            at += length

        packed_ids = ids.reshape(-1, k).index_select(0, dest) + final["slot_voff"][safe][row].unsqueeze(1)
        out = teacher_logprobs_from_hidden(
            h_pack, lse_pack, self._stacked, packed_ids,
            temperature=final["slot_temp"][safe][row],
        )
        # Padding positions are reconstructed as zeros, which is the value they
        # had: they were never stored, and the loss masks them anyway.
        values = torch.zeros((n * resp_len, k), dtype=torch.float32, device=dev)
        values.index_copy_(0, dest, out)
        return values.view(n, resp_len, k)

    def check_witness(self, atol: float = 1e-3, ulps: float = 8.0) -> float:
        """Recompute at the teacher's own top-k and return the largest deviation.

        The teacher builds its top-k for a couple of micro-batches a step anyway,
        so this costs one narrow GEMM over what is already cached. Consistent
        h/lse/W reproduce those values to the precision they were stored at; an
        entry paired with the wrong row is off by whole nats. Raises rather than
        reporting, because a cache that fails this has been feeding wrong targets
        to the loss.

        The tolerance is relative to the LOGIT, not to the log-prob, and that is
        the whole subtlety. The stored witness is ``logit - lse`` where both came
        out of a bfloat16 forward, so it carries a bfloat16 quantum of |logit| --
        0.25 nats where |logit| is 32, five orders of magnitude above the 1e-3
        this used to demand. That is what the first real run tripped on, and no
        recomputation can do better: the reference itself is that coarse. What a
        recomputation CAN be held to is the storage precision, which is
        ``ulps * eps(dtype) * |logit|``. A row scored from another row's entry
        misses by ~10 nats at those magnitudes, ~40 eps, so the check still fires
        by a wide margin -- and unlike rounding it misses on nearly every
        position, which the message reports so the two are told apart on sight.
        """
        worst = worst_ratio = 0.0
        over = total = 0
        for key, w_ids in self._witness_ids.items():
            # Every position of the row, not just one: a cache that keeps a single
            # position per row is exactly the failure this has to catch, and it is
            # self-consistent at whichever position survived. Entries are already
            # packed to the real positions, so this is all of them.
            h, lse = self._h[key], self._lse[key]          # (len, H), (len,)
            if h.shape[0] == 0:
                continue
            if h.device.type == "cpu" and self._stacked is not None:
                # Offloaded store, device projection. One witness row, once a step.
                h = h.to(self._stacked.device)
                lse = lse.to(self._stacked.device)
            got = teacher_logprobs_from_hidden(
                h, lse, self.lm_head(self._task[key]), w_ids.to(h.device),
                temperature=self._temperature.get(key, 1.0),
            )
            want = self._witness_lp[key].to(got.device).float()
            err = (got - want).abs()
            # |logit| = |log p + lse|, the magnitude the stored value was rounded at.
            scale = (want + lse.to(want.device).float().unsqueeze(-1)).abs()
            eps = torch.finfo(h.dtype).eps if h.dtype.is_floating_point else 0.0
            tol = atol + ulps * eps * scale
            worst = max(worst, float(err.max()))
            worst_ratio = max(worst_ratio, float((err / tol).max()))
            over += int((err > tol).sum())
            total += err.numel()
        if worst_ratio > 1.0:
            raise RuntimeError(
                f"teacher hidden-state cache failed its witness check: worst deviation {worst:.3e}, "
                f"{worst_ratio:.1f}x the storage precision it was stored at, on {over}/{total} positions "
                f"({100.0 * over / max(total, 1):.1f}%). Rounding misses a handful of positions by ~1x; a "
                f"cache entry paired with the wrong row misses nearly all of them by tens of x. If this says "
                f"nearly all, the ids being scored do not belong to the rows they are filed under."
            )
        return worst


def exchange_teacher_logprobs(
    cache: TeacherHiddenCache,
    cache_ids: torch.Tensor,
    ids: torch.Tensor,
    group=None,
    world_size: Optional[int] = None,
    fingerprints: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Get ``log p_t`` at each rank's ids from whichever rank cached that row.

    One plane. See :func:`exchange_teacher_logprobs_multi`, which this is a
    single-column call into -- there is one implementation so the two cannot
    drift, and on the arms that read four models the multi form is what the
    actor calls.

    Args:
        cache_ids: (n,) int64 keys, one per ROW; -1 for rows scored elsewhere.
        ids: (n, response_length, k) this rank's student-chosen token ids.
        fingerprints: (n,) int64 derived from the tokens of the rows THIS rank is
            training. The owner's fingerprint comes back with the values and must
            match. Without it the key is taken on trust: the witness proves the
            stored states still reproduce the teacher, and the ownership count
            proves one rank answered, but neither notices a ``teacher_cache_ids``
            column shifted against its batch -- the answer is then a real teacher
            log-prob for the wrong row, which is the failure mode this whole
            module exists to make loud.

    Returns:
        (n, response_length, k) float32 teacher log-probs for this rank's ids.
    """
    return exchange_teacher_logprobs_multi(
        cache, cache_ids.unsqueeze(1), ids,
        group=group, world_size=world_size, fingerprints=fingerprints,
    )[0]


def exchange_teacher_logprobs_multi(
    cache: TeacherHiddenCache,
    cache_ids: torch.Tensor,
    ids: torch.Tensor,
    group=None,
    world_size: Optional[int] = None,
    fingerprints: Optional[torch.Tensor] = None,
) -> list:
    """The same exchange for SEVERAL models at once, over one support.

    Every rank broadcasts what it wants, every rank answers what it owns, and the
    answers are summed back. Ownership is unique so the sum is exact.

    Shapes must agree across ranks -- they do, because ``adjust_batch`` rounds the
    batch to a multiple of ``ppo_micro_batch_size_per_gpu * world_size``, so every
    rank runs the same number of micro-batches of the same size. That is also what
    makes calling a collective from inside the micro-batch loop safe.

    THE PLANES SHARE THEIR IDS, WHICH IS THE WHOLE REASON TO BATCH THEM. The
    cross-teacher arms read four models -- the on-task teacher, the base policy
    and two off-task teachers -- at ONE support the student just chose. Run one
    plane at a time that support is all-gathered four times, and it is by far the
    largest thing that crosses: (n, response_length, k) int64 against (n,) keys.
    Four calls also meant four ``all_reduce`` pairs and, offloaded, four
    device-to-host copies of the gathered keys -- and a sync inside the
    micro-batch loop drains the CPU's run-ahead every time. Sixteen collectives a
    micro-batch become four, on a box whose two cards have no NVLink and where
    the boundary costs more than the bytes.

    Values are bit-identical to calling the single-plane form P times:
    ``logprobs_at`` is invoked per (rank, plane) on exactly the same inputs, and
    the reduction is elementwise over the same ranks in the same order.

    Args:
        cache_ids: (n, P) int64 keys, one per ROW per PLANE; -1 for rows scored
            elsewhere. Column order is the caller's and is carried through.
        ids: (n, response_length, k) this rank's student-chosen token ids, shared
            by every plane.
        fingerprints: (n,) int64 as in the single-plane form. One row is one row
            whichever model is being read off it, so the same column checks all
            P planes.

    Returns:
        list of P tensors, each (n, response_length, k) float32, in column order.
    """
    import torch.distributed as dist

    if world_size is None:
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1
    n_planes = cache_ids.size(1)

    if world_size == 1:
        # One host copy of the keys for all P planes, which is the sync the
        # per-plane calls were each paying for themselves.
        cpu = cache_ids.to("cpu", non_blocking=False) if cache.offloaded else None
        out = []
        for p in range(n_planes):
            values, found, owner_fp = cache.logprobs_at(
                cache_ids[:, p], ids, cache_ids_cpu=None if cpu is None else cpu[:, p]
            )
            _record_ownership(found, cache_ids[:, p], owner_fp, fingerprints)
            out.append(values)
        return out

    n, resp_len, k = ids.shape
    all_cache_ids = [torch.empty_like(cache_ids) for _ in range(world_size)]
    all_ids = [torch.empty_like(ids) for _ in range(world_size)]
    dist.all_gather(all_cache_ids, cache_ids.contiguous(), group=group)
    # Once, not once per plane: this is the tensor the batching is for.
    dist.all_gather(all_ids, ids.contiguous(), group=group)

    # Answer every rank's request from this rank's cache; zeros elsewhere.
    values = torch.zeros(
        (world_size, n_planes, n, resp_len, k), dtype=torch.float32, device=ids.device
    )
    # (found, fingerprint) in one tensor so the identity check rides along on the
    # reduce the count already needed -- no extra collective in the micro-batch loop.
    meta = torch.zeros((world_size, n_planes, n, 2), dtype=torch.int64, device=ids.device)
    # One host copy for the whole gathered set, not one per rank and not one per
    # plane: offloaded, the slot arithmetic runs on the host, and doing it lazily
    # inside logprobs_at would be world_size * P device-to-host syncs per micro
    # batch instead of world_size. Skipped entirely when the store is resident,
    # where the index belongs on the device and no copy is wanted.
    gathered_cpu = (
        [t.to("cpu", non_blocking=False) for t in all_cache_ids]
        if cache.offloaded else [None] * world_size
    )
    for r in range(world_size):
        for p in range(n_planes):
            v, f, fp = cache.logprobs_at(
                all_cache_ids[r][:, p], all_ids[r],
                cache_ids_cpu=None if gathered_cpu[r] is None else gathered_cpu[r][:, p],
            )
            values[r, p] = v
            meta[r, p, :, 0] = f
            meta[r, p, :, 1] = fp

    dist.all_reduce(values, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(meta, op=dist.ReduceOp.SUM, group=group)

    rank = dist.get_rank(group) if dist.is_initialized() else 0
    for p in range(n_planes):
        _record_ownership(meta[rank, p, :, 0], cache_ids[:, p], meta[rank, p, :, 1], fingerprints)
    return [values[rank, p] for p in range(n_planes)]


_OWNERSHIP: Optional[torch.Tensor] = None


def _record_ownership(found, cache_ids, owner_fingerprint=None, fingerprints=None):
    """Tally rows that were not answered by exactly one rank, or by the wrong row.

    0 is a row whose hidden states nobody kept -- the loss would silently take a
    zero target. 2 is two ranks claiming the same key, which means the ids are not
    unique and some row is being answered from another row's cache. A fingerprint
    mismatch is the third way and the quietest: exactly one rank answered, with a
    real teacher log-prob, for a different row than the asker is training.

    Counted on the device rather than asserted here, because "here" is inside the
    micro-batch loop: reading a count is a synchronisation, and one per
    micro-batch drains the CPU's run-ahead thousands of times a step. The tally is
    read once per mini-batch by ``assert_rows_were_owned_once``, which still runs
    BEFORE the optimizer step, so a bad row can never reach the weights.
    """
    global _OWNERSHIP
    wanted = cache_ids.to(found.device) >= 0
    if fingerprints is None or owner_fingerprint is None:
        mismatched = torch.zeros((), dtype=torch.bool, device=found.device)
    else:
        mismatched = wanted & (found == 1) & (owner_fingerprint != fingerprints.to(found.device))
    counts = torch.stack(
        [
            (wanted & (found == 0)).sum(),
            (wanted & (found > 1)).sum(),
            wanted.sum(),
            mismatched.sum(),
        ]
    ).to(torch.int64)
    if _OWNERSHIP is None or _OWNERSHIP.device != counts.device:
        _OWNERSHIP = counts
    else:
        _OWNERSHIP = _OWNERSHIP + counts


def assert_rows_were_owned_once():
    """Read the tally and raise if anything went unresolved. Clears it.

    One synchronisation per call, so call it per mini-batch (before the optimizer
    step), not per micro-batch.
    """
    global _OWNERSHIP
    if _OWNERSHIP is None:
        return
    missing, duplicated, scored, mismatched = (int(x) for x in _OWNERSHIP)
    _OWNERSHIP = None
    if missing or duplicated or mismatched:
        raise RuntimeError(
            f"teacher hidden-state exchange did not resolve every row: {missing} unanswered, "
            f"{duplicated} answered more than once, {mismatched} answered for a DIFFERENT row "
            f"(of {scored} scored). Unanswered rows would train against a zero teacher target; "
            f"duplicated ones mean cache ids are not unique; a fingerprint mismatch means the key "
            f"resolved cleanly but to another row's tokens, which is a real teacher log-prob "
            f"attached to the wrong sample."
        )
