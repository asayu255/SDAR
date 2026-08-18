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
"""Keep Python's generation-2 sweep off the critical path.

A gen-2 collection walks every container object the interpreter tracks. On a
rank holding a sharded 1.7B model that set is dominated by things that live for
the whole run and can never be garbage: the module tree, the parameter and
optimizer-state tensors, FSDP's flat-parameter bookkeeping, the tokenizer, Ray's
actor plumbing. Measured on this worker: 1.09M tracked objects, and one full
collection over them costs 0.42 s -- inside the 0.6-0.8 s band of the solo GPU
excursions this module exists to remove.

Why a host sweep shows up as a GPU dip: the sweep freezes the interpreter, so no
kernel is launched while it runs. If the launch queue were deep that would be
invisible, but the forward has synchronisation points that drain it (see
``docs/spike_investigation.md``), so a host stop lands on the device as an
equally long stop. The rank stops submitting, its card falls to sm 0, and the
other two sit in the collective spinning at 100 -- exactly the measured
signature.

Three layers, each with its own knob, plus an escape hatch:

  ``ACTOR_GC_FREEZE=1``     (default) One ``gc.collect()`` then ``gc.freeze()``
                            after the model, optimizer scaffolding and FSDP
                            wrap are built. ``freeze`` moves everything
                            currently alive into the permanent generation,
                            which no collection ever visits again. The live set
                            at that moment IS the set that lives for the whole
                            run, so nothing collectable is being hidden; later
                            allocations are unaffected and collected normally.
  ``ACTOR_GC_REFREEZE_STEP=1`` (default) Freeze once more at the boundary after
                            this many steps have completed. The init-time
                            freeze runs before anything has executed, so
                            everything the warm-up step creates and keeps --
                            Dynamo's guards and cache entries, the state Adam
                            allocates lazily on its first step, FSDP's deferred
                            views -- lands in the ordinary generations, and
                            every later sweep walks it. One step in, all of it
                            exists and is as permanent as the model. 0 turns
                            the second freeze off.
  ``ACTOR_GC_BOUNDARY_COLLECT=1`` (default) One ``gc.collect()`` per step at
                            the boundary, with the automatic collector left ON.
                            After the freezes that sweep costs milliseconds, it
                            runs inside the before-step window where the device
                            is idle anyway (measured 0.24-0.71 s), and it
                            drains the survivor count that trips CPython's
                            automatic full collection -- so the mid-forward
                            gen-2, the one that lands on the device, becomes
                            rare, and cheap when it does fire.
  ``ACTOR_GC_MANUAL=1``     (off) Additionally ``gc.disable()``, leaving the
                            boundary sweep as the only collection. Off by
                            default because a cycle-heavy step can then grow
                            the heap without bound between boundaries, and
                            that trade wants a measurement behind it.

``ACTOR_GC_FREEZE=0`` switches every layer off at once and restores stock
behaviour exactly, so the single flag is also the A/B. Judge that A/B by the
solo excursions, not by ``stall/gc_gen2``: freezing does not stop collections,
it empties what they walk. With the permanent set out of the oldest
generation's accounting, CPython's pending>25%-of-total rule can trip *more*
often -- the counter rising while each firing costs milliseconds instead of
~0.4 s is the fix working, not failing.

Frozen objects are only exempt from cycle collection; refcounting still frees
them the moment the last reference drops. What a freeze can leak is a frozen
reference cycle that later becomes garbage -- bounded by what init and one
warm-up step create, which is the trade both freezes make.
"""

import gc
import os
import time

__all__ = [
    "freeze_permanent_heap",
    "refreeze_if_due",
    "collect_at_step_boundary",
    "freeze_enabled",
    "manual_enabled",
    "boundary_collect_enabled",
    "refreeze_step",
]


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() not in ("0", "false", "no", "")


def freeze_enabled() -> bool:
    return _flag("ACTOR_GC_FREEZE", "1")


def manual_enabled() -> bool:
    return _flag("ACTOR_GC_MANUAL", "0")


def boundary_collect_enabled() -> bool:
    return _flag("ACTOR_GC_BOUNDARY_COLLECT", "1")


def refreeze_step() -> int:
    try:
        return int(os.environ.get("ACTOR_GC_REFREEZE_STEP", "1"))
    except ValueError:
        return 1


_boundaries_seen = 0    # update_policy calls observed; call k runs after k completed steps
_refrozen = False


def freeze_permanent_heap() -> dict:
    """Collect once, then move what survived out of the collector's reach.

    Called after the model and optimizer exist and before the first step. Returns
    what it did, so the caller can print one line and the effect is visible in
    the log rather than inferred -- ``frozen`` is the object count that gen-2
    sweeps will no longer walk, which is the whole point and is worth seeing.
    """
    if not freeze_enabled():
        return {"enabled": False, "frozen": 0, "collected": 0, "seconds": 0.0}

    started = time.perf_counter()
    collected = gc.collect()
    gc.freeze()
    frozen = gc.get_freeze_count()
    if manual_enabled():
        gc.disable()
    return {
        "enabled": True,
        "frozen": frozen,
        "collected": collected,
        "manual": manual_enabled(),
        "seconds": time.perf_counter() - started,
    }


def refreeze_if_due() -> "dict | None":
    """The second freeze: once, at the boundary after the warm-up step.

    Called once per update_policy, before the step body, and counts those calls
    itself -- call k happens with k steps completed, so the default of 1 fires
    at the start of step 1, when everything step 0 built and kept exists.
    Returns the report the one time it acts, None otherwise.
    """
    global _boundaries_seen, _refrozen
    completed = _boundaries_seen
    _boundaries_seen += 1

    step = refreeze_step()
    if not freeze_enabled() or _refrozen or step <= 0 or completed < step:
        return None

    started = time.perf_counter()
    before = gc.get_freeze_count()
    collected = gc.collect()
    gc.freeze()
    _refrozen = True
    total = gc.get_freeze_count()
    return {
        "frozen_delta": total - before,
        "frozen_total": total,
        "collected": collected,
        "seconds": time.perf_counter() - started,
    }


def collect_at_step_boundary() -> float:
    """One sweep per step, in time the device has already lost. Seconds spent.

    Runs under the default configuration: after the freezes a full collection
    costs milliseconds, doing it in the measured device-idle before-step window
    makes it free, and promoting the survivors here resets the pending count
    that CPython's automatic full collection triggers on -- which is what keeps
    gen-2 from firing mid-forward, where it lands on the device. Off with the
    freeze (stock behaviour, the A/B) or with ``ACTOR_GC_BOUNDARY_COLLECT=0``;
    manual mode always sweeps here, because it turned the automatic collector
    off and this is the replacement schedule.
    """
    if not freeze_enabled():
        return 0.0
    if not (manual_enabled() or boundary_collect_enabled()):
        return 0.0
    started = time.perf_counter()
    gc.collect()
    return time.perf_counter() - started
