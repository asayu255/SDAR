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
actor plumbing. The sweep cannot free any of them and visits all of them anyway,
every time the gen-2 threshold trips.

Why that shows up as a GPU dip rather than as a slow host: the sweep freezes the
interpreter, so no kernel is launched while it runs. If the launch queue were
deep that would be invisible, but the forward has synchronisation points that
drain it (see ``docs/spike_investigation.md`` 3.2), so a host stop lands on the
device as an equally long stop. The rank stops submitting, its card falls to
sm 0, and the other two sit in the collective spinning at 100 -- which is
exactly the signature that was measured: 14 solo excursions of 0.6-0.8 s.

Two knobs, because they carry different risk:

  ``ACTOR_GC_FREEZE=1``  (default) One ``gc.collect()`` then ``gc.freeze()``
                         after the model, optimizer and FSDP wrap are built.
                         ``freeze`` moves everything currently alive into the
                         permanent generation, which no collection ever visits
                         again. Cheap and close to free of risk: the live set at
                         that moment IS the set that lives for the whole run, so
                         nothing collectable is being hidden. Later allocations
                         are unaffected and still collected normally.
  ``ACTOR_GC_MANUAL=1``  (off) Additionally ``gc.disable()``, and run one
                         explicit ``gc.collect()`` per step at the boundary --
                         where the device is idle anyway (``before-step`` is
                         0.24-0.71 s of measured idle). This moves the residual
                         cost into time that is already lost instead of removing
                         it. Off by default because disabling the automatic
                         collector means a cycle-heavy step can grow the heap
                         without bound between boundaries, and that trade wants
                         a measurement behind it rather than a default.

Turning the freeze off restores stock behaviour exactly, so the pair is also the
A/B: ``stall/gc_gen2`` is already logged per rank per step, and it should drop to
~0 with the freeze on. If the solo excursions drop with it, the mechanism is
settled; if ``gc_gen2`` goes to zero and the excursions stay, this is ruled out
and the next candidate is the allocator.
"""

import gc
import os
import time

__all__ = ["freeze_permanent_heap", "collect_at_step_boundary", "freeze_enabled", "manual_enabled"]


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() not in ("0", "false", "no", "")


def freeze_enabled() -> bool:
    return _flag("ACTOR_GC_FREEZE", "1")


def manual_enabled() -> bool:
    return _flag("ACTOR_GC_MANUAL", "0")


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


def collect_at_step_boundary() -> float:
    """One sweep per step, in time the device has already lost. Seconds spent.

    A no-op unless ``ACTOR_GC_MANUAL`` asked for the automatic collector to be
    off; with it on, Python's own schedule is still in charge and forcing an
    extra full collection here would add work rather than move it.
    """
    if not (freeze_enabled() and manual_enabled()):
        return 0.0
    started = time.perf_counter()
    gc.collect()
    return time.perf_counter() - started
