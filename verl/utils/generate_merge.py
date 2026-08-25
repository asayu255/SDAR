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
"""Merge generate calls that are already queued behind one another.

A search batch's last turns decode for a handful of trajectories spread over
three ranks, and the ranks do not finish together: measured, twelve active
trajectories gave per-GPU occupancy of 61/89/56, because the call returns only
once the rank holding the longest responses is done and the other two wait. The
seats are there; the work to fill them is in another batch.

The pipeline runs two batches, but deliberately out of phase -- one generates
while the other steps its environment, which is where its 16.5% came from.
Waiting for the second batch to arrive would trade that away. So nothing waits
here: a caller merges only what is ALREADY queued. The worker group is a Ray
actor and serialises its calls, so a second batch reaching generate while the
first is in flight is queued anyway; merging costs it nothing and buys the first
batch's idle ranks its rows.

Merging is only safe between calls that would have been identical apart from
their rows -- same sampling parameters, same tensor widths -- so the caller
supplies a key and calls with the same key never mix.
"""

import threading
from typing import Any, Callable, Dict, List


class _Waiter:
    __slots__ = ("batch", "rows", "result", "error", "done")

    def __init__(self, batch, rows: int):
        self.batch = batch
        self.rows = rows
        self.result = None
        self.error = None
        self.done = False


class GenerateMerger:
    """Coalesce concurrent calls sharing a key into one call, without waiting."""

    def __init__(self, concat: Callable[[List[Any]], Any], split: Callable[[Any, List[int]], List[Any]]):
        self._concat = concat
        self._split = split
        self._cv = threading.Condition()
        self._pending: Dict[Any, List[_Waiter]] = {}
        self._issuing: set = set()
        self.merges = 0
        self.calls = 0
        self.rows_merged = 0

    def call(self, key, batch, rows: int, issue: Callable[[Any], Any]):
        """Run ``issue`` on this batch, possibly merged with others queued now."""
        waiter = _Waiter(batch, rows)
        with self._cv:
            self._pending.setdefault(key, []).append(waiter)
            group = self._claim(key, waiter)
            if group is None:
                # someone else is issuing; wait for them, then re-contend
                while not waiter.done:
                    self._cv.wait()
                    if waiter.done:
                        break
                    group = self._claim(key, waiter)
                    if group is not None:
                        break
        if waiter.done:
            return self._deliver(waiter)

        try:
            merged = group[0].batch if len(group) == 1 else self._concat([w.batch for w in group])
            self.calls += 1
            if len(group) > 1:
                self.merges += 1
                self.rows_merged += sum(w.rows for w in group[1:])
            output = issue(merged)
            parts = [output] if len(group) == 1 else self._split(output, [w.rows for w in group])
            for member, part in zip(group, parts):
                member.result = part
        except BaseException as exc:  # noqa: BLE001 - a leader that dies must not hang its followers
            for member in group:
                member.error = exc
            raise
        finally:
            with self._cv:
                self._issuing.discard(key)
                for member in group:
                    member.done = True
                self._cv.notify_all()
        return self._deliver(waiter)

    def _claim(self, key, waiter):
        """Take every waiter queued under ``key`` if nobody else is issuing."""
        if key in self._issuing:
            return None
        group = self._pending.pop(key, [])
        if waiter not in group:
            # another leader already took this waiter's group
            return None
        self._issuing.add(key)
        return group

    @staticmethod
    def _deliver(waiter):
        if waiter.error is not None:
            raise waiter.error
        return waiter.result
