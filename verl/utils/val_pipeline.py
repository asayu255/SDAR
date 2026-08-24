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
"""Keep more than one validation batch in flight, retire them in order.

A validation batch is a turn loop: tokenise, generate, step the environment,
repeat. Only the generate is on the GPU. Measured on the multitask SFT arm after
the retriever was batched, one search batch is 11.87 s of generation against
1.20 s of tokenising and 1.17 s of environment -- so 16.5% of the run is the GPU
waiting for work that cannot be brought forward, because the next turn's prompt
is the environment's answer to this turn.

Nothing inside one batch can fill that. Another batch can: batches are
independent, so while one waits on its environment the other can generate. The
worker group is a Ray actor and runs one call at a time, so the two generations
serialise on their own -- what overlaps is one batch's environment and
tokenising against the other's generation, which is exactly the gap.

Order is preserved. Results are yielded in submission order, and the caller does
its accumulation on the calling thread, so nothing downstream of this sees a
concurrent run: the same batches produce the same rows in the same sequence.

A slot carries the resources one batch needs to itself -- its own environment
manager and trajectory collector, since both hold per-rollout state. Slots may
be restricted to certain tasks: alfworld draws its episodes from a seeded
game-file cycle indexed by position within its manager, so a second manager
would score different games and it does not get one. Search has no such state
(each row carries its own question), which is the 411 batches of 413 that
matter.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional


class Slot:
    """One batch's worth of dedicated resources.

    ``tasks=None`` means the slot can run any batch; a set restricts it. Two
    batches must never share a slot at the same time -- the environment manager
    keeps per-rollout state (observation history, per-env step counters) that a
    second concurrent rollout would corrupt.
    """

    __slots__ = ("name", "tasks", "envs", "collector")

    def __init__(self, name: str, envs: Any, collector: Any, tasks: Optional[Iterable[str]] = None):
        self.name = name
        self.envs = envs
        self.collector = collector
        self.tasks = None if tasks is None else set(tasks)

    def accepts(self, task: Optional[str]) -> bool:
        return self.tasks is None or task in self.tasks

    def __repr__(self):
        return f"Slot({self.name}, tasks={'any' if self.tasks is None else sorted(self.tasks)})"


def run_pipelined(
    items: Iterable[Any],
    prepare: Callable[[Any], Any],
    task_of: Callable[[Any], Optional[str]],
    launch: Callable[[Any, Slot], Any],
    slots: List[Slot],
):
    """Yield ``(prepared, result)`` in item order, several launches in flight.

    ``prepare`` runs on the calling thread, in item order: it is the part that
    touches shared state (tokeniser, trainer fields) and must stay sequential.
    ``launch`` runs on a worker thread with a slot to itself.

    With one slot nothing is submitted to a thread at all -- the call is inline,
    so a run that does not ask for a pipeline behaves exactly as it did before.
    """
    if len(slots) <= 1:
        slot = slots[0]
        for item in items:
            prepared = prepare(item)
            yield prepared, launch(prepared, slot)
        return

    executor = ThreadPoolExecutor(max_workers=len(slots), thread_name_prefix="val-slot")
    inflight = []  # [(prepared, future, slot)] oldest first -- also the retirement order
    free = list(slots)

    def retire():
        """Finish the oldest batch, hand it back, and free its slot."""
        prepared, future, slot = inflight.pop(0)
        try:
            result = future.result()
        finally:
            free.append(slot)
        return prepared, result

    try:
        for item in items:
            prepared = prepare(item)
            task = task_of(prepared)
            # Retire until a slot this batch can use is free. A batch whose task
            # only one slot serves waits for that slot specifically, which is why
            # this drains rather than picking any free one.
            while True:
                slot = next((candidate for candidate in free if candidate.accepts(task)), None)
                if slot is not None:
                    break
                if not inflight:
                    raise RuntimeError(f"no slot can run task {task!r}: {slots}")
                yield retire()
            free.remove(slot)
            inflight.append((prepared, executor.submit(launch, prepared, slot), slot))
        while inflight:
            yield retire()
    finally:
        # Never leave a rollout running into the caller's next phase; a batch
        # still generating would be holding the worker group and the engine.
        for _, future, _ in inflight:
            future.cancel()
        executor.shutdown(wait=True)
