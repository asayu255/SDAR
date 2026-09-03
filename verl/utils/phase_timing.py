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
"""Running means of a call's phases, printed every N calls from one rank.

The caller cannot print per call: a generate call happens once per turn, which
is tens of thousands of times in an evaluation. A mean over a period says the
same thing and does not bury the log.

On this branch the one consumer is vllm_rollout_spmd, which splits a
generate_sequences call at the engine boundary. The source branch has a second,
worker-side breakdown; it is a diagnostic rather than a speedup and was not
ported, so nothing here declares the phases that test pinned.

Kept out of the rollout module so it can be tested without a GPU or a vllm
install -- which matters more than it looks. vllm_rollout_spmd is unimportable
in CI for want of vllm, so the fact that it names this module is checked by
reading it: see tests/utils/test_no_dangling_verl_imports.py, written after this
file was left behind by the port that introduced the import.
"""

from typing import Callable, Dict, Optional, Sequence


def _default_rank() -> Optional[int]:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:  # noqa: BLE001 - a print must not break what it observes
        return None
    return None


class PhaseTimer:
    """Sum per-call phase times; emit the mean every ``every`` calls.

    ``every <= 0`` disables the print while still accumulating, so a caller can
    turn the reporting off without a second branch at the call site.
    """

    def __init__(
        self,
        label: str,
        phases: Sequence[str],
        every: int = 50,
        note: str = "",
        rank: Optional[Callable[[], Optional[int]]] = None,
        printer: Callable[[str], None] = print,
    ):
        self.label = label
        self.phases = tuple(phases)
        self.every = every
        self.note = note
        self._rank = rank or _default_rank
        self._print = printer
        self.totals: Dict[str, float] = dict.fromkeys(self.phases, 0.0)
        self.calls = 0

    def record(self, marks: Dict[str, float]) -> None:
        """Fold one call in. A phase the caller did not reach counts as zero."""
        for name in self.phases:
            self.totals[name] += marks.get(name, 0.0)
        self.calls += 1
        if self.every <= 0 or self.calls % self.every:
            return
        rank = self._rank()
        if rank not in (0, None):
            return
        self._print(self.line())

    def line(self) -> str:
        n = max(1, self.calls)
        parts = "  ".join(f"{name} {self.totals[name] / n:.3f}" for name in self.phases)
        note = f"  {self.note}" if self.note else ""
        return (
            f"[{self.label}] rank 0, mean over {self.calls} calls (s): {parts}  "
            f"total {sum(self.totals.values()) / n:.3f}{note}"
        )


def mark(marks: Optional[Dict[str, float]], name: str, start: float) -> float:
    """Record the span since ``start`` under ``name``; return the new start.

    Returns ``start`` unchanged when timing is off, so the call site stays
    branch-free.
    """
    if marks is None:
        return start
    import time

    now = time.perf_counter()
    marks[name] = now - start
    return now
