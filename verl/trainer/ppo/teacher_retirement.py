# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Retire each task's teacher when its term stops pulling with the reward.

WHAT DECIDES IT. ``opd/<task>/grpo/first_order`` is the teacher term's first-order
effect on the reward objective, in units of that objective's own step
(cross_teacher_kl_weight.gradient_metrics). One step's sign is noisy -- on the
rho=0.1 control it is positive on 75% of ALFWorld's steps, 80% of Search's and 45%
of WebShop's -- so the rule reads the mean over a WINDOW of steps, and only after a
WARMUP during which the cosine is still finding its sign.

WHY 25 AND 50. Measured on that control's 300 steps, sweeping the window with the
same rule (mean < 0 after step 50):

    W       5     10    15    25    40    50
    ALFWorld 271   -     -     -     -     -
    WebShop   84  133   137   145   156   165
    Search    90   -     -     -     -     -

W = 5 fires on both HEALTHY tasks -- at five points a run of negative steps is
ordinary. From W = 10 the decision is the same whatever the window (WebShop only),
and the cost of a longer one is lag: a trailing mean fires about W/2 steps later.
25 sits 2.5x above the false-alarm floor and pays 12 steps for it, against a
collapse that reaches validation 150 steps later. The window is a config key
because that reasoning rests on ONE collapse.

WHAT IT DOES. The task's teacher-KL coefficient goes to zero, for good: this
answers "when should the teacher stop", not "how much should it be worth", and a
coefficient that can come back would confound the two. The state travels with the
checkpoint so a resumed run does not revive a retired teacher.

WHAT IT DELIBERATELY DOES NOT DO. It does not read the format channel
(traj/<task>/think_block_share) or the per-token pushes, and it does not move the
retired task's coefficient to another task. Both were considered and left out: the
first because the alignment rule already fires at 145 where the format damage first
shows at 151, the second because the measurements say there is nowhere to move it
(Search's teacher is inert, ALFWorld's is worth at most a couple of points).
"""

from collections import deque
from typing import Dict, Iterable, List, Optional

__all__ = ["TeacherRetirement", "RETIRE_WINDOW", "RETIRE_WARMUP"]

RETIRE_WINDOW = 25
RETIRE_WARMUP = 50


def _finite(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


class TeacherRetirement:
    """Per-task retirement on the teacher term's measured first-order effect."""

    def __init__(self, tasks: Iterable[str], *, window: int = RETIRE_WINDOW,
                 warmup: int = RETIRE_WARMUP):
        self.tasks: List[str] = [str(t) for t in tasks]
        self.window = int(window)
        self.warmup = int(warmup)
        assert self.window >= 1 and self.warmup >= 0, "window >= 1 and warmup >= 0"
        self.hist: Dict[str, deque] = {t: deque(maxlen=self.window) for t in self.tasks}
        self.retired: Dict[str, int] = {}
        self.seen = 0

    # --- what the actor is told ------------------------------------------ #

    def coef_by_task(self) -> Dict[str, float]:
        """``{task: 1.0}``, and 0.0 for a retired one: a multiplier on its KL term."""
        return {t: (0.0 if t in self.retired else 1.0) for t in self.tasks}

    # --- what it reads --------------------------------------------------- #

    def observe(self, metrics: dict, step: Optional[int] = None) -> Dict[str, float]:
        """Fold one update's metrics in, retire what qualifies, and report.

        ``metrics`` is the actor's own output for the step, which carries
        ``opd/<task>/grpo/first_order`` while that task's teacher is still on. A
        retired task stops producing it (its coefficient is zero), and its window
        is left alone -- there is nothing left to decide.
        """
        self.seen += 1
        step = int(step if step is not None else self.seen)
        out: Dict[str, float] = {}
        for task in self.tasks:
            if task in self.retired:
                out[f"opd/{task}/retired"] = 1.0
                out[f"opd/{task}/retire_step"] = float(self.retired[task])
                continue
            v = _finite(metrics.get(f"opd/{task}/grpo/first_order"))
            if v is not None:
                self.hist[task].append(v)
            w = self.hist[task]
            out[f"opd/{task}/retired"] = 0.0
            out[f"opd/{task}/first_order_window"] = float(len(w))
            if w:
                mean = sum(w) / len(w)
                out[f"opd/{task}/first_order_window_mean"] = mean
                if step >= self.warmup and len(w) == self.window and mean < 0.0:
                    self.retired[task] = step
                    out[f"opd/{task}/retired"] = 1.0
                    out[f"opd/{task}/retire_step"] = float(step)
                    print(f"[teacher-retirement] {task}: the teacher's first-order effect "
                          f"averaged {mean:+.3e} over the last {self.window} updates; its "
                          f"coefficient is 0 from step {step}", flush=True)
        out["opd/retired_tasks"] = float(len(self.retired))
        return out

    # --- state ----------------------------------------------------------- #

    def state_dict(self) -> dict:
        return {"version": 1, "window": self.window, "warmup": self.warmup, "seen": self.seen,
                "retired": dict(self.retired),
                "hist": {t: list(v) for t, v in self.hist.items()}}

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self.seen = int(state.get("seen", 0) or 0)
        self.retired = {str(k): int(v) for k, v in (state.get("retired", {}) or {}).items()}
        for t, v in (state.get("hist", {}) or {}).items():
            if str(t) in self.hist:
                self.hist[str(t)] = deque([float(x) for x in v][-self.window:], maxlen=self.window)
