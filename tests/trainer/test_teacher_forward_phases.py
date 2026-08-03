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
"""``teacher_forward`` is two unlike things, and one of them is not a forward pass.

``build_teacher_batch`` decodes and re-encodes every row on the driver -- two
tokenizer calls per row, GPU parked -- and only then does the batch reach the
GPU. Charged to one bucket named "forward", the driver-side half is invisible,
which is exactly the reading that made §8 of the optimization report call a
communication-bound phase compute-bound.

Covered here: the two phases are pushed in order and always popped, the report
knows how to order them, and the per-task size diagnostics report the cost driver
the wall clock cannot be split by -- including the case where the skill prepend
was truncated away, which quietly turns the teacher into the student.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:
    from verl import DataProto
    from verl.trainer.ppo import rlsd_ray_trainer as mod
    from verl.utils import gpu_profiler
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _batch(prompt_lens, response_len=4, max_prompt_length=16, tasks=None):
    """A DataProto with left-padded prompts of the given valid lengths."""
    n = len(prompt_lens)
    total = max_prompt_length + response_len
    attention_mask = torch.zeros(n, total, dtype=torch.long)
    for row, valid in enumerate(prompt_lens):
        attention_mask[row, max_prompt_length - valid : max_prompt_length] = 1
    attention_mask[:, max_prompt_length:] = 1  # every response fully valid
    tensors = {
        "input_ids": torch.zeros(n, total, dtype=torch.long),
        "attention_mask": attention_mask,
        "responses": torch.zeros(n, response_len, dtype=torch.long),
    }
    batch = DataProto.from_dict(tensors=tensors)
    if tasks is not None:
        batch.non_tensor_batch["task_name"] = np.array(tasks, dtype=object)
    return batch


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #


def test_the_two_phases_are_pushed_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu_profiler, "push_phase", lambda n: calls.append(("push", n)))
    monkeypatch.setattr(gpu_profiler, "pop_phase", lambda n: calls.append(("pop", n)))

    with mod._profiler_phase("teacher_forward/build"):
        pass
    with mod._profiler_phase("teacher_forward/logprob"):
        pass

    assert calls == [
        ("push", "teacher_forward/build"), ("pop", "teacher_forward/build"),
        ("push", "teacher_forward/logprob"), ("pop", "teacher_forward/logprob"),
    ]


def test_a_raising_body_still_pops(monkeypatch):
    """An unbalanced stack mis-tags every later sample in the run."""
    calls = []
    monkeypatch.setattr(gpu_profiler, "push_phase", lambda n: calls.append(("push", n)))
    monkeypatch.setattr(gpu_profiler, "pop_phase", lambda n: calls.append(("pop", n)))

    with pytest.raises(RuntimeError):
        with mod._profiler_phase("teacher_forward/build"):
            raise RuntimeError("boom")

    assert calls == [("push", "teacher_forward/build"), ("pop", "teacher_forward/build")]


def test_the_report_orders_both_sub_phases_under_their_parent():
    """Unordered phases land in the alphabetical tail, which breaks the
    top-to-bottom reading of a step's interior."""
    order = gpu_profiler._PHASE_ORDER
    for name in ("teacher_forward/build", "teacher_forward/logprob"):
        assert name in order, name
    assert order.index("teacher_forward") < order.index("teacher_forward/build")
    assert order.index("teacher_forward/build") < order.index("teacher_forward/logprob")
    assert order.index("teacher_forward/logprob") < order.index("ref")


def test_the_phases_used_are_the_phases_ordered():
    """Read from the file: the trainer's methods are long and getsource on the
    class would not tell us which names actually reach the profiler."""
    source = open(mod.__file__.replace(".pyc", ".py")).read()
    used = {
        line.split('_profiler_phase("')[1].split('"')[0]
        for line in source.splitlines()
        if '_profiler_phase("' in line and not line.strip().startswith("def ")
    }
    assert used, "the teacher forward is not instrumented"
    assert used <= set(gpu_profiler._PHASE_ORDER), used - set(gpu_profiler._PHASE_ORDER)


# --------------------------------------------------------------------------- #
# per-task size diagnostics
# --------------------------------------------------------------------------- #


def test_prompt_tokens_and_overhead_are_reported_per_task():
    student = _batch([4, 4, 10, 10], tasks=["alfworld", "alfworld", "webshop", "webshop"])
    # alfworld gains 3 tokens of skill text, webshop gains 1.
    teacher = _batch([7, 7, 11, 11], tasks=["alfworld", "alfworld", "webshop", "webshop"])

    m = mod._teacher_batch_metrics(student, teacher, max_prompt_length=16)

    assert m["teacher_forward/prompt_tokens/mean/alfworld"] == pytest.approx(7.0)
    assert m["teacher_forward/prompt_tokens/mean/webshop"] == pytest.approx(11.0)
    assert m["teacher_forward/skill_overhead_tokens/mean/alfworld"] == pytest.approx(3.0)
    assert m["teacher_forward/skill_overhead_tokens/mean/webshop"] == pytest.approx(1.0)
    assert m["teacher_forward/rows/alfworld"] == 2
    assert m["teacher_forward/rows/webshop"] == 2
    # the batch-wide numbers are the mean over all rows, not over tasks
    assert m["teacher_forward/prompt_tokens/mean"] == pytest.approx(9.0)
    assert m["teacher_forward/skill_overhead_tokens/mean"] == pytest.approx(2.0)


def test_a_truncated_skill_prepend_is_reported():
    """The prepend sits at the START and the truncation keeps the END, so a row
    at the cap is one whose privileged information was cut away -- the teacher is
    then running on the student's own prompt and the distillation signal is not
    what the recipe says it is."""
    student = _batch([16, 4], tasks=["webshop", "alfworld"])
    teacher = _batch([16, 7], tasks=["webshop", "alfworld"])  # webshop pinned at the cap

    m = mod._teacher_batch_metrics(student, teacher, max_prompt_length=16)

    assert m["teacher_forward/skill_truncated_ratio/webshop"] == pytest.approx(1.0)
    assert m["teacher_forward/skill_truncated_ratio/alfworld"] == pytest.approx(0.0)
    assert m["teacher_forward/skill_truncated_ratio"] == pytest.approx(0.5)
    # and the overhead it reports for that row is 0, which is the honest answer
    assert m["teacher_forward/skill_overhead_tokens/mean/webshop"] == pytest.approx(0.0)


def test_a_batch_without_task_names_still_reports_the_totals():
    """Single-task runs carry no task_name; the per-task keys just do not appear."""
    student = _batch([4, 6])
    teacher = _batch([9, 11])

    m = mod._teacher_batch_metrics(student, teacher, max_prompt_length=16)

    assert m["teacher_forward/skill_overhead_tokens/mean"] == pytest.approx(5.0)
    assert not [k for k in m if k.startswith("teacher_forward/rows/")]
