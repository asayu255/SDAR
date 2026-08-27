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
"""Which of these logs is the control?

A filename is a claim. /tmp/eval_pump_ms4.log was read as "the pump run" and is
actually V0 + multi-step 4 + pump, so judging it against a V1 run measured two
changes and credited one. This script reads what each run reported about
itself, so the pair is chosen from the configuration and not from the name.

The fixtures are long for the same reason as test_judge_eval.py: every check
here can exit early, and a short log hides SIGPIPE under pipefail.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

INVENTORY = Path(__file__).resolve().parents[2] / "examples" / "sft_trainer" / "eval_log_inventory.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def write_log(path, *, pump, core, steps, depth, batches, wall, ms_row, score=None):
    prefix = "(SFTMultiTaskTaskRunner pid=1) "
    lines = [
        prefix + f"[rollout-engine] vllm 0.8.5, core={core}; overlap knobs: "
        f"num_scheduler_steps={steps}, async_scheduling=absent"
    ]
    if pump:
        lines.append(prefix + "[rollout-pump] driving 3 ranks as a pool; 3 ranks")
    lines.append(prefix + f"[val-pipeline] VAL_PIPELINE_DEPTH={depth}: {depth} slot(s), threads")
    lines += [prefix + f"noise {i}" for i in range(3000)]
    lines.append(prefix + f"[rollout-turn-timing] WALL 18.5s  ms/row last20={ms_row} all={ms_row}")
    tag = "final" if score else f"after {batches}"
    lines.append(prefix + f"[val-pipeline] {tag}: {batches} batches over {wall}s: at least one slot running")
    if score:
        lines.append(prefix + "Initial validation metrics: {'val/success_rate': %s}" % score)
    path.write_text("\n".join(lines) + "\n")
    return path


def inventory(*logs):
    done = subprocess.run(["bash", str(INVENTORY), *[str(log) for log in logs]], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


def row(out, name):
    for line in out.splitlines():
        if line.startswith(name):
            return line.split()
    raise AssertionError(f"{name} not in:\n{out}")


@pytest.fixture
def logs(tmp_path):
    return [
        write_log(tmp_path / "eval_pump.log", pump=True, core="V1", steps="<default>", depth=3, batches=208, wall=3456.0, ms_row=73, score="0.3875"),
        write_log(tmp_path / "eval_pump_ms4.log", pump=True, core="V0", steps=4, depth=3, batches=208, wall=3847.2, ms_row=73, score="0.3874"),
        write_log(tmp_path / "eval_arraywire.log", pump=True, core="V1", steps="<default>", depth=3, batches=180, wall=3075.4, ms_row=67),
    ]


def test_it_reads_the_engine_the_run_reported_not_the_filename(logs):
    """The whole point: eval_pump_ms4.log says "pump" and is V0 multi-step."""
    out = inventory(*logs)
    assert row(out, "eval_pump_ms4.log")[1:5] == ["ON", "V0", "4", "3"]
    assert row(out, "eval_pump.log")[1:5] == ["ON", "V1", "1", "3"]


def test_a_default_or_absent_knob_reads_as_one_step(logs):
    """num_scheduler_steps=<default> is not multi-step, and must not print as 0."""
    assert row(inventory(*logs), "eval_pump.log")[3] == "1"


def test_an_unfinished_run_is_marked_so_its_wall_is_not_compared(logs):
    out = inventory(*logs)
    assert row(out, "eval_arraywire.log")[5] == "180*"
    assert row(out, "eval_arraywire.log")[-1] == "unfinished"
    assert row(out, "eval_pump.log")[5] == "208"


def test_a_file_that_is_not_an_evaluation_is_skipped(tmp_path, logs):
    stray = tmp_path / "geckodriver.log"
    stray.write_text("1755000000\tgeckodriver\tINFO\tListening on 127.0.0.1:4444\n")
    out = inventory(*logs, stray)
    assert "geckodriver" not in out


def test_a_pump_line_in_a_long_log_is_not_lost_to_sigpipe(logs):
    """Same defect as judge_eval.sh had: it announces the pool on line 2."""
    out = inventory(*logs)
    assert out.count(" ON ") == 3
    assert " off " not in out


def test_a_run_older_than_the_engine_line_reports_no_engine_columns(tmp_path, logs):
    """An unknown engine must not print as "1 step", which reads as measured.

    The 8/26 runs predate [rollout-engine]. Printing STEPS=1 for them would
    have made eval_window.log look like a valid V1 control for a V1 candidate
    when nothing in it says which core it ran.
    """
    old = write_log(tmp_path / "eval_window.log", pump=False, core="V1", steps=1, depth=3,
                    batches=208, wall=4534.0, ms_row=86, score="0.3869")
    old.write_text("\n".join(l for l in old.read_text().splitlines() if "rollout-engine" not in l) + "\n")
    fields = row(inventory(old, *logs), "eval_window.log")
    assert fields[2] == "?" and fields[3] == "?"
    assert fields[5] == "208"  # everything not derived from the engine still reads


def test_a_run_with_no_report_yet_still_says_how_many_batches(tmp_path, logs):
    """"?" reads as "unreadable log"; the run is just young.

    [val-pipeline] prints every VAL_PIPELINE_REPORT_EVERY batches, but the WALL
    lines are per batch and are already there.
    """
    young = write_log(tmp_path / "eval_378.log", pump=True, core="V1", steps="<default>",
                      depth=3, batches=3, wall=0, ms_row=112)
    # Drop the [val-pipeline] report but keep the WALL lines, which is what a
    # run under 25 batches looks like at the default report interval.
    kept = [line for line in young.read_text().splitlines() if "val-pipeline] after" not in line]
    wall_line = next(line for line in kept if "ms/row" in line)
    young.write_text("\n".join(kept + [wall_line, wall_line]) + "\n")

    fields = row(inventory(young, *logs), "eval_378.log")
    assert fields[5] == "3*"          # counted from the WALL lines
    assert fields[6] == "not-yet"     # rather than "?", which reads as unreadable


def test_the_cumulative_and_windowed_speed_are_both_shown(logs):
    """all= is dominated by startup on a young run; last= is not.

    A run three batches in read all=112 against a finished run's all=67, and
    the whole of the difference was ~70 s of env-manager construction and the
    first CUDA graph capture divided by three batches instead of 183.
    """
    out = inventory(*logs)
    assert "ALL/LAST" in out
    assert row(out, "eval_pump.log")[-2] == "73/73"
