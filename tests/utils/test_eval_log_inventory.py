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

import os
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
    """The named row as a dict keyed by the table's own header.

    By name, not by position. These indexed fields[2], fields[5] and so on, and
    inserting a WIDTH column broke four tests that had nothing to do with the
    width -- failures that say only "the columns moved".
    """
    lines = out.splitlines()
    header = next(l for l in lines if l.startswith("LOG")).split()
    for line in lines:
        if line.startswith(name):
            return dict(zip(header, line.split()))
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
    r = row(out, "eval_pump_ms4.log")
    assert [r["PUMP"], r["CORE"], r["STEPS"], r["DEPTH"]] == ["ON", "V0", "4", "3"]
    r = row(out, "eval_pump.log")
    assert [r["PUMP"], r["CORE"], r["STEPS"], r["DEPTH"]] == ["ON", "V1", "1", "3"]


def test_a_default_or_absent_knob_reads_as_one_step(logs):
    """num_scheduler_steps=<default> is not multi-step, and must not print as 0."""
    assert row(inventory(*logs), "eval_pump.log")["STEPS"] == "1"


def test_an_unfinished_run_is_marked_so_its_wall_is_not_compared(logs):
    out = inventory(*logs)
    assert row(out, "eval_arraywire.log")["BATCHES"] == "180*"
    assert row(out, "eval_arraywire.log")["SCORE"] == "unfinished"
    assert row(out, "eval_pump.log")["BATCHES"] == "208"


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
    assert fields["CORE"] == "?" and fields["STEPS"] == "?"
    assert fields["BATCHES"] == "208"  # everything not from the engine still reads


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
    assert fields["BATCHES"] == "3*"      # counted from the WALL lines
    assert fields["WALL"] == "not-yet"    # rather than "?", which reads as unreadable


def test_the_cumulative_and_windowed_speed_are_both_shown(logs):
    """all= is dominated by startup on a young run; last= is not.

    A run three batches in read all=112 against a finished run's all=67, and
    the whole of the difference was ~70 s of env-manager construction and the
    first CUDA graph capture divided by three batches instead of 183.
    """
    out = inventory(*logs)
    assert "ALL/LAST" in out
    assert row(out, "eval_pump.log")["ALL/LAST"] == "73/73"


# --------------------------------------------------------------------------- #
# The width and the KV budget are one decision
# --------------------------------------------------------------------------- #
EVAL_SH = Path(__file__).resolve().parents[2] / "examples" / "sft_trainer" / "eval_checkpoints.sh"
RUN_SH = Path(__file__).resolve().parents[2] / "examples" / "sft_trainer" / "run_multitask_sft_qwen3.sh"
PINNED = Path(__file__).resolve().parents[2] / "examples" / "sft_trainer" / "expected_multitask_sft_config.yaml"


def _width_from(path, pattern):
    """re.MULTILINE, so an anchored pattern cannot match a commented example.

    It already did: the pinned file carries an illustrative
    "#   "env.multitask.val_per_task_batch_size": {...}" line above the real
    one, and an unanchored search found the comment's number.
    """
    import re
    m = re.search(pattern, path.read_text(), re.M)
    assert m, f"{pattern} not in {path}"
    return int(m.group(1))


def test_the_pinned_width_and_the_passed_width_agree():
    """They are checked against each other at runtime; a mismatch aborts the run.

    Changing one and not the other is the easiest way to lose an hour, and the
    check that catches it only runs on a GPU box.
    """
    passed = _width_from(RUN_SH, r"val_per_task_batch_size='\{[^}]*search:(\d+)")
    pinned = _width_from(PINNED, r'^"env\.multitask\.val_per_task_batch_size":.*search: (\d+)')
    assert passed == pinned


def test_the_shipped_width_fits_the_shipped_kv_budget():
    """504 does not fit below 0.85, and a run that starts anyway just preempts.

    Sized on the conservative reading of peak KV per row (468 tokens, linear in
    the 252 measurement) rather than the optimistic one (414, what the 378
    estimate implies), because no run so far can tell the two apart -- both fit
    at 378 -- and the pessimistic one is the side that fails loudly.
    """
    import re

    width = _width_from(RUN_SH, r"val_per_task_batch_size='\{[^}]*search:(\d+)")
    text = EVAL_SH.read_text()
    m = re.search(r'ROLLOUT_GPU_MEM_UTIL:-([0-9.]+)', text)
    assert m, "eval_checkpoints.sh no longer sets a default budget"
    util = float(m.group(1))
    d = re.search(r'VAL_PIPELINE_DEPTH:-(\d+)', text)
    assert d, "eval_checkpoints.sh no longer sets a default depth"
    depth = int(d.group(1))

    budget = (util * 48 - 10.9) * 8920          # KV tokens per GPU
    # DEPTH IS PART OF THE PEAK. 468 tokens per row of width was measured at
    # depth 3, and the pump submits every row as its own request with no
    # in-flight cap, so slots stack in the engine.
    needed = width * 468 * depth / 3
    assert needed <= 0.92 * budget, (
        f"search width {width} needs {needed:,.0f} KV tokens and "
        f"gpu_memory_utilization={util} gives {budget:,.0f} ({100 * needed / budget:.0f}%)"
    )


def test_the_guard_refuses_a_width_the_budget_cannot_hold():
    """And it has to actually fire -- 504 at 0.80 is 96%, which preempts."""
    done = subprocess.run(
        ["bash", str(EVAL_SH)],
        capture_output=True, text=True,
        env={**os.environ, "ROLLOUT_GPU_MEM_UTIL": "0.75", "PATH": os.environ.get("PATH", "")},
    )
    assert done.returncode != 0
    assert "will preempt" in done.stderr, done.stderr[-2000:]


def test_ray_gets_a_temp_directory_that_is_not_shared_with_other_users():
    """/tmp/ray is per machine; two Ray versions in it kill the raylet.

    The failure is not obviously about this: the raylet reports
    "Runtime Env Agent timed out in 30000ms ... on_read bad version" and the
    job dies with ActorDiedError before any actor runs, which reads as load or
    as a stale session of our own. It is neither -- it is a protocol mismatch
    against somebody else's agent on a shared box.
    """
    import re
    text = EVAL_SH.read_text()
    m = re.search(r'export RAY_TMPDIR="\$\{RAY_TMPDIR:-([^"]+)\}"', text)
    assert m, "eval_checkpoints.sh no longer isolates RAY_TMPDIR"
    default = m.group(1)
    assert default != "/tmp/ray", "that is the shared default this exists to avoid"
    assert "id -un" in default or "USER" in default, default


def test_the_run_records_what_else_was_on_the_machine():
    """ms/row is the judgement axis and it moves with the neighbours."""
    done = subprocess.run(
        ["bash", str(EVAL_SH)], capture_output=True, text=True,
        env={**os.environ, "PATH": os.environ.get("PATH", "")},
    )
    out = done.stdout + done.stderr
    assert "[eval] machine" in out
    assert "load" in out


# --------------------------------------------------------------------------- #
# The OPD arm's two knobs
# --------------------------------------------------------------------------- #
OPD_SH = Path(__file__).resolve().parents[2] / "examples" / "opd_trainer" / "run_multitask_offpolicy_qwen3.sh"


def test_the_opd_arm_asks_for_matched_mini_batches():
    """BALANCE_MINIBATCH must be exported, not just assigned.

    ray_trainer reads it at IMPORT time into a module global, so a plain shell
    variable -- or one set after the driver starts -- is silently ignored and
    the run looks identical while doing nothing.
    """
    import re
    text = OPD_SH.read_text()
    m = re.search(r'^export BALANCE_MINIBATCH=\$\{BALANCE_MINIBATCH:-(\d)\}', text, re.M)
    assert m, "BALANCE_MINIBATCH is not exported by the off-policy OPD script"
    assert m.group(1) == "1"
    # Overridable, so a run that needs the old trajectory can still have it.
    assert "${BALANCE_MINIBATCH:-" in text


def test_the_switch_is_read_at_import_and_defaults_off_in_the_library():
    """The script opts in; the library must not.

    Turning it on changes which rows share a mini-batch, so it changes the
    trajectory. That is a per-arm decision recorded in a script, not a default
    that silently moves every other arm's results.
    """
    import re
    lib = (Path(__file__).resolve().parents[2] / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text()
    m = re.search(r'_BALANCE_MINIBATCH = os\.environ\.get\("BALANCE_MINIBATCH", "(\d)"\)', lib)
    assert m and m.group(1) == "0", "the library default must stay off"


def test_the_micro_batch_default_is_ten():
    """The v2 run ran at 5: 682 micro-batches for 6,820 rows over two ranks."""
    import re
    m = re.search(r'ppo_micro_per_gpu=\$\{PPO_MICRO_PER_GPU:-(\d+)\}', OPD_SH.read_text())
    assert m and m.group(1) == "10"


def test_depth_has_exactly_one_default_and_the_guard_reads_it():
    """It had two, and they disagreed.

    The guard defaulted VAL_PIPELINE_DEPTH to 3 while the script exported 4, so
    every default run was validated against two thirds of the KV it would
    actually ask for -- the class of mistake the guard exists to catch, made by
    the guard.
    """
    import re
    text = EVAL_SH.read_text()
    defaults = re.findall(r'VAL_PIPELINE_DEPTH:-(\d+)', text)
    assert len(defaults) == 1, f"depth has {len(defaults)} defaults: {defaults}"
    # And the guard's own variable comes from it rather than repeating it.
    assert '_DEPTH="$VAL_PIPELINE_DEPTH"' in text


def test_the_shipped_pair_is_the_same_envelope_as_the_one_it_replaces():
    """504 x 3 and 378 x 4 hold the same 1512 rows in flight."""
    import re
    width = _width_from(RUN_SH, r"val_per_task_batch_size='\{[^}]*search:(\d+)")
    depth = int(re.search(r'VAL_PIPELINE_DEPTH:-(\d+)', EVAL_SH.read_text()).group(1))
    assert width * depth == 504 * 3, f"{width} x {depth} = {width * depth}, not 1512"


def test_a_log_is_identifiable_from_its_first_lines(tmp_path, logs):
    """Before Ray starts, and before it can be overwritten by the next run.

    /tmp/eval_504.log was overwritten by a later invocation and its wall,
    ms/row and score went with it. Only the trace survived, and a trace cannot
    say how long a run took. The [rollout-engine] and [val-pipeline] lines this
    table used to depend on appear minutes in, so a run that dies in startup
    is unidentifiable without this.
    """
    early = tmp_path / "eval_mystery.log"
    early.write_text(
        "[eval] machine     : 112 cores, load 4.0 3.0 2.0\n"
        "[eval] config      : search=378 depth=4 gpu_mem_util=0.85 pump=1\n"
        "(SFTMultiTaskTaskRunner pid=1) [val-pipeline] after 2: 2 batches over 30.0s\n"
        "(SFTMultiTaskTaskRunner pid=1) [rollout-turn-timing] WALL  ms/row last20=90 all=90\n"
    )
    fields = row(inventory(early, *logs), "eval_mystery.log")
    assert fields["WIDTH"] == "378"
    assert fields["DEPTH"] == "4"


def test_the_config_line_and_the_in_run_line_agree_on_depth(logs):
    """Older logs have only the in-run line; it must still be read."""
    fields = row(inventory(*logs), "eval_pump.log")
    assert fields["DEPTH"] == "3"      # from [val-pipeline], no config line
    assert fields["WIDTH"] == "?"      # which the old logs never carried
