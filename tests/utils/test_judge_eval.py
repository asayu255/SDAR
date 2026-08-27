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
"""judge_eval.sh must read a log the size of a real one.

THE LOGS HERE ARE LONG ON PURPOSE. The bug these tests exist for is invisible
in a short log: `sed log | grep -q PATTERN` under `set -o pipefail` returns 141
when grep matches EARLY, because grep leaves and sed dies of SIGPIPE -- so a
pattern that MATCHED reads as absent. A fixture of twenty lines fits in the
64 KB pipe buffer, sed finishes before grep exits, and every assertion passes
while the shipped script tells you the pump was off on a run that printed
"driving 3 ranks as a pool" in its first second.

That is the same failure as the one in test_pump_client.py: the check was
correct about something other than what runs in production. So these build a
log big enough for the pipe to matter, and assert on the script's output.
"""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

JUDGE = Path(__file__).resolve().parents[2] / "examples" / "sft_trainer" / "judge_eval.sh"

# Comfortably past the 64 KB pipe buffer, which is what makes the reader's early
# exit reach the writer at all.
FILLER_PER_BATCH = 60

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def write_log(path, *, batches, ms_row, wall_per_batch, report_every, salt, pump=True, final=False):
    prefix = "(SFTMultiTaskTaskRunner pid=689806) "
    lines = []
    if pump:
        lines.append(prefix + "[rollout-pump] driving 3 ranks as a pool; 3 ranks, block=1")
    lines.append(prefix + "[val-pipeline] VAL_PIPELINE_DEPTH=3: 3 slot(s)")
    for batch in range(1, batches + 1):
        lines += [prefix + f"chatter {batch} {i}" for i in range(FILLER_PER_BATCH)]
        digest = hashlib.sha1(f"{salt}-{batch}".encode()).hexdigest()
        lines.append(prefix + f"[val-hash] batch#{batch} rows=252 sha1 {digest}")
        lines.append(prefix + f"[rollout-turn-timing] WALL 18.5s  ms/row last20={ms_row} all={ms_row}")
        if batch % report_every == 0:
            lines.append(
                prefix + f"[val-pipeline] after {batch}: {batch} batches over "
                f"{batch * wall_per_batch:.1f}s: at least one slot running"
            )
            lines.append(prefix + f"[gpu-residency] {batch * 15}s sampled: 3gpu 87.0%, 0gpu 7.4%")
            lines.append(prefix + "[gpu-residency] -> EMPTY is the DRIVER RUNNING PYTHON: 70% of one core")
    if final:
        lines.append(prefix + f"[val-pipeline] final: {batches} batches over {batches * wall_per_batch:.1f}s")
        lines.append(prefix + "Initial validation metrics: {'val/search/test_score': 0.3577398068712026}")
    path.write_text("\n".join(lines) + "\n")
    return path


def judge(*logs):
    done = subprocess.run(["bash", str(JUDGE), *[str(log) for log in logs]], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.fixture
def pair(tmp_path):
    control = write_log(tmp_path / "control.log", batches=209, ms_row=73, wall_per_batch=18.5, report_every=25, salt="a", final=True)
    candidate = write_log(tmp_path / "candidate.log", batches=152, ms_row=65, wall_per_batch=16.8, report_every=5, salt="b")
    return control, candidate


def test_a_pump_that_engaged_in_a_long_log_is_reported_as_on(pair):
    """The regression. Both logs announce the pool on their first line."""
    out = judge(*pair)
    assert "pump off" not in out
    assert out.count("PUMP ON") == 2


def test_the_slot_count_is_read_without_a_stray_not_found(pair):
    out = judge(*pair)
    assert "VAL_PIPELINE_DEPTH=3: 3 slot(s)" in out
    # "NOT FOUND" used to print directly under the line it had just found.
    assert "NOT FOUND" not in out.split("== 2.")[0]


def test_a_pump_that_refused_is_not_reported_as_on(tmp_path):
    """The detector must still be able to say no."""
    log = write_log(tmp_path / "refused.log", batches=40, ms_row=73, wall_per_batch=18.5, report_every=25, salt="c", pump=False)
    lines = log.read_text().splitlines()
    lines.insert(0, "(SFTMultiTaskTaskRunner pid=1) [rollout-pump] staying on the blocking path: eos_token_id is a list")
    log.write_text("\n".join(lines) + "\n")
    out = judge(log)
    assert "PUMP REFUSED" in out
    assert "PUMP ON" not in out


def test_the_verdict_does_not_call_a_printed_residency_report_missing(pair):
    out = judge(*pair)
    assert "TOO EARLY" not in out
    assert "NEEDS A FINISHED RUN" in out  # the digests differ, so the scores decide


def test_speed_is_compared_over_the_batches_both_runs_reached(pair):
    """152 batches against 209 is two experiments, not one measurement."""
    out = judge(*pair)
    assert "same-prefix (first 152 batches of each)" in out
    assert "control all=73  candidate all=65" in out


def test_wall_is_compared_at_a_batch_count_both_runs_reported(pair):
    out = judge(*pair)
    # control reports every 25, candidate every 5: the last count they share.
    assert "same-prefix (150 batches each)" in out


def test_one_log_is_a_reading_not_a_comparison(pair):
    out = judge(pair[0])
    assert "this is a reading, not a comparison" in out
    # The advice mentioning the same-prefix line stays; the line itself needs
    # two runs and must not be invented from one.
    assert "same-prefix (" not in out
