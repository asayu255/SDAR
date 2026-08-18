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
"""The trace summary, checked against traces whose true answer is known.

Three claims carry the conclusion, and each is a way the previous reads of this
data went wrong:

*Busy time is a union, not a sum.* Streams overlap constantly, and summing
kernel durations reports a device more than 100% busy -- which would turn every
real gap into a negative number and hide it. ``test_overlapping_streams_*``
pins that down.

*A kernel belongs to the micro-batch that launched it.* Kernels lag their
launch by however long the queue is, so attributing by the kernel's own start
drifts the boundary work into the next window -- exactly the boundary this is
trying to measure. ``test_a_kernel_is_attributed_*`` covers it.

*Long and in NCCL is not the same as long and computing.* This is the question
NVML cannot answer, because a spinning collective counts as busy to it, and it
is the entire reason for the file. The rank everyone waits for is the one with
the LEAST NCCL time, and it is last for one of two reasons -- it was handed more
work, or it stalled -- which want completely different fixes. Both directions
are covered below.
"""

import importlib.util
import json
import pathlib
import sys

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "actor_trace_summary.py"
_spec = importlib.util.spec_from_file_location("actor_trace_summary", _SCRIPT)
summary = importlib.util.module_from_spec(_spec)
sys.modules["actor_trace_summary"] = summary
_spec.loader.exec_module(summary)


def _kernel(name, ts, dur, correlation=None):
    return {"ph": "X", "cat": "kernel", "name": name, "pid": 0, "tid": 7,
            "ts": ts, "dur": dur, "args": {"correlation": correlation}}


def _launch(ts, correlation):
    return {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
            "pid": 1, "tid": 1, "ts": ts, "dur": 5, "args": {"correlation": correlation}}


def _micro(counter, ts, dur):
    return {"ph": "X", "cat": "user_annotation", "name": f"micro/{counter}/0",
            "pid": 1, "tid": 1, "ts": ts, "dur": dur}


def _op(name, ts, dur):
    return {"ph": "X", "cat": "cpu_op", "name": name, "pid": 1, "tid": 1,
            "ts": ts, "dur": dur}


def _write(tmp_path, rank, events, base_ns=0):
    path = tmp_path / f"actor_rank{rank}_pid999.json"
    path.write_text(json.dumps({"schemaVersion": 1, "traceEvents": events,
                                "baseTimeNanoseconds": base_ns}))
    return str(path)


def _embedding(ts, dur, tokens):
    """aten::embedding as torch records it under record_shapes: the weight's
    dims, then the indices', whose last entry is the token count."""
    return {"ph": "X", "cat": "cpu_op", "name": "aten::embedding", "pid": 1, "tid": 1,
            "ts": ts, "dur": dur, "args": {"Input Dims": [[151936, 2048], [1, tokens]]}}


def test_metadata_and_flow_events_are_not_intervals(tmp_path):
    """A Chrome trace is mostly not spans: process names, counters and the
    launch->kernel arrows all lack a duration. Treating them as zero-width
    intervals would put thousands of them in the merge and cost nothing but
    time; treating a missing dur as 0.0 would be worse, since the arrows land
    exactly on kernel boundaries and would look like work."""
    events = [
        {"ph": "M", "name": "process_name", "pid": 0, "args": {"name": "python"}},
        {"ph": "f", "cat": "ac2g", "name": "ac2g", "pid": 0, "tid": 7, "ts": 100, "id": 1},
        _micro(40, 0, 1000),
        _kernel("gemm", 0, 1000),
    ]
    rows = summary.analyse(_write(tmp_path, 0, events))

    assert len(rows) == 1
    assert rows[0]["kernels"] == 1
    assert rows[0]["idle_ms"] == 0.0


def test_overlapping_streams_never_report_more_than_the_wall(tmp_path):
    """Three streams each busy for the whole micro-batch is a 100% busy device,
    not a 300% one."""
    events = [_micro(40, 0, 1000)]
    events += [_kernel(f"s{i}", 0, 1000) for i in range(3)]
    rows = summary.analyse(_write(tmp_path, 0, events))

    assert rows[0]["busy_ms"] == 1.0            # 1000 us, not 3000
    assert rows[0]["idle_ms"] == 0.0
    assert rows[0]["gap_ms"] == 0.0


def test_a_real_gap_is_found_and_described(tmp_path):
    """Two kernels with 400 us of nothing between them, and the host op that was
    running across it. The op is the difference between "the host stopped
    submitting" and "the device was waiting on an event"."""
    events = [
        _micro(40, 0, 1000),
        _kernel("gemm_a", 0, 300),
        _kernel("gemm_b", 700, 300),
        _op("aten::item", 300, 400),            # a device->host sync, the classic
    ]
    rows = summary.analyse(_write(tmp_path, 0, events))

    assert rows[0]["gap_ms"] == 0.4
    assert rows[0]["idle_ms"] == 0.4
    assert rows[0]["gap_host"] == "aten::item"
    assert rows[0]["gap_bracket"] == ("gemm_a", "gemm_b")


def test_the_innermost_host_op_is_the_informative_one(tmp_path):
    """Host ops nest many deep. The outermost is always "the model", which says
    nothing; the narrowest one covering the gap is the call that blocked."""
    events = [
        _micro(40, 0, 1000),
        _kernel("gemm_a", 0, 300),
        _kernel("gemm_b", 900, 100),
        _op("nn.Module: Qwen3ForCausalLM", 0, 1000),
        _op("aten::linear", 200, 600),
        _op("cudaStreamSynchronize", 250, 450),
    ]
    rows = summary.analyse(_write(tmp_path, 0, events))

    assert rows[0]["gap_host"] == "cudaStreamSynchronize"


def test_a_kernel_is_attributed_to_the_micro_batch_that_launched_it(tmp_path):
    """The queue runs behind the host. A kernel launched at the end of one
    micro-batch and executed at the start of the next belongs to the first --
    otherwise every window inherits its predecessor's tail and the boundary,
    which is the thing under investigation, is measured in the wrong place."""
    events = [
        _micro(40, 0, 500),
        _micro(41, 500, 500),
        _launch(490, correlation=1),            # launched in 40...
        _kernel("gemm_late", 520, 100, correlation=1),   # ...ran in 41
        _launch(510, correlation=2),
        _kernel("gemm_next", 700, 100, correlation=2),
    ]
    rows = {r["micro"]: r for r in summary.analyse(_write(tmp_path, 0, events))}

    assert rows[40]["kernels"] == 1
    assert rows[41]["kernels"] == 1
    # and its time is clipped to the window it was launched in, so no window
    # can report more busy time than it has wall time
    assert rows[40]["busy_ms"] <= rows[40]["wall_ms"]


def test_a_kernel_with_no_correlation_still_lands_somewhere(tmp_path):
    """Not every device event carries one -- memsets and some copies do not --
    and dropping them would understate busy time and invent gaps."""
    events = [_micro(40, 0, 1000), _kernel("memset", 100, 800, correlation=None)]
    rows = summary.analyse(_write(tmp_path, 0, events))

    assert rows[0]["kernels"] == 1
    assert rows[0]["busy_ms"] == 0.8


def test_the_rank_handed_more_work_is_the_one_everyone_waits_for(tmp_path, capsys):
    """The signature NVML shows as "one card at 0, two at 99" is ambiguous to
    it: a spinning collective is busy. Here rank 1 computes for 800 us while
    ranks 0 and 2 sit in NCCL for most of theirs, which is only readable
    because the collectives are named."""
    def rank_events(compute, nccl):
        return [_micro(40, 0, 1000),
                _kernel("gemm", 0, compute),
                _kernel("ncclDevKernel_AllGather", compute, nccl)]

    by_rank = {
        "0": summary.analyse(_write(tmp_path, 0, rank_events(200, 780))),
        "1": summary.analyse(_write(tmp_path, 1, rank_events(800, 180))),
        "2": summary.analyse(_write(tmp_path, 2, rank_events(250, 730))),
    }
    summary.report(by_rank, top=3)
    out = capsys.readouterr().out

    assert "rank 1 got more work" in out


def test_a_stalled_rank_is_not_reported_as_having_more_work(tmp_path, capsys):
    """The distinction the whole investigation turns on. Rank 1 is last to the
    collective by 380 ms, and the others wait for it either way -- but its
    compute matches theirs exactly and the 380 ms is a hole in its own stream.
    Reading that as a token imbalance would send the next day into load
    balancing, which cannot touch it."""
    fast = [_micro(40, 0, 1000), _kernel("gemm", 0, 200),
            _kernel("ncclDevKernel_AllGather", 200, 790)]
    stalled = [_micro(40, 0, 1000), _kernel("gemm", 0, 200),
               _op("aten::_local_scalar_dense", 200, 380),   # a device->host sync
               _kernel("ncclDevKernel_AllGather", 580, 410)]

    by_rank = {
        "0": summary.analyse(_write(tmp_path, 0, fast)),
        "1": summary.analyse(_write(tmp_path, 1, stalled)),
        "2": summary.analyse(_write(tmp_path, 2, fast)),
    }
    summary.report(by_rank, top=3)
    out = capsys.readouterr().out

    assert "a stall on rank 1" in out
    assert "got more work" not in out


def test_matched_ranks_are_not_reported_as_a_straggler(tmp_path, capsys):
    """Ranks are never equal to the microsecond, so somebody is always last and
    somebody always waits. Naming it would send the next day into load
    balancing over a 2% difference that no fix can reach."""
    def rank_events(compute):
        return [_micro(40, 0, 1000),
                _kernel("gemm", 0, compute),
                _kernel("ncclDevKernel_AllGather", compute, 20)]

    by_rank = {
        "0": summary.analyse(_write(tmp_path, 0, rank_events(500))),
        "1": summary.analyse(_write(tmp_path, 1, rank_events(520))),
        "2": summary.analyse(_write(tmp_path, 2, rank_events(510))),
    }
    summary.report(by_rank, top=3)

    assert "balanced" in capsys.readouterr().out


def test_a_trace_with_no_window_says_so_rather_than_printing_an_empty_table(tmp_path, capsys):
    """The cheapest failure to make: run without ACTOR_TORCH_MICRO, get a trace
    with no annotations, and read a clean empty report as "no stalls"."""
    summary.report({"0": summary.analyse(_write(tmp_path, 0, [_kernel("gemm", 0, 10)]))}, top=3)

    assert "was ACTOR_TORCH_MICRO set?" in capsys.readouterr().out


def test_a_directory_is_expanded_to_its_traces(tmp_path):
    _write(tmp_path, 0, [_micro(40, 0, 10)])
    _write(tmp_path, 1, [_micro(40, 0, 10)])

    assert len(summary._expand([str(tmp_path)])) == 2
    assert len(summary._expand([str(tmp_path / "actor_rank0*.json")])) == 1


def test_end_to_end_over_three_ranks(tmp_path, capsys):
    for rank in range(3):
        _write(tmp_path, rank, [
            _micro(40, 0, 1000), _kernel("gemm", 0, 900 - rank * 100),
            _micro(41, 1000, 1000), _kernel("gemm", 1000, 900),
        ])

    assert summary.main([str(tmp_path), "--top", "2"]) == 0
    out = capsys.readouterr().out
    assert "per micro-batch, per rank" in out
    assert "largest idle gap" in out
    assert out.count("micro-batches,") == 3      # one totals line per rank


def test_a_window_with_no_device_work_is_not_reported_as_a_stall(tmp_path, capsys):
    """A capture that recorded no kernels makes every window 100% "idle" by
    construction. That is the profiler seeing nothing, not the GPU being empty,
    and it would otherwise sit at the top of the gap table looking like the
    largest stall in the run."""
    events = [_micro(40, 0, 1000),                       # no kernels at all
              _micro(41, 1000, 1000), _kernel("gemm", 1000, 900)]
    summary.report({"0": summary.analyse(_write(tmp_path, 0, events))}, top=5)
    out = capsys.readouterr().out

    assert "recorded no device work at all" in out
    assert "micro   40" not in out                       # kept out of the gap table
    assert "micro   41" in out


def test_the_ranks_are_checked_to_be_on_the_same_micro_batch(tmp_path, capsys):
    """Every comparison below assumes rank N's micro k ran while rank M's did.
    The traces carry an epoch, so it can be checked instead of assumed -- and if
    they do not overlap, 'who waited for whom' is not a question the data
    answers."""
    events = [_micro(40, 0, 1000), _kernel("gemm", 0, 900)]
    apart = {
        "0": summary.analyse(_write(tmp_path, 0, events, base_ns=1_000_000_000_000_000)),
        "1": summary.analyse(_write(tmp_path, 1, events, base_ns=1_000_000_005_000_000)),
    }
    summary.report(apart, top=2)
    out = capsys.readouterr().out

    assert "barely overlap" in out                       # 5 ms apart, 1 ms windows

    together = {
        "0": summary.analyse(_write(tmp_path, 0, events, base_ns=1_000_000_000_000_000)),
        "1": summary.analyse(_write(tmp_path, 1, events, base_ns=1_000_000_000_100_000)),
    }
    summary.report(together, top=2)
    out = capsys.readouterr().out

    assert "barely overlap" not in out                   # 0.1 ms apart of 1 ms
    assert "r1:    +0.1" in out


def test_the_recorded_shapes_give_the_per_micro_token_count(tmp_path, capsys):
    """_balance_batch equalises tokens per rank over the whole batch, and the
    mini-batches are a contiguous chunk of that -- so the per-chunk counts can
    still differ across ranks while the per-rank totals match. The ranks meet at
    the gradient reduce ending each mini-batch, so that difference is a real wait
    no whole-batch balance removes. Reading it off the trace rather than a
    derived number is the point."""
    def rank_events(tokens):
        return [_micro(40, 0, 1000), _kernel("gemm", 0, 900),
                _embedding(10, 5, tokens)]

    by_rank = {"0": summary.analyse(_write(tmp_path, 0, rank_events(9800))),
               "1": summary.analyse(_write(tmp_path, 1, rank_events(10400)))}
    summary.report(by_rank, top=2)
    out = capsys.readouterr().out

    assert "9800" in out and "10400" in out


def test_the_op_the_shapes_come_from_is_selectable(tmp_path):
    """aten::embedding is the default because it is where input_ids enter an HF
    model, but the arm that packs sequences or the model that names it something
    else should not need the script edited."""
    events = [_micro(40, 0, 1000), _kernel("gemm", 0, 900), _embedding(10, 5, 9800)]
    path = _write(tmp_path, 0, events)

    assert summary.analyse(path)[0]["shape"] == [[151936, 2048], [1, 9800]]
    assert summary.analyse(path, shapes_of=("aten::linear",))[0]["shape"] is None


def test_the_optimizer_tail_is_carry_not_idle(tmp_path):
    """The artifact that produced this tool's one wrong conclusion.

    The optimizer step is launched between two micro-batch annotations and its
    kernels execute into the next window's head while the host sits at the first
    sync point (unpad's nonzero). The device is fully busy; only the launch-based
    attribution called it idle. It must land in carry, the true idle must exclude
    it, and the biggest-gap table must not report the head.
    """
    events = [
        _micro(40, 0, 1000),
        _launch(100, correlation=1),
        _kernel("gemm", 150, 800, correlation=1),
        # optimizer, launched AFTER window 40 closes and BEFORE 41 opens...
        _launch(1005, correlation=2),
        _kernel("adam_fused", 1050, 80, correlation=2),
        # ...still running when window 41 opens at 1100 (carry until 1130)
        _micro(41, 1100, 1000),
        _op("aten::nonzero", 1105, 30),           # host blocked behind the tail
        _launch(1135, correlation=3),
        _kernel("gemm", 1140, 900, correlation=3),
        # and a real 60 us hole later in the window: 2040 -> 2100
        _launch(2035, correlation=4),
        _kernel("gemm_late", 2060, 40, correlation=4),
    ]
    rows = {r["micro"]: r for r in summary.analyse(_write(tmp_path, 0, events))}

    r41 = rows[41]
    assert r41["kernels"] == 2                       # ownership unchanged
    assert r41["carry_ms"] == 0.03                   # the tail: 1100 -> 1130
    # idle excludes the tail: window is 1000, resident 30+900+40 = 970
    assert abs(r41["idle_ms"] - 0.03) < 1e-6
    # and the biggest gap is the real mid-window hole, not the head
    assert r41["gap_at"] > 1130
    assert r41["gap_bracket"][0] == "gemm"
