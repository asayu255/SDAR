# WebShop worker host-memory growth

## Symptom

Multitask runs on `tamago` (2x RTX PRO 6000, 256GB RAM) die of host RAM around
step 24. `perf/cpu_memory_used_gb` climbs monotonically -- 201.8 GB at the start
of a run to 245.0 GB by step 23, about **1.8 GB per step** -- and Ray kills the
run when `used/total` crosses `RAY_memory_usage_threshold`:

```
was 246.61GB / 251.52GB (0.980492), which exceeds the memory usage threshold of 0.98
```

Ray's report of the top memory consumers shows the webshop workers growing in
lockstep: 0.78 GB each early in the run, 1.19 GB each at the OOM. With 120 of
them alive, ~17 MB/step/worker accounts for essentially all of the 1.8 GB/step.

The same code ran for 300 steps on `yamabuki`. Nothing about it got worse -- the
old host simply had 512 GB, so 43 GB of drift over a run fit inside the headroom.

## Cause 1: the per-worker JVM sizes its heap off physical RAM

WebShop's search backend is `pyserini.search.lucene.LuceneSearcher`, which starts
an **embedded JVM inside every worker process**. Nothing in WebShop, verl-agent,
or this repo passed `-Xmx`, so each JVM used HotSpot's default max heap of 1/4 of
physical RAM. Two things about that default get worse on the new host, both
because the machine is bigger:

| | yamabuki | tamago |
| --- | --- | --- |
| physical RAM | 512 GB | 256 GB |
| default max heap per JVM | 128 GB | 62 GB |
| cores (drives G1's GC thread count) | 56 | 128 |

A JVM with a 62 GB ceiling has no reason to run a full GC over a working set of a
few MB, so each worker's heap simply grows. Multiply by 120 workers and the run
walks into the OOM killer. G1 (the default collector) also sizes its parallel GC
thread pool from the core count, so tamago's 128 cores add per-JVM thread stacks
and region bookkeeping that yamabuki's 56 did not.

`WebshopWorker.__init__` now sets, before the import that starts the JVM:

```
_JAVA_OPTIONS=-Xmx512m -Xms64m -XX:+UseSerialGC
```

512 MB is roughly 50x the steady-state Lucene working set. `UseSerialGC` drops
the GC thread pool, which matters only because there are 120 JVMs. Override with
`SDAR_WEBSHOP_JVM_OPTIONS` if a larger index ever needs more.

The env var has to be set *before* `web_agent_site.engine` imports pyserini --
the JVM reads it at startup and ignores it afterwards. That ordering is what
`tests/trainer/test_webshop_worker_memory.py` pins.

## Cause 2: `SimServer.user_sessions` is never pruned

`SimServer.receive` inserts a session on first contact and nothing ever removes
it, so a worker running one episode per training step accumulates one dead
session for the life of the run. Each entry is small (the `goal` is a shared
reference, not a copy), so this is worth kilobytes, not gigabytes -- but it is
unbounded, and only the live episode is ever read back. `WebAgentTextEnv.reset`
now clears the map, guarded on the env owning its server: a `SimServer` passed in
from outside may be serving other envs' sessions.

## Confirming it on a host

`scripts/probe_webshop_memory.py` runs one worker through N episodes and reports
RSS split into anonymous and file-backed:

```bash
python scripts/probe_webshop_memory.py --episodes 30 --turns 8
SDAR_WEBSHOP_JVM_OPTIONS="-Xmx62g" python scripts/probe_webshop_memory.py --episodes 30 --turns 8
```

The split is the whole point. `psutil.virtual_memory().used` -- what
`perf/cpu_memory_used_gb` reports and what Ray's OOM killer thresholds on --
excludes page cache, so only the **anonymous** half counts against the run.
Lucene opens its index through `MMapDirectory`, so paging the index in inflates a
worker's RSS without costing the run anything; a real leak shows up as anonymous
growth. Judging by RSS alone confuses the two.

## What this does not change

Nothing here touches rollout, reward, or sampling: the JVM cap is a memory
setting, and the session map is write-only state. Goal selection still comes from
`WebshopMultiProcessEnv._rng`, untouched. No entry in
`examples/opd_trainer/expected_multitask_config.yaml` is affected.
