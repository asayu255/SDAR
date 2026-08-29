"""Ask the installed vLLM which host/GPU overlap knobs it actually has.

Kept out of the vllm_rollout package on purpose: that package refuses to import
without vLLM installed, and a diagnostic that cannot be tested without a GPU
image is a diagnostic that rots. Nothing here imports vLLM at module scope.
"""

import os

# Engine arguments that decide whether the engine's own per-step host work is
# exposed between kernel launches or hidden behind them. Which of these exists
# depends on the vLLM version -- V0's multi-step scheduler was removed in V1 and
# replaced by async scheduling -- so this asks the installed engine rather than
# assuming, and prints what it found.
_REQUIRE_CORE = os.environ.get("ROLLOUT_REQUIRE_CORE", "").strip().lower()


_OVERLAP_ARGS = (
    "async_scheduling",           # V1: overlap the scheduler with the forward
    "num_scheduler_steps",        # V0 multi-step; absent on V1
    "disable_async_output_proc",  # V0: overlap output processing
    "enable_chunked_prefill",
    "max_num_seqs",
    "enforce_eager",
)


def report_engine_overlap(engine_kwargs, explicit=None, engine=None) -> None:
    """Say which host/GPU overlap knobs this vLLM has, and which are in force.

    All-three-cards-busy samples read 87-90%, not 100, and the missing tenth is
    the engine's per-step host work -- schedule, build inputs, process outputs --
    sitting between kernel launches. That is not physics and there is machinery
    for hiding it, but the machinery is version-specific: guessing a flag name
    costs a failed launch, and assuming a default costs a wrong conclusion about
    what the ceiling is.

    ``engine_kwargs.vllm`` passes any engine argument straight through, so the
    work is knowing which ones exist. ``explicit`` carries the ones the rollout
    already names in its own ``LLM(...)`` call -- without it those would print as
    "<default>", which is a lie about a value we chose.

    ``engine`` is used only to name which engine core is running. That decides
    whether the knobs above mean anything at all: num_scheduler_steps and
    disable_async_output_proc are V0 features, so on a V1 core they are present
    in the signature and ignored in effect -- which is the worst of both, a flag
    that accepts a value and changes nothing. Reading the class is the only
    honest way to tell.

    Print it once, at build, next to the settings already chosen. Best-effort:
    an engine whose internals moved gets a shorter line, never an exception --
    this is a diagnostic, not a dependency.
    """
    try:
        import inspect

        import vllm
        from vllm import LLM

        accepted = set(inspect.signature(LLM.__init__).parameters)
        # engine_kwargs wins: it is applied last in the LLM(...) call, so a key
        # in both is the one that takes effect.
        chosen = dict(explicit or {})
        chosen.update(engine_kwargs or {})
        found = []
        for name in _OVERLAP_ARGS:
            if name in chosen:
                found.append(f"{name}={chosen[name]} (set here)")
            elif name in accepted:
                found.append(f"{name}=<default> (available)")
            else:
                found.append(f"{name}=absent")
        core = _engine_core(engine)
        print(
            f"[rollout-engine] vllm {getattr(vllm, '__version__', '?')}, "
            f"core={core}; overlap knobs: " + ", ".join(found),
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - diagnostics never fail a run
        print(f"[rollout-engine] could not report overlap knobs: {exc}", flush=True)
        core = _engine_core(engine)
    require_core(core)


def require_core(core: str) -> None:
    """Fail the run when the engine core is not the one that was asked for.

    VLLM_USE_V1=0 is a request. vLLM decides for itself and falls back to V1 for
    configurations V0 cannot serve -- and this rollout hands it two that V0 may
    not: distributed_executor_backend="external_launcher" and
    enable_sleep_mode=True. A fallback does not raise; it produces a working run
    that measures the core you were trying to compare against.

    That is the exact shape of the two failures that have already cost whole
    runs here: a pool that refused and fell back to the blocking path, and a
    retriever that rejected batched queries and fell back to one at a time.
    Both ran to completion, both looked ordinary, and both measured the control
    twice. So this is opt-in and loud rather than a warning:

        ROLLOUT_REQUIRE_CORE=v0 ... bash examples/sft_trainer/eval_checkpoints.sh

    num_scheduler_steps is a V0 feature. Setting it on a V1 core is accepted and
    ignored, which is worse than an error, and this is what catches that.
    """
    if not _REQUIRE_CORE:
        return
    if core == _REQUIRE_CORE:
        print(f"[rollout-engine] ROLLOUT_REQUIRE_CORE={_REQUIRE_CORE}: confirmed.", flush=True)
        return
    raise RuntimeError(
        f"ROLLOUT_REQUIRE_CORE={_REQUIRE_CORE} but vLLM built core={core}. "
        "vLLM chose the other engine rather than failing -- most likely this "
        "configuration is one the requested core cannot serve (external_launcher "
        "and enable_sleep_mode are the two this rollout passes). Refusing to "
        "start, because the run would otherwise measure the core you were "
        "comparing against. Unset ROLLOUT_REQUIRE_CORE to run anyway."
    )


def _engine_core(engine) -> str:
    """"v1", "v0", or "unknown" -- read from the object, not from an env var.

    VLLM_USE_V1 is a request, not an outcome: vLLM falls back to V0 for
    configurations V1 does not support and the variable keeps its value, so a run
    can ask for V1 and get V0 with nothing saying so. The class the engine
    actually built cannot lie about it.
    """
    inner = getattr(engine, "llm_engine", engine)
    module = type(inner).__module__ if inner is not None else ""
    if ".v1." in f".{module}.":
        return "v1"
    if module.startswith("vllm."):
        return "v0"
    return "unknown"
