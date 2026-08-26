"""Ask the installed vLLM which host/GPU overlap knobs it actually has.

Kept out of the vllm_rollout package on purpose: that package refuses to import
without vLLM installed, and a diagnostic that cannot be tested without a GPU
image is a diagnostic that rots. Nothing here imports vLLM at module scope.
"""

# Engine arguments that decide whether the engine's own per-step host work is
# exposed between kernel launches or hidden behind them. Which of these exists
# depends on the vLLM version -- V0's multi-step scheduler was removed in V1 and
# replaced by async scheduling -- so this asks the installed engine rather than
# assuming, and prints what it found.
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
        print(
            f"[rollout-engine] vllm {getattr(vllm, '__version__', '?')}, "
            f"core={_engine_core(engine)}; overlap knobs: " + ", ".join(found),
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - diagnostics never fail a run
        print(f"[rollout-engine] could not report overlap knobs: {exc}", flush=True)


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
