"""Reach past whatever vLLM has wrapped around the model runner.

Kept out of the sharding manager because that module cannot be imported without
vLLM installed, and a helper that needs a GPU image to test is a helper that
rots. Nothing here imports vLLM.
"""

# vLLM wraps the model runner when features are enabled around it, and the
# wrapper does not forward `.model`. num_scheduler_steps>1 on the V0 core puts a
# MultiStepModelRunner in front, holding the real runner in `_base_model_runner`,
# and FSDP -> vLLM weight sync then dies with
#
#     AttributeError: 'MultiStepModelRunner' object has no attribute 'model'
#
# which names neither multi-step nor weight sync. The engine builds, the config
# is accepted, and the run gets as far as loading a checkpoint before failing.
_WRAPPER_ATTRS = ("_base_model_runner", "base_model_runner", "model_runner")


def unwrap_model_runner(runner, _max_depth: int = 4):
    """The runner that actually owns `.model`, past any wrappers in front of it.

    Bounded, because a wrapper that points at itself would otherwise hang here
    rather than fail. Returns the outermost object unchanged when nothing
    unwraps -- the caller's AttributeError is then the honest one.
    """
    seen = set()
    for _ in range(_max_depth):
        if runner is None or hasattr(runner, "model"):
            return runner
        if id(runner) in seen:
            return runner
        seen.add(id(runner))
        for attr in _WRAPPER_ATTRS:
            inner = getattr(runner, attr, None)
            if inner is not None and inner is not runner:
                runner = inner
                break
        else:
            return runner
    return runner
