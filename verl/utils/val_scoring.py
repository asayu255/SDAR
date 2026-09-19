"""A val-only run's score must be reproducible, and must say how it was produced.

WHY THIS IS CODE AND NOT A NOTE. The rule was written down (docs/validation_
determinism.md, the control launcher's own comment, the project's notes): with
ROLLOUT_KEEP_VLLM_AWAKE=1, ROLLOUT_ASYNC_GENERATE=1 makes validation
nondeterministic -- one checkpoint re-scored gave 0.744 to 0.811, SD 1.60 pp over
ten repeats -- while ROLLOUT_ASYNC_GENERATE=0 reproduces byte for byte at the same
speed. The launchers still default async ON for training, and on 2026-09-19 three
scoring jobs of the progress_rank arm ran with it on anyway, their numbers then
compared against the five-method sweep, which had it off. A rule that lives only
where someone has to remember to read it did not hold, so a val-only run now
refuses the pair outright (VAL_ALLOW_NONDETERMINISTIC=1 is the explicit way out)
and writes its inference configuration next to its instance dump, so two scores
produced under different configurations cannot be compared unknowingly.
"""

import json
import os
import subprocess
from typing import Mapping, Optional

__all__ = ["check_deterministic_scoring", "scoring_config", "write_scoring_config"]

_TRUE = ("1", "true", "yes", "on")


def _on(env: Mapping[str, str], key: str, default: str = "0") -> bool:
    return str(env.get(key, default)).strip().lower() in _TRUE


def check_deterministic_scoring(env: Optional[Mapping[str, str]] = None) -> str:
    """Raise unless a val-only run is deterministic; return what was checked.

    The code defaults (rollout_loop.py) are async OFF and keep-awake OFF, so an
    unset variable counts as off -- exactly what the running process will do.
    """
    env = os.environ if env is None else env
    asy, awake = _on(env, "ROLLOUT_ASYNC_GENERATE"), _on(env, "ROLLOUT_KEEP_VLLM_AWAKE")
    if asy and awake:
        if _on(env, "VAL_ALLOW_NONDETERMINISTIC"):
            return ("NONDETERMINISTIC scoring allowed by VAL_ALLOW_NONDETERMINISTIC=1 "
                    "(ROLLOUT_ASYNC_GENERATE=1 with ROLLOUT_KEEP_VLLM_AWAKE=1): this score "
                    "carries ~1.6 pp of run-to-run noise and must not be compared as exact")
        raise AssertionError(
            "val-only run with ROLLOUT_ASYNC_GENERATE=1 and ROLLOUT_KEEP_VLLM_AWAKE=1: that pair "
            "makes validation nondeterministic (one checkpoint re-scored 0.744-0.811). Score with "
            "ROLLOUT_ASYNC_GENERATE=0 (byte-reproducible, same speed), or set "
            "VAL_ALLOW_NONDETERMINISTIC=1 to accept the noise knowingly. See "
            "docs/validation_determinism.md.")
    return (f"deterministic scoring: ROLLOUT_ASYNC_GENERATE={int(asy)}, "
            f"ROLLOUT_KEEP_VLLM_AWAKE={int(awake)}")


def _commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def scoring_config(config, env: Optional[Mapping[str, str]] = None) -> dict:
    """Everything known to change a validation score, from the composed config and the env."""
    from omegaconf import OmegaConf

    env = os.environ if env is None else env
    sel = lambda key, default=None: OmegaConf.select(config, key, default=default)  # noqa: E731
    spec = sel("actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config")
    val_kwargs = sel("actor_rollout_ref.rollout.val_kwargs_by_task")
    return {
        "ROLLOUT_ASYNC_GENERATE": env.get("ROLLOUT_ASYNC_GENERATE", "0"),
        "ROLLOUT_KEEP_VLLM_AWAKE": env.get("ROLLOUT_KEEP_VLLM_AWAKE", "0"),
        "VAL_PIPELINE_DEPTH": env.get("VAL_PIPELINE_DEPTH", "1"),
        "VAL_ALLOW_NONDETERMINISTIC": env.get("VAL_ALLOW_NONDETERMINISTIC", "0"),
        "speculative_config": OmegaConf.to_container(spec) if spec is not None else None,
        "enable_prefix_caching": sel("actor_rollout_ref.rollout.enable_prefix_caching"),
        "enforce_eager": sel("actor_rollout_ref.rollout.enforce_eager"),
        "val_kwargs_by_task": OmegaConf.to_container(val_kwargs) if val_kwargs is not None else None,
        "max_response_length": sel("data.max_response_length"),
        "n_gpus_per_node": sel("trainer.n_gpus_per_node"),
        "val_files": sel("data.val_files"),
        "resume_from_path": sel("trainer.resume_from_path"),
        "model_path": sel("actor_rollout_ref.model.path"),
        "commit": _commit(),
    }


def write_scoring_config(config, out_dir: Optional[str], env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Write scoring_config() as ``<out_dir>/scoring_config.json``; return the path."""
    if not out_dir:
        return None
    out_dir = os.path.expanduser(str(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "scoring_config.json")
    with open(path, "w") as f:
        json.dump(scoring_config(config, env), f, indent=2, sort_keys=True, default=str)
    return path
