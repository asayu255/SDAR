"""Unit tests for the FSDP sharding-strategy selection.

Covered (CPU-only; no distributed init, no model build):
* the defaults are exactly what they were before the knob existed, whether the
  config is absent, ``None``, or predates the key;
* ``shard_grad_op`` maps to the ZeRO-2 strategy for both mesh shapes;
* a typo, or a value that belongs to the other mesh shape, raises rather than
  silently falling back to the default -- the whole point of the knob is that a
  run's sharding is what the config says it is.

If the verl stack (torch/...) is not installed, the module is skipped.
"""

import types

import pytest

torch = pytest.importorskip("torch")

try:
    from omegaconf import OmegaConf
    from torch.distributed.fsdp import ShardingStrategy

    from verl.workers.fsdp_workers import get_sharding_strategy
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _mesh(ndim):
    return types.SimpleNamespace(ndim=ndim)


@pytest.mark.parametrize(
    "ndim, expected",
    [(1, ShardingStrategy.FULL_SHARD), (2, ShardingStrategy.HYBRID_SHARD)],
)
@pytest.mark.parametrize(
    "config",
    [
        None,
        OmegaConf.create({"sharding_strategy": None}),  # what the yaml ships
        OmegaConf.create({"param_offload": False}),  # a config predating the key
    ],
    ids=["absent", "null", "key-missing"],
)
def test_default_is_unchanged(ndim, expected, config):
    assert get_sharding_strategy(_mesh(ndim), config) is expected


def test_shard_grad_op_selects_zero2():
    assert (
        get_sharding_strategy(_mesh(1), OmegaConf.create({"sharding_strategy": "shard_grad_op"}))
        is ShardingStrategy.SHARD_GRAD_OP
    )
    # A 2-D mesh's ZeRO-2 is the hybrid variant, not the flat one.
    assert (
        get_sharding_strategy(_mesh(2), OmegaConf.create({"sharding_strategy": "shard_grad_op"}))
        is ShardingStrategy._HYBRID_SHARD_ZERO2
    )


def test_default_can_be_named_explicitly():
    assert (
        get_sharding_strategy(_mesh(1), OmegaConf.create({"sharding_strategy": "FULL_SHARD"}))
        is ShardingStrategy.FULL_SHARD
    )
    assert (
        get_sharding_strategy(_mesh(2), OmegaConf.create({"sharding_strategy": "hybrid_shard"}))
        is ShardingStrategy.HYBRID_SHARD
    )


def test_plain_dict_config_works():
    """Not every call site hands over an OmegaConf node."""
    assert (
        get_sharding_strategy(_mesh(1), {"sharding_strategy": "shard_grad_op"})
        is ShardingStrategy.SHARD_GRAD_OP
    )


@pytest.mark.parametrize(
    "value, ndim",
    [
        ("zero2", 1),  # plausible typo
        ("hybrid_shard", 1),  # belongs to a 2-D mesh
        ("full_shard", 2),  # belongs to a 1-D mesh
    ],
)
def test_unsupported_value_raises(value, ndim):
    with pytest.raises(ValueError, match="sharding_strategy"):
        get_sharding_strategy(_mesh(ndim), OmegaConf.create({"sharding_strategy": value}))


def test_offload_combination_warns():
    """ZeRO-2 keeps resident exactly what offloading sends away."""
    config = OmegaConf.create({"sharding_strategy": "shard_grad_op", "param_offload": True})
    with pytest.warns(UserWarning, match="param_offload"):
        assert get_sharding_strategy(_mesh(1), config) is ShardingStrategy.SHARD_GRAD_OP


def test_mesh_ndim_is_still_validated():
    with pytest.raises(NotImplementedError):
        get_sharding_strategy(_mesh(3))
