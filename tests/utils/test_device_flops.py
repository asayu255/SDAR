"""The MFU denominator must be a number for the GPUs this project runs on.

``get_device_flops`` returns ``float('inf')`` for a GPU it does not recognise, and
``estimate_flops`` divides by it -- so an unrecognised card does not make MFU
missing, it makes it exactly 0.0 at every step, which reads as a broken metric
rather than an unsupported one.
"""

import pytest

torch = pytest.importorskip("torch")

from verl.utils.flops_counter import get_device_flops


def _flops_for(monkeypatch, name):
    """get_device_flops as it would answer on a machine reporting `name`."""
    import verl.utils.flops_counter as fc

    class _FakeDevice:
        @staticmethod
        def get_device_name():
            return name

    monkeypatch.setattr(fc, "get_torch_device", lambda: _FakeDevice)
    return fc.get_device_flops()


@pytest.mark.parametrize(
    "name, expected",
    [
        ("NVIDIA RTX A6000", 154.8),
        ("NVIDIA A40", 149.7),
        ("NVIDIA A100-SXM4-80GB", 312.0),
    ],
)
def test_known_cards_have_a_finite_denominator(monkeypatch, name, expected):
    assert _flops_for(monkeypatch, name) == pytest.approx(expected)


def test_the_a6000_branch_does_not_shadow_or_get_shadowed(monkeypatch):
    """The table is a chain of substring tests, so a new entry can be swallowed.

    'RTX A6000' must not match any branch above it, and adding it must not have
    changed what the neighbouring names resolve to.
    """
    assert _flops_for(monkeypatch, "NVIDIA RTX A6000") != _flops_for(monkeypatch, "NVIDIA A40")
    assert _flops_for(monkeypatch, "NVIDIA L40S") == pytest.approx(362.05)
    assert _flops_for(monkeypatch, "NVIDIA L40") == pytest.approx(181.05)


def test_an_unknown_card_still_yields_zero_mfu(monkeypatch):
    """The behaviour this fix is about, pinned so the cause stays legible."""
    promised = _flops_for(monkeypatch, "NVIDIA MADE-UP-9000")
    assert promised == float("inf")
    assert 12.3 / promised == 0.0
