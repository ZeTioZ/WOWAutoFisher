"""Tests for WASAPI loopback device selection."""

import builtins
from unittest.mock import Mock

import pytest

from autofisher.devices import choose_loopback_device


@pytest.mark.parametrize(
    ("selection", "expected_index"),
    [
        ("", 2),
        ("0", 1),
    ],
)
def test_choose_loopback_device_selection(
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    expected_index: int,
) -> None:
    """Verify default matching and explicit index selection."""
    devices = [
        {"name": "Speakers (loopback)", "index": 1, "maxInputChannels": 2, "defaultSampleRate": 48000},
        {"name": "Headphones (loopback)", "index": 2, "maxInputChannels": 2, "defaultSampleRate": 44100},
    ]
    p = Mock()
    p.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 10}
    p.get_loopback_device_info_generator.return_value = iter(devices)
    p.get_device_info_by_index.return_value = {"name": "Headphones"}

    def choose(_prompt: str) -> str:
        return selection

    monkeypatch.setattr(builtins, "input", choose)
    assert choose_loopback_device(p)["index"] == expected_index


def test_choose_loopback_device_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify unavailable and invalid device selections are explicit."""
    p = Mock()
    p.get_host_api_info_by_type.side_effect = OSError
    with pytest.raises(RuntimeError, match="WASAPI"):
        choose_loopback_device(p)

    p.get_host_api_info_by_type.side_effect = None
    p.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 1}
    p.get_loopback_device_info_generator.return_value = iter(())
    with pytest.raises(RuntimeError, match="loopback"):
        choose_loopback_device(p)

    device = {"name": "Only", "index": 1, "maxInputChannels": 1, "defaultSampleRate": 1}
    p.get_loopback_device_info_generator.return_value = iter((device,))
    p.get_device_info_by_index.return_value = {"name": "Other"}

    def choose_invalid(_prompt: str) -> str:
        return "bad"

    monkeypatch.setattr(builtins, "input", choose_invalid)
    with pytest.raises(ValueError, match="Invalid"):
        choose_loopback_device(p)


def test_choose_loopback_device_falls_back_to_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the first loopback is used when the default name does not match."""
    device = {"name": "Only", "index": 1, "maxInputChannels": 1, "defaultSampleRate": 1}
    p = Mock()
    p.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 1}
    p.get_loopback_device_info_generator.return_value = iter((device,))
    p.get_device_info_by_index.return_value = {"name": "Other"}

    def choose_first(_prompt: str) -> str:
        return ""

    monkeypatch.setattr(builtins, "input", choose_first)
    assert choose_loopback_device(p) is device
