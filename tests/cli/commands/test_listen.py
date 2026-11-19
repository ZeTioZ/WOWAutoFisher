"""Tests for the streaming listen command using deterministic fakes."""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest

import autofisher.cli.commands.listen as listen_module


def test_listen_processes_audio_and_stops_cleanly(
    listener_setup: Callable[[float], tuple[Any, Any]],
) -> None:
    """Verify normal listening, status output, and cleanup."""
    p, _thread = listener_setup(0.0)

    listen_module.listen()

    assert p.terminated
    assert p.stream.read_count == 2


def test_listen_triggers_action_and_handles_busy_stream(
    listener_setup: Callable[[float], tuple[Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify confirmation starts an action and busy state suppresses matching."""
    p, thread = listener_setup(0.9)
    monkeypatch.setattr(listen_module.time, "monotonic", iter((10.0, 10.1, 20.0)).__next__)

    listen_module.listen()

    assert thread.started is False
    assert p.terminated


def test_listen_skips_matching_while_action_is_busy(
    listener_setup: Callable[[float], tuple[Any, Any]],
) -> None:
    """Verify an already-running action takes the busy path before matching."""
    p, thread = listener_setup(0.0)
    thread.started = True

    listen_module.listen()

    assert p.terminated


def test_listen_reports_cooldown_after_trigger(
    listener_setup: Callable[[float], tuple[Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify cooldown status is reported when an action has just triggered."""
    p, thread = listener_setup(0.9)
    monkeypatch.setattr(thread, "is_alive", lambda: False)

    listen_module.listen()

    assert p.terminated


def test_listen_cleans_up_when_device_selection_fails(
    listener_setup: Callable[[float], tuple[Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PyAudio is terminated when no stream was opened."""
    p, _thread = listener_setup(0.0)
    monkeypatch.setattr(
        listen_module,
        "choose_loopback_device",
        Mock(side_effect=RuntimeError("no device")),
    )

    with pytest.raises(RuntimeError, match="no device"):
        listen_module.listen()

    assert p.terminated
