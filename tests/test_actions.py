"""Tests for fishing keyboard actions."""

from unittest.mock import Mock

import pytest

from autofisher import actions


@pytest.mark.parametrize(
    ("wait_results", "expected_presses"),
    [
        ([True], 0),
        ([False, True], 1),
        ([False, False], 2),
    ],
)
def test_pull_and_recast_cancellation_paths(
    monkeypatch: pytest.MonkeyPatch,
    wait_results: list[bool],
    expected_presses: int,
) -> None:
    """Verify cancellation and elapsed-delay behavior around both key presses."""
    stop_event = Mock()
    stop_event.wait.side_effect = wait_results
    press = Mock()
    monkeypatch.setattr(actions.pyautogui, "press", press)
    monkeypatch.setattr(actions.time, "monotonic", Mock(side_effect=[10.0, 10.25]))

    actions.pull_and_recast(stop_event, 0.1, 0.2)

    assert press.call_count == expected_presses
