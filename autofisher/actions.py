"""Fishing keyboard actions."""

import threading
import time

import pyautogui


def pull_and_recast(
    stop_event: threading.Event,
    pull_delay: float,
    recast_delay: float,
) -> None:
    """Pull the fishing line and recast after randomized delays.

    :param stop_event: Event used to cancel a pending action.
    :param pull_delay: Seconds to wait before pressing the pull key.
    :param recast_delay: Seconds after the first key press to recast.
    """
    if stop_event.wait(pull_delay):
        return

    first_press_at = time.monotonic()
    pyautogui.press("r")

    elapsed = time.monotonic() - first_press_at
    remaining = max(0.0, recast_delay - elapsed)
    if stop_event.wait(remaining):
        return
    pyautogui.press("r")
