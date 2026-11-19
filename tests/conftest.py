"""Shared test fixtures."""

from collections.abc import Callable
from unittest.mock import Mock

import numpy as np
import pytest

import autofisher.cli.commands.listen as listen_module
from autofisher.audio import Template


class FakeStream:
    """Return two chunks, then stop the listener."""

    def __init__(self) -> None:
        self.read_count = 0

    def read(self, _size: int, exception_on_overflow: bool = False) -> bytes:
        """Return stereo samples for two iterations."""
        del exception_on_overflow
        if self.read_count >= 2:
            raise KeyboardInterrupt
        self.read_count += 1
        return np.array([[1000, 500], [1000, 500]], dtype=np.int16).tobytes()

    def stop_stream(self) -> None:
        """Record stream shutdown through the mock replacement."""

    def close(self) -> None:
        """Record stream shutdown through the mock replacement."""


class FakePyAudio:
    """Provide a fake audio API and stream."""

    def __init__(self) -> None:
        self.stream = FakeStream()
        self.terminated = False

    def open(self, **_kwargs: object) -> FakeStream:
        """Return the fake input stream."""
        return self.stream

    def terminate(self) -> None:
        """Record termination."""
        self.terminated = True


class FakeThread:
    """Keep the action busy long enough to exercise that listener branch."""

    def __init__(self, **_kwargs: object) -> None:
        self.started = False

    def start(self) -> None:
        """Mark the action as started."""
        self.started = True

    def is_alive(self) -> bool:
        """Report whether the fake action is active."""
        return self.started

    def join(self) -> None:
        """Complete the fake action."""
        self.started = False


@pytest.fixture
def listener_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[float], tuple[FakePyAudio, FakeThread]]:
    """Install small deterministic listener dependencies."""

    def setup(score: float) -> tuple[FakePyAudio, FakeThread]:
        p = FakePyAudio()
        thread = FakeThread()
        device: dict[str, object] = {
            "name": "Fake loopback",
            "index": 4,
            "maxInputChannels": 2,
            "defaultSampleRate": 1,
        }
        template = Template("splash.ogg", 1.0, np.ones((1, 1), dtype=np.float32), 1.0)
        monkeypatch.setattr(listen_module, "load_templates", lambda: [template])
        monkeypatch.setattr(listen_module.pyaudio, "PyAudio", lambda: p)

        def choose_device(_p: object) -> dict[str, object]:
            return device

        def resample(samples: np.ndarray, _rate: int) -> np.ndarray:
            return samples

        def features(samples: np.ndarray) -> np.ndarray:
            return np.ones((1, 1), dtype=np.float32) * len(samples)

        def match(_features: np.ndarray, _templates: list[Template]) -> tuple[float, str]:
            return score, "splash.ogg (1.00x)"

        def thread_factory(**_kwargs: object) -> FakeThread:
            return thread

        def uniform(*_args: float) -> float:
            return 0.1

        monkeypatch.setattr(listen_module, "choose_loopback_device", choose_device)
        monkeypatch.setattr(listen_module, "resample_audio", resample)
        monkeypatch.setattr(
            listen_module,
            "sound_features",
            features,
        )
        monkeypatch.setattr(
            listen_module,
            "find_best_template_match",
            match,
        )
        monkeypatch.setattr(listen_module.threading, "Thread", thread_factory)
        monkeypatch.setattr(listen_module, "ANALYSIS_RATE", 1)
        monkeypatch.setattr(listen_module, "N_FFT", 1)
        monkeypatch.setattr(listen_module, "HOP", 1)
        monkeypatch.setattr(listen_module, "CHUNK", 2)
        monkeypatch.setattr(listen_module, "COOLDOWN_SECONDS", 3.0)
        monkeypatch.setattr(listen_module, "SIMILARITY_THRESHOLD", 0.5)
        monkeypatch.setattr(listen_module, "CONFIRMATION_CHUNKS", 1)
        monkeypatch.setattr(listen_module, "PULL_DELAY_RANGE", (0.1, 0.1))
        monkeypatch.setattr(listen_module, "RECAST_DELAY_RANGE", (0.2, 0.2))
        monkeypatch.setattr(
            listen_module.time,
            "monotonic",
            Mock(side_effect=[10.0, 10.1, 10.2]),
        )
        monkeypatch.setattr(listen_module.random, "uniform", uniform)
        return p, thread

    return setup
