"""Tests for application helpers and hardware-independent branches."""

import builtins
import runpy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import autofisher.audio as audio_module
from autofisher import actions
from autofisher.audio import Template, make_filterbank, sound_features, trim_silence
from autofisher.cli import app as cli_module
from autofisher.devices import choose_loopback_device


def test_actions_can_be_cancelled_before_each_press(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify cancellation prevents the corresponding key press."""
    stop_event = Mock()
    stop_event.wait.side_effect = [True]
    press = Mock()
    monkeypatch.setattr(actions.pyautogui, "press", press)

    actions.pull_and_recast(stop_event, 0.1, 0.2)

    press.assert_not_called()


def test_actions_press_twice_and_use_elapsed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify both presses occur and the second wait accounts for elapsed time."""
    stop_event = Mock()
    stop_event.wait.side_effect = [False, False]
    press = Mock()
    monotonic = Mock(side_effect=[10.0, 10.25])
    monkeypatch.setattr(actions.pyautogui, "press", press)
    monkeypatch.setattr(actions.time, "monotonic", monotonic)

    actions.pull_and_recast(stop_event, 0.1, 0.2)

    assert press.call_count == 2
    assert stop_event.wait.call_args_list[1].args[0] == 0.0


def test_actions_can_be_cancelled_after_first_press(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify cancellation after pulling prevents recasting."""
    stop_event = Mock()
    stop_event.wait.side_effect = [False, True]
    press = Mock()
    monkeypatch.setattr(actions.pyautogui, "press", press)

    actions.pull_and_recast(stop_event, 0.1, 0.2)

    press.assert_called_once_with("r")


def test_trim_silence_handles_short_and_active_audio() -> None:
    """Verify short inputs are preserved and active audio receives padding."""
    short = np.ones(10, dtype=np.float32)
    assert trim_silence(short, 16000) is short

    audio = np.zeros(3200, dtype=np.float32)
    audio[1600:1610] = 1.0
    trimmed = trim_silence(audio, 16000)
    assert len(trimmed) < len(audio)
    assert np.max(trimmed) == 1.0


def test_trim_silence_returns_unmodified_audio_when_no_chunk_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the defensive no-active-chunk path."""
    audio = np.ones(1600, dtype=np.float32)

    def no_active_chunks(_values: np.ndarray) -> np.ndarray:
        return np.array([], dtype=int)

    monkeypatch.setattr(
        audio_module.np,
        "flatnonzero",
        no_active_chunks,
    )
    assert trim_silence(audio, 16000) is audio


def test_audio_feature_and_matching_edge_cases() -> None:
    """Verify empty, short, and zero-energy feature paths."""
    assert sound_features(np.zeros(100, dtype=np.float32)).shape == (32, 0)
    features = sound_features(np.ones(1024, dtype=np.float32))
    assert features.shape[0] == 32
    assert make_filterbank().shape == (32, 257)
    assert (
        audio_module.best_spectral_match(
            np.zeros((32, 2), dtype=np.float32),
            np.ones((32, 3), dtype=np.float32),
            1.0,
        )
        == 0.0
    )
    assert (
        audio_module.best_spectral_match(
            np.zeros((32, 3), dtype=np.float32),
            np.zeros((32, 3), dtype=np.float32),
            0.0,
        )
        == 0.0
    )
    assert audio_module.find_best_template_match(np.zeros((32, 3), dtype=np.float32), []) == (0.0, "")


def test_filterbank_handles_empty_frequency_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the fallback bin is selected when a band has no weights."""

    def empty_weights(values: np.ndarray, minimum: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.minimum(values, minimum))

    monkeypatch.setattr(
        audio_module.np,
        "maximum",
        empty_weights,
    )
    bank = make_filterbank()
    assert np.isfinite(bank).all()


def test_load_templates_skips_short_and_zero_norm_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify short files fail and zero-norm speed variants are ignored."""
    signal = np.ones((100, 1), dtype=np.float32)
    monkeypatch.setattr(audio_module, "TEMPLATE_FILES", ("short.ogg",))
    monkeypatch.setattr(audio_module.sf, "read", Mock(return_value=(signal, 16000)))
    with pytest.raises(RuntimeError, match="No usable templates"):
        audio_module.load_templates(tmp_path)

    signal = np.sin(np.linspace(0, 100, 4000)).astype(np.float32)[:, None]
    monkeypatch.setattr(audio_module.sf, "read", Mock(return_value=(signal, 16000)))
    monkeypatch.setattr(audio_module, "SPEED_VARIANTS", (1.0,))

    def zero_features(_audio: np.ndarray) -> np.ndarray:
        return np.zeros((32, 2), dtype=np.float32)

    monkeypatch.setattr(
        audio_module,
        "sound_features",
        zero_features,
    )
    with pytest.raises(RuntimeError, match="No usable templates"):
        audio_module.load_templates(tmp_path)


def test_load_templates_loads_variants_and_skips_bad_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify usable files produce variants while unreadable files are reported."""
    signal = np.sin(np.linspace(0, 100, 4000)).astype(np.float32)[:, None]
    monkeypatch.setattr(audio_module, "TEMPLATE_FILES", ("good.ogg", "bad.ogg"))
    monkeypatch.setattr(
        audio_module.sf,
        "read",
        Mock(side_effect=[(signal, 16000), OSError("missing")]),
    )
    templates = audio_module.load_templates(tmp_path)
    assert len(templates) == 3
    assert all(isinstance(template, Template) for template in templates)


def test_load_templates_uses_project_assets_when_default_directory_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify default asset directory resolution reaches the project fallback."""
    signal = np.sin(np.linspace(0, 100, 4000)).astype(np.float32)[:, None]
    monkeypatch.setattr(audio_module, "TEMPLATE_FILES", ("good.ogg",))
    monkeypatch.setattr(audio_module.sf, "read", Mock(return_value=(signal, 16000)))
    templates = audio_module.load_templates()
    assert templates


def test_load_templates_rejects_when_nothing_is_usable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify the loader raises when every template fails."""
    monkeypatch.setattr(audio_module, "TEMPLATE_FILES", ("bad.ogg",))
    monkeypatch.setattr(audio_module.sf, "read", Mock(side_effect=OSError("missing")))

    with pytest.raises(RuntimeError, match="No usable templates"):
        audio_module.load_templates(tmp_path)


def test_choose_loopback_device_selects_default_and_explicit_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify default matching and explicit index selection."""
    devices = [
        {"name": "Speakers (loopback)", "index": 1, "maxInputChannels": 2, "defaultSampleRate": 48000},
        {"name": "Headphones (loopback)", "index": 2, "maxInputChannels": 2, "defaultSampleRate": 44100},
        {"name": "Disabled", "index": 3, "maxInputChannels": 0, "defaultSampleRate": 48000},
    ]
    p = Mock()
    p.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 10}
    p.get_loopback_device_info_generator.return_value = iter(devices)
    p.get_device_info_by_index.return_value = {"name": "Headphones"}

    def choose_default(_prompt: str) -> str:
        return ""

    def choose_explicit(_prompt: str) -> str:
        return "0"

    monkeypatch.setattr(builtins, "input", choose_default)
    assert choose_loopback_device(p)["index"] == 2

    monkeypatch.setattr(builtins, "input", choose_explicit)
    p.get_loopback_device_info_generator.return_value = iter(devices)
    assert choose_loopback_device(p)["index"] == 1


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


def test_cli_callback_and_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the callback delegates to listening and main invokes Typer."""
    listen = Mock()
    monkeypatch.setattr(cli_module, "listen", listen)
    cli_module.root()
    listen.assert_called_once_with()

    typer_app = Mock()
    monkeypatch.setattr(cli_module, "app", typer_app)
    cli_module.main()
    typer_app.assert_called_once_with()


def test_module_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the module entry point delegates to the CLI."""
    main = Mock()
    monkeypatch.setattr("autofisher.cli.app.main", main)
    runpy.run_module("autofisher.__main__", run_name="__main__")
    main.assert_called_once_with()


def test_imported_module_entry_point_does_not_run() -> None:
    """Keep the non-entry branch of the module guard covered."""
    runpy.run_module("autofisher.__main__", run_name="autofisher.__main__")


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
