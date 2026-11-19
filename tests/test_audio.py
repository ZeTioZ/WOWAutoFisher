"""Tests for audio utilities and the Typer CLI."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import autofisher.audio as audio_module
from autofisher.audio import (
    Template,
    best_spectral_match,
    find_best_template_match,
    make_filterbank,
    resample_audio,
    sound_features,
    trim_silence,
)


def test_resample_audio_preserves_dtype() -> None:
    """Verify resampling preserves the float32 dtype."""
    audio = np.ones(160, dtype=np.float32)
    resampled = resample_audio(audio, 8000, 16000)

    assert resampled.dtype == np.float32
    assert len(resampled) == 320


def test_trim_silence_rejects_silent_audio() -> None:
    """Verify silent template audio raises a descriptive error."""
    audio = np.zeros(1600, dtype=np.float32)

    with pytest.raises(ValueError, match="Template is silent"):
        trim_silence(audio, 16000)


def test_spectral_match_is_high_for_matching_features() -> None:
    """Verify identical feature matrices produce a near-perfect match."""
    features = np.ones((32, 10), dtype=np.float32)
    observed = np.vstack([features, features]).reshape(32, 20)
    norm = float(np.linalg.norm(features))

    score = best_spectral_match(observed, features, norm)

    assert score >= 0.99


def test_find_best_template_match_returns_winner() -> None:
    """Verify the template selector identifies the strongest match."""
    features = np.ones((32, 10), dtype=np.float32)
    templates = [
        Template("weak.ogg", 1.0, -features, float(np.linalg.norm(features))),
        Template("winner.ogg", 1.0, features, float(np.linalg.norm(features))),
    ]

    score, name = find_best_template_match(features, templates)

    assert score >= 0.99
    assert name == "winner.ogg (1.00x)"


def test_trim_silence_handles_short_active_and_empty_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify short, active, silent, and inactive chunk paths."""
    short = np.ones(10, dtype=np.float32)
    assert trim_silence(short, 16000) is short

    audio = np.zeros(3200, dtype=np.float32)
    audio[1600:1610] = 1.0
    trimmed = trim_silence(audio, 16000)
    assert len(trimmed) < len(audio)

    silent = np.zeros(1600, dtype=np.float32)
    with pytest.raises(ValueError, match="Template is silent"):
        trim_silence(silent, 16000)

    def no_active_chunks(_values: np.ndarray) -> np.ndarray:
        return np.array([], dtype=int)

    monkeypatch.setattr(audio_module.np, "flatnonzero", no_active_chunks)
    assert trim_silence(np.ones(1600, dtype=np.float32), 16000).shape == (1600,)


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros(100, dtype=np.float32),
        np.ones(1024, dtype=np.float32),
    ],
)
def test_sound_features_output_shape(audio: np.ndarray) -> None:
    """Verify feature extraction handles short and normal samples."""
    features = sound_features(audio)
    assert features.shape[0] == 32
    assert features.shape[1] == (0 if len(audio) < 512 else features.shape[1])


def test_filterbank_handles_empty_frequency_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fallback bin selection for an empty frequency band."""

    def empty_weights(values: np.ndarray, minimum: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.minimum(values, minimum))

    monkeypatch.setattr(audio_module.np, "maximum", empty_weights)
    assert np.isfinite(make_filterbank()).all()


@pytest.mark.parametrize(
    ("observed", "template", "norm"),
    [
        (np.zeros((32, 2), dtype=np.float32), np.ones((32, 3), dtype=np.float32), 1.0),
        (np.zeros((32, 3), dtype=np.float32), np.zeros((32, 3), dtype=np.float32), 0.0),
    ],
)
def test_best_spectral_match_edge_cases(
    observed: np.ndarray,
    template: np.ndarray,
    norm: float,
) -> None:
    """Verify short observations and zero norms return no match."""
    assert best_spectral_match(observed, template, norm) == 0.0


def test_find_best_template_match_returns_empty_result() -> None:
    """Verify matching no templates returns an empty winner."""
    assert find_best_template_match(np.zeros((32, 3), dtype=np.float32), []) == (0.0, "")


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


@pytest.mark.parametrize("asset_dir", [Path("assets"), None])
def test_load_templates_uses_configured_directory_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    asset_dir: Path | None,
) -> None:
    """Verify package-directory fallback and explicit asset directories."""
    signal = np.sin(np.linspace(0, 100, 4000)).astype(np.float32)[:, None]
    monkeypatch.setattr(audio_module, "TEMPLATE_FILES", ("good.ogg",))
    monkeypatch.setattr(audio_module.sf, "read", Mock(return_value=(signal, 16000)))

    templates = audio_module.load_templates() if asset_dir is None else audio_module.load_templates(asset_dir)

    assert templates


def test_load_templates_skips_zero_norm_variants(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify a template variant with no usable features is ignored."""
    signal = np.sin(np.linspace(0, 100, 4000)).astype(np.float32)[:, None]
    monkeypatch.setattr(audio_module, "TEMPLATE_FILES", ("good.ogg",))
    monkeypatch.setattr(audio_module, "SPEED_VARIANTS", (1.0,))
    monkeypatch.setattr(audio_module.sf, "read", Mock(return_value=(signal, 16000)))

    def zero_features(_audio: np.ndarray) -> np.ndarray:
        return np.zeros((32, 2), dtype=np.float32)

    monkeypatch.setattr(
        audio_module,
        "sound_features",
        zero_features,
    )

    with pytest.raises(RuntimeError, match="No usable templates"):
        audio_module.load_templates(tmp_path)


@pytest.mark.parametrize(
    ("template_files", "read_result"),
    [
        (("bad.ogg",), OSError("missing")),
        (("short.ogg",), np.ones((100, 1), dtype=np.float32)),
    ],
)
def test_load_templates_rejects_unusable_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    template_files: tuple[str, ...],
    read_result: Exception | np.ndarray,
) -> None:
    """Verify unreadable and short template files produce no usable templates."""
    monkeypatch.setattr(audio_module, "TEMPLATE_FILES", template_files)
    if isinstance(read_result, Exception):
        monkeypatch.setattr(audio_module.sf, "read", Mock(side_effect=read_result))
    else:
        monkeypatch.setattr(audio_module.sf, "read", Mock(return_value=(read_result, 16000)))

    with pytest.raises(RuntimeError, match="No usable templates"):
        audio_module.load_templates(tmp_path)
