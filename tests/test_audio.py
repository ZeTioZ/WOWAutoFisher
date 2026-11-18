"""Tests for audio utilities and the Typer CLI."""

import numpy as np
import pytest
from typer.testing import CliRunner

from autofisher.audio import (
    Template,
    best_spectral_match,
    find_best_template_match,
    resample_audio,
    trim_silence,
)
from autofisher.cli.app import app


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


def test_cli_runs_without_subcommand() -> None:
    """Verify the root CLI help works without a subcommand."""
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "listen" not in result.stdout


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
