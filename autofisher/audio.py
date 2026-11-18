"""Audio resampling, feature extraction, and template matching."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.signal import resample_poly, stft

from autofisher.config import (
    ANALYSIS_RATE,
    HOP,
    MAX_TEMPLATE_SECONDS,
    N_BANDS,
    N_FFT,
    SPEED_VARIANTS,
    TEMPLATE_FILES,
)

type FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Template:
    """Describe one normalized fishing-splash audio template."""

    name: str
    speed: float
    features: FloatArray
    norm: float


def resample_audio(
    audio: FloatArray,
    original_rate: int,
    target_rate: int = ANALYSIS_RATE,
) -> FloatArray:
    """Resample a mono audio signal to the configured analysis rate.

    :param audio: Mono audio samples.
    :param original_rate: Source sample rate in hertz.
    :param target_rate: Destination sample rate in hertz.
    :return: Resampled single-precision audio.
    """
    if original_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    divisor = math.gcd(original_rate, target_rate)
    resampled = resample_poly(
        audio,
        target_rate // divisor,
        original_rate // divisor,
    )
    return resampled.astype(np.float32)


def trim_silence(audio: FloatArray, rate: int) -> FloatArray:
    """Trim leading, trailing, and quiet portions of template audio.

    :param audio: Mono audio samples.
    :param rate: Audio sample rate in hertz.
    :return: Trimmed mono audio samples.
    :raises ValueError: If the template contains only silence.
    """
    step = max(1, int(rate * 0.01))
    count = len(audio) // step
    if count < 2:
        return audio

    chunks = audio[: count * step].reshape(count, step)
    rms = np.sqrt(np.mean(chunks * chunks, axis=1))
    peak = float(np.max(rms))

    if peak < 1e-5:
        raise ValueError("Template is silent")

    active = np.flatnonzero(rms >= max(peak * 0.07, 1e-5))
    if not len(active):
        return audio

    start = max(0, int(active[0] * step - 0.05 * rate))
    end = min(len(audio), int((active[-1] + 1) * step + 0.12 * rate))
    return audio[start:end]


def make_filterbank() -> FloatArray:
    """Create the logarithmic mel-style frequency filter bank.

    :return: Filter-bank weights indexed by band and frequency.
    """
    frequencies = np.fft.rfftfreq(N_FFT, 1.0 / ANALYSIS_RATE)
    anchors = np.geomspace(180, 6500, N_BANDS + 2)
    bank = np.zeros((N_BANDS, len(frequencies)), dtype=np.float32)

    for i in range(N_BANDS):
        left, center, right = anchors[i : i + 3]
        up = (frequencies - left) / (center - left)
        down = (right - frequencies) / (right - center)
        weights = np.maximum(0.0, np.minimum(up, down))

        if weights.sum() < 1e-10:
            weights[np.argmin(abs(frequencies - center))] = 1.0

        bank[i] = weights / weights.sum()

    return bank


FILTERBANK = make_filterbank()


def sound_features(audio: FloatArray) -> FloatArray:
    """Extract normalized logarithmic spectral-band features.

    :param audio: Mono audio samples at the analysis sample rate.
    :return: Spectral feature matrix.
    """
    if len(audio) < N_FFT:
        return np.empty((N_BANDS, 0), dtype=np.float32)

    _, _, spectrum = stft(
        audio,
        fs=ANALYSIS_RATE,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        boundary=None,
        padded=False,
    )

    energies = FILTERBANK @ np.abs(spectrum)
    return np.log(energies + 1e-6).astype(np.float32)


def load_templates(asset_dir: Path | None = None) -> list[Template]:
    """Load and prepare all configured fishing-splash templates.

    :param asset_dir: Optional directory containing template sound files.
    :return: Available template variants at configured speeds.
    :raises RuntimeError: If no usable template can be loaded.
    """
    if asset_dir is None:
        package_sounds = Path(__file__).resolve().parent / "sounds"
        project_sounds = Path(__file__).resolve().parent.parent / "sounds"
        asset_dir = package_sounds if package_sounds.is_dir() else project_sounds

    templates: list[Template] = []
    for filename in TEMPLATE_FILES:
        path = asset_dir / filename
        try:
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            rms = np.sqrt(np.mean(audio * audio, axis=0))
            channel = np.asarray(audio[:, int(np.argmax(rms))], dtype=np.float32)
            channel = resample_audio(channel, int(rate))
            channel = trim_silence(channel, ANALYSIS_RATE)
            channel = channel[: int(MAX_TEMPLATE_SECONDS * ANALYSIS_RATE)]

            if len(channel) < N_FFT + 5 * HOP:
                raise ValueError("Template is too short after trimming")

            for speed in SPEED_VARIANTS:
                changed = resample_poly(channel, 100, round(speed * 100))
                features = sound_features(changed)
                features = features - features.mean(axis=1, keepdims=True)
                norm = float(np.linalg.norm(features))
                if norm >= 1e-5:
                    templates.append(Template(path.name, speed, features, norm))

            print(f"Loaded {path.name} ({len(channel) / ANALYSIS_RATE:.2f}s active audio)")
        except Exception as exc:
            print(f"Cannot load {filename}: {exc}")

    if not templates:
        raise RuntimeError("No usable templates found next to this script.")
    return templates


def best_spectral_match(
    observed: FloatArray,
    template: FloatArray,
    template_norm: float,
) -> float:
    """Find the highest similarity between observed and template features.

    :param observed: Observed spectral feature matrix.
    :param template: Template spectral feature matrix.
    :param template_norm: Euclidean norm of the template features.
    :return: Highest centered cosine similarity score.
    """
    duration = template.shape[1]
    if observed.shape[1] < duration:
        return 0.0

    windows = sliding_window_view(observed, duration, axis=1)
    numerator = np.einsum("bpt,bt->p", windows, template, optimize=True)
    band_sums = windows.sum(axis=2)
    square_sums = np.einsum("bpt,bpt->p", windows, windows, optimize=True)
    centered_energy = square_sums - (band_sums * band_sums).sum(axis=0) / duration
    denominator = template_norm * np.sqrt(np.maximum(centered_energy, 1e-12))
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-8,
    )
    return float(np.max(values)) if len(values) else 0.0


def find_best_template_match(
    observed: FloatArray,
    templates: list[Template],
) -> tuple[float, str]:
    """Match observed features against all prepared templates.

    :param observed: Observed spectral feature matrix.
    :param templates: Prepared template variants.
    :return: Highest match score and a display name for the winning template.
    """
    best_score = 0.0
    best_template = ""

    for template in templates:
        score = best_spectral_match(
            observed,
            template.features,
            template.norm,
        )
        if score > best_score:
            best_score = score
            best_template = f"{template.name} ({template.speed:.2f}x)"

    return best_score, best_template
