"""Listen command implementation."""

import random
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio

from autofisher.actions import pull_and_recast
from autofisher.audio import (
    find_best_template_match,
    load_templates,
    resample_audio,
    sound_features,
)
from autofisher.config import (
    ANALYSIS_RATE,
    CHUNK,
    CONFIRMATION_CHUNKS,
    COOLDOWN_SECONDS,
    HOP,
    N_FFT,
    PULL_DELAY_RANGE,
    RECAST_DELAY_RANGE,
    SIMILARITY_THRESHOLD,
)
from autofisher.devices import choose_loopback_device


def listen() -> None:
    """Listen through a selected WASAPI device and automate fishing actions.

    :raises RuntimeError: If no usable sound templates or loopback device exists.
    """
    templates = load_templates()
    max_feature_frames = max(template.features.shape[1] for template in templates)
    required_seconds = (N_FFT + (max_feature_frames - 1) * HOP) / ANALYSIS_RATE

    p = pyaudio.PyAudio()
    stream = None
    stop_actions = threading.Event()
    action_thread: threading.Thread | None = None

    try:
        device = choose_loopback_device(p)
        native_rate = int(device["defaultSampleRate"])
        channels = int(device["maxInputChannels"])
        required_samples = int(required_seconds * native_rate)
        buffer_capacity = required_samples + CHUNK
        audio_buffer = np.zeros(
            (buffer_capacity, channels),
            dtype=np.float32,
        )
        valid_frames = 0

        print(f"\nListening to: {device['name']} ({native_rate} Hz, {channels} channels)")
        print(f"Threshold: {SIMILARITY_THRESHOLD:.2f} | Ctrl+C to stop")
        print(f"Pull delay: {PULL_DELAY_RANGE[0]:.2f}-{PULL_DELAY_RANGE[1]:.2f}s")
        print(f"Recast delay: {RECAST_DELAY_RANGE[0]:.2f}-{RECAST_DELAY_RANGE[1]:.2f}s")
        print("\nCast your line manually to begin.\n")

        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=native_rate,
            input=True,
            input_device_index=int(device["index"]),
            frames_per_buffer=CHUNK,
        )

        last_trigger = -float("inf")
        last_print = 0.0
        consecutive_matches = 0

        while True:
            raw = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
            samples = samples.astype(np.float32) / 32768.0
            frames = len(samples)

            audio_buffer[:-frames] = audio_buffer[frames:]
            audio_buffer[-frames:] = samples
            valid_frames = min(buffer_capacity, valid_frames + frames)

            now = time.monotonic()
            action_busy = action_thread is not None and action_thread.is_alive()
            in_cooldown = now - last_trigger < COOLDOWN_SECONDS

            best_score = 0.0
            best_template = ""
            if not action_busy and not in_cooldown and valid_frames >= required_samples:
                recent = audio_buffer[-required_samples:]
                for channel in range(channels):
                    downsampled = resample_audio(
                        recent[:, channel],
                        native_rate,
                    )
                    features = sound_features(downsampled)
                    channel_score, channel_template = find_best_template_match(
                        features,
                        templates,
                    )
                    if channel_score > best_score:
                        best_score = channel_score
                        best_template = channel_template

            if now - last_print >= 0.15:
                volume = float(np.sqrt(np.mean(samples * samples)))
                if action_busy:
                    status = "ACTION"
                elif in_cooldown:
                    status = "COOLDOWN"
                else:
                    status = "LISTENING"
                print(
                    f"Status: {status:9} "
                    f"| Volume: {volume:.4f} "
                    f"| Splash: {best_score:.3f} "
                    f"| Best: {best_template[:30]:30}",
                    end="\r",
                    flush=True,
                )
                last_print = now

            if action_busy or in_cooldown:
                consecutive_matches = 0
                continue

            if best_score >= SIMILARITY_THRESHOLD:
                consecutive_matches += 1
            else:
                consecutive_matches = 0

            if consecutive_matches >= CONFIRMATION_CHUNKS:
                pull_delay = random.uniform(*PULL_DELAY_RANGE)  # nosec B311
                recast_delay = random.uniform(*RECAST_DELAY_RANGE)  # nosec B311
                print(f"\nSplash detected! Score={best_score:.3f}, {best_template}")
                print(f"  Pull in after: {pull_delay:.2f}s")
                print(f"  Recast after:  {recast_delay:.2f}s (from first R)")
                action_thread = threading.Thread(
                    target=pull_and_recast,
                    args=(stop_actions, pull_delay, recast_delay),
                    daemon=True,
                )
                last_trigger = time.monotonic()
                consecutive_matches = 0
                action_thread.start()

    except KeyboardInterrupt:
        print("\nStopped listening.")
    finally:
        stop_actions.set()
        if action_thread is not None:
            action_thread.join()
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()
