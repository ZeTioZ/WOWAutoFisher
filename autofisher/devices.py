"""WASAPI loopback device selection."""

from typing import TypedDict

import pyaudiowpatch as pyaudio


class LoopbackDevice(TypedDict):
    """Describe a WASAPI loopback output device."""

    name: str
    index: int
    maxInputChannels: int
    defaultSampleRate: int


def choose_loopback_device(p: pyaudio.PyAudio) -> LoopbackDevice:
    """Prompt for and return a WASAPI loopback output device.

    :param p: Initialized PyAudioWPatch instance.
    :return: Selected loopback device information.
    :raises RuntimeError: If WASAPI or a loopback device is unavailable.
    :raises ValueError: If the requested device number is invalid.
    """
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as exc:
        raise RuntimeError("Windows WASAPI is unavailable") from exc

    loopbacks = [
        device for device in p.get_loopback_device_info_generator() if int(device["maxInputChannels"]) > 0
    ]
    if not loopbacks:
        raise RuntimeError("No WASAPI loopback output devices found")

    print("\n=== AVAILABLE PLAYBACK DEVICES ===")
    for n, device in enumerate(loopbacks):
        print(f"[{n}] {device['name']}")

    default_output = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    default_loopback = next(
        (device for device in loopbacks if default_output["name"] in device["name"]),
        loopbacks[0],
    )
    default_number = loopbacks.index(default_loopback)

    selection = input(f"\nChoose device (Enter = {default_number}): ").strip()
    if not selection:
        return default_loopback

    try:
        return loopbacks[int(selection)]
    except ValueError, IndexError:
        raise ValueError("Invalid device selection") from None
