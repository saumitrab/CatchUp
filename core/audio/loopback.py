from __future__ import annotations

import numpy as np
import sounddevice as sd
from typing import Optional, List, Dict


def list_input_devices() -> list[dict]:
    """
    Return all input-capable devices from sounddevice.query_devices().
    """
    devices = sd.query_devices()
    return [d for d in devices if d.get("max_input_channels", 0) > 0]


def find_device(name_contains: str) -> Optional[int]:
    """
    Find the first INPUT device index whose name contains the given substring (case-insensitive).
    Returns None if not found.

    Examples:
        find_device("BlackHole 2ch") -> 0
        find_device("BlackHole")     -> first BlackHole input device
    """
    name_contains = (name_contains or "").lower()
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) <= 0:
            continue
        if name_contains in d.get("name", "").lower():
            return i
    return None


class AudioStream:
    """
    Thin wrapper around sounddevice.InputStream that:
      - opens a specific input device
      - reads frames as float32
      - mixes to mono by taking LEFT channel only (prevents phase cancellation)
      - returns a 1-D numpy array of shape (frames,)
    """

    def __init__(self, device_index: int, samplerate: int, channels: int = 2):
        if channels < 1:
            raise ValueError("channels must be >= 1")
        self.device_index = device_index
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self._stream: Optional[sd.InputStream] = None

    def __enter__(self) -> "AudioStream":
        # dtype float32 keeps things simple and matches Whisper expectations post-resample
        self._stream = sd.InputStream(
            device=self.device_index,
            channels=self.channels,
            samplerate=self.samplerate,
            dtype="float32",
            blocksize=0,  # let PortAudio choose
        )
        self._stream.start()
        return self

    def read(self, frames: int) -> np.ndarray:
        """
        Read 'frames' samples from the device.
        Returns a 1-D float32 array (mono). If the device is multi-channel,
        we select the LEFT channel to avoid potential L/R phase cancellation.
        """
        assert self._stream is not None, "Stream not started. Use 'with AudioStream(...) as s:'"
        data, overflowed = self._stream.read(int(frames))
        # data shape: (frames, channels) float32

        # If stereo (or more), take LEFT channel only to avoid rare phase-cancel artifacts.
        if data.ndim == 2 and data.shape[1] > 1:
            data = data[:, 0:1]

        # Squeeze to 1-D mono
        data = data.astype("float32", copy=False).squeeze()

        # If overflowed, we just return what we have; caller can handle pacing/backpressure.
        # (Optionally, you could log a warning here.)
        return data

    def __exit__(self, exc_type, exc, tb):
        if self._stream:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
            self._stream = None
