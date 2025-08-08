from __future__ import annotations
import sounddevice as sd
import numpy as np
from typing import Optional, List, Dict

def list_input_devices() -> list[dict]:
    devices = sd.query_devices()
    return [d for d in devices]

def find_device(name_contains: str) -> Optional[int]:
    for i, d in enumerate(sd.query_devices()):
        if name_contains.lower() in d.get("name","").lower() and d.get("max_input_channels",0) > 0:
            return i
    return None

class AudioStream:
    def __init__(self, device_index: int, samplerate: int, channels: int):
        self.device_index = device_index
        self.samplerate = samplerate
        self.channels = channels
        self._stream = None

    def __enter__(self):
        self._stream = sd.InputStream(
            device=self.device_index,
            channels=self.channels,
            samplerate=self.samplerate,
            dtype="float32",
        )
        self._stream.start()
        return self

    def read(self, frames: int) -> np.ndarray:
        data, overflowed = self._stream.read(frames)
        if overflowed:
            # best-effort; caller can drop if needed
            pass
        # mix to mono if necessary
        if data.ndim == 2 and data.shape[1] > 1:
            data = data.mean(axis=1, keepdims=True)
        return data.squeeze()

    def __exit__(self, exc_type, exc, tb):
        if self._stream:
            self._stream.stop()
            self._stream.close()
