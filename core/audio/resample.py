from __future__ import annotations
import numpy as np

TARGET_SR = 16000

def resample_to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    """Lightweight linear resample to 16 kHz (good enough for speech)."""
    if sr_in == TARGET_SR:
        return x.astype("float32", copy=False)
    # ensure 1-D float32
    x = x.astype("float32", copy=False).reshape(-1)
    n_out = int(round(len(x) * (TARGET_SR / float(sr_in))))
    if n_out <= 0:
        return x
    # linspace-based linear interpolation (fast, no SciPy dep)
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype="float32")
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype="float32")
    y = np.interp(xq, xp, x).astype("float32", copy=False)
    return y
