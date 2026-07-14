"""Transfer functions and the neutral scene-to-display render.

Deliberately contains no "look". The LUT is the look — the CLUTs this project
applies were authored against a neutral, standard-contrast sRGB render, so
anything opinionated here would be double-counted.
"""

from __future__ import annotations

import numpy as np

SRGB_KNEE = np.float32(0.04045)
SRGB_LINEAR_KNEE = np.float32(0.0031308)


def srgb_decode(x):
    """sRGB-encoded [0,1] -> linear light."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(
        x <= SRGB_KNEE,
        x / np.float32(12.92),
        np.power((np.maximum(x, 0.0) + np.float32(0.055)) / np.float32(1.055),
                 np.float32(2.4)),
    ).astype(np.float32)


def srgb_encode(x):
    """Linear light -> sRGB-encoded [0,1]."""
    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, 0.0)
    return np.where(
        x <= SRGB_LINEAR_KNEE,
        x * np.float32(12.92),
        np.float32(1.055) * np.power(x, np.float32(1.0 / 2.4)) - np.float32(0.055),
    ).astype(np.float32)
