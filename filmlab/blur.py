"""Float Gaussian blur.

PIL cannot blur float images (mode "F" raises), and blurring through uint8 is
what broke halation: a Gaussian spreads a highlight's energy over ~sigma^2 px,
so a small highlight's blurred peak fell below 1/255 and rounded to zero. Three
box passes converge on a Gaussian by the central limit theorem — the same trick
PIL uses internally, in float.
"""

from __future__ import annotations

import math

import numpy as np


def _box_blur_axis(arr, radius, axis):
    """One box-blur pass along an axis, O(n) in the image and independent of radius."""
    if radius < 1:
        return arr

    moved = np.moveaxis(arr, axis, 0)
    n = moved.shape[0]
    padded = np.pad(moved, [(radius, radius)] + [(0, 0)] * (moved.ndim - 1), mode="edge")
    cumulative = np.cumsum(padded, axis=0, dtype=np.float32)
    zero = np.zeros((1,) + cumulative.shape[1:], dtype=np.float32)
    cumulative = np.concatenate([zero, cumulative], axis=0)

    window = 2 * radius + 1
    out = (cumulative[window:window + n] - cumulative[:n]) / np.float32(window)
    return np.moveaxis(out, 0, axis)


def gaussian_blur(arr, sigma):
    """Gaussian blur, approximated by three box passes."""
    if sigma <= 0:
        return arr

    width = math.sqrt(12.0 * sigma * sigma / 3.0 + 1.0)
    radius = max(1, int(round((width - 1) / 2.0)))

    out = arr.astype(np.float32, copy=True)
    for _ in range(3):
        out = _box_blur_axis(out, radius, axis=0)
        out = _box_blur_axis(out, radius, axis=1)
    return out
