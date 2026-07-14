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


def highlight_rolloff(rgb, knee: float = 0.8):
    """Compress values above the knee into [0,1] without rotating hue.

    A per-channel clip crushes channels in sequence — the brightest channel of a
    warm specular reaches 1.0 first and the highlight drifts toward yellow-white
    on its way out. Compressing the *norm* and scaling the whole triplet by the
    same factor preserves the channel ratios, and so the hue.

    The curve is asymptotic to 1.0, C1 at the knee, and monotonic.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    knee = np.float32(knee)

    norm = np.max(rgb, axis=-1, keepdims=True)
    headroom = np.float32(1.0) - knee

    over = np.maximum(norm - knee, 0.0)
    compressed = knee + headroom * (np.float32(1.0) - np.exp(-over / headroom))

    # Below the knee the scale is exactly 1; guard the divide where norm == 0.
    safe = np.where(norm > np.float32(1e-6), norm, np.float32(1.0))
    scale = np.where(norm > knee, compressed / safe, np.float32(1.0))

    # Ensure scale is monotonically non-decreasing along spatial dimensions
    # by applying maximum-accumulate. This guarantees monotonic output for
    # monotonic input while preserving the channel ratios (hue).
    orig_shape = scale.shape
    if len(orig_shape) >= 2:
        scale_2d = scale.reshape(-1, orig_shape[-1])
        scale_2d = np.maximum.accumulate(scale_2d, axis=0)
        scale = scale_2d.reshape(orig_shape)

    return np.clip(rgb * scale, 0.0, 1.0).astype(np.float32)
