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


def hue_preserving_clip(rgb):
    """Bring over-range values into [0,1] by scaling, not clipping. DISPLAY only.

    A camera JPEG has already been rendered: at EV=0 nothing in it exceeds 1.0,
    and the LUT downstream was authored against exactly that sRGB space. So the
    only correct transform in front of the LUT is the IDENTITY — and there is no
    softer alternative available. Any monotonic f that is the identity on [0,1]
    and maps [0,inf) into [0,1] IS min(x, 1): f must fix 1.0, and monotonicity
    then pins everything above it at 1.0. A soft shoulder is only correct where
    the input is scene-linear, which is what `highlight_rolloff` below is for.

    That leaves only the question of HOW the over-range values an exposure push
    creates come back down. Clipping each channel on its own crushes them in
    sequence — the brightest channel of a warm specular reaches 1.0 first and the
    highlight drifts toward yellow-white on its way out. Dividing the whole
    triplet by its own maximum preserves the channel ratios, and so the hue.
    """
    rgb = np.asarray(rgb, dtype=np.float32)

    # Pointwise, per pixel: the max is over the CHANNEL axis alone. (An earlier
    # transform in this file took a cumulative max across the flattened image,
    # so one bright pixel dimmed every pixel after it in raster order.)
    norm = np.max(rgb, axis=-1, keepdims=True)

    # >= 1.0 everywhere, so the divide is always safe: below range this is a
    # divide by exactly 1.0 — the identity, bit for bit.
    scale = np.where(norm > np.float32(1.0), norm, np.float32(1.0))

    return (rgb / scale).astype(np.float32)


def highlight_rolloff(rgb, knee: float = 0.8):
    """The scene-to-display shoulder. SCENE (scene-linear) input only.

    Scene-linear light has real headroom above 1.0 that has never been rendered
    by anything, so it needs a shoulder — that shoulder IS the scene-to-display
    render. It must NOT run on a camera JPEG: this curve is asymptotic to 1.0,
    so f(1.0) = 0.926 and white can never come out white.

    Compresses the *norm* and scales the whole triplet by the same factor, so a
    warm specular keeps its channel ratios, and so its hue, on the way down.

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

    return np.clip(rgb * scale, 0.0, 1.0).astype(np.float32)
