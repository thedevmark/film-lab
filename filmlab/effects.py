"""Halation and grain.

Both operate on a fraction of the long edge rather than a pixel count, so the
look does not change between a downscaled preview and a full-resolution export.
"""

from __future__ import annotations

import numpy as np

from filmlab.blur import gaussian_blur

# Linear Rec.709 luminance.
LUMA = (0.2126, 0.7152, 0.0722)

# The red-sensitive layer sits deepest in the emulsion, so it catches most of
# the light that punched through and reflected off the base. Blue is spared by
# the anti-halation backing. The asymmetry is in the AMOUNT, not the radius.
HALATION_GAIN = (1.0, 0.28, 0.0)
HALATION_THRESHOLD = 0.70  # linear; below this a highlight does not scatter meaningfully


def _luminance(rgb):
    return (rgb[:, :, 0] * np.float32(LUMA[0])
            + rgb[:, :, 1] * np.float32(LUMA[1])
            + rgb[:, :, 2] * np.float32(LUMA[2]))


def add_halation(linear_rgb, intensity: float, radius: float):
    """Highlights scatter, reflect off the film base, and re-expose the emulsion.

    Operates in LINEAR light. `radius` is a fraction of the long edge.

    Strictly additive. A normalised mix — (1-s)*I + s*blur(I) — conserves the
    local mean and so cannot add density around a highlight; it merely softens
    the whole frame and lays colour fringes on every edge, including dark ones.
    That is veiling glare, not halation.
    """
    linear_rgb = np.asarray(linear_rgb, dtype=np.float32)
    if intensity <= 0 or radius <= 0:
        return linear_rgb

    height, width = linear_rgb.shape[:2]
    sigma = float(radius) * max(height, width)
    if sigma <= 0:
        return linear_rgb

    highlights = np.maximum(_luminance(linear_rgb) - np.float32(HALATION_THRESHOLD), 0.0)
    bloom = gaussian_blur(highlights, sigma)

    out = linear_rgb.copy()
    for channel, gain in enumerate(HALATION_GAIN):
        if gain:
            out[:, :, channel] += bloom * np.float32(gain * intensity)
    return out
