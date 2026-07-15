"""Halation and grain.

Both are sized as a fraction of an edge rather than as a pixel count, so the look
does not change between a downscaled preview and a full-resolution export.
Halation's radius is a fraction of the LONG edge; grain's size is a fraction of
the SHORT edge.
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


def _midtone_weight(luma):
    """Grain amplitude as a function of lightness. Peaks at mid-grey.

    A simplification of darktable's paper-response model, in which the noise
    perturbs *exposure* and is then pushed through a paper S-curve — so the same
    noise yields a large delta on the steep midtones and little on the shoulders.
    The parabola reproduces that shape: maximum at 0.5, vanishing at both ends.
    """
    return (np.float32(4.0) * luma * (np.float32(1.0) - luma)).astype(np.float32)


def add_grain(rgb, intensity: float, size: float, seed: int = 0):
    """Film grain, in display (sRGB-encoded) space.

    `size` is a fraction of the short edge, so the look survives a resize.

    The noise field is MONOCHROME. Independent per-channel noise is chroma
    speckle — it reads as a noisy digital sensor, not as film. And it is
    generated fine and then Gaussian-filtered to size, rather than np.repeat-ed
    into square axis-aligned blocks, so the spectrum is isotropic.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if intensity <= 0 or size <= 0:
        return rgb

    height, width = rgb.shape[:2]
    sigma = float(size) * min(height, width)
    # Clamp: a sigma larger than the frame blurs the field to a flat DC offset,
    # and the renormalisation below would then divide by ~0.
    sigma = max(0.0, min(sigma, min(height, width) / 4.0))

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((height, width)).astype(np.float32)

    if sigma >= 0.5:
        noise = gaussian_blur(noise, sigma)
        deviation = float(noise.std())
        if deviation > 1e-6:
            noise /= np.float32(deviation)  # blurring shrinks variance; restore it

    luma = _luminance(rgb)
    weighted = noise * _midtone_weight(luma) * np.float32(intensity)

    return np.clip(rgb + weighted[:, :, None], 0.0, 1.0).astype(np.float32)
