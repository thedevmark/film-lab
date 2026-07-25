"""Halation and grain.

Both are sized as a fraction of an edge rather than as a pixel count, so the look
does not change between a downscaled preview and a full-resolution export.
Halation's radius is a fraction of the LONG edge; grain's size is a fraction of
the SHORT edge.
"""

from __future__ import annotations

import math

import numpy as np

from filmlab.blur import box_blur, gaussian_blur

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


# ── Grain ─────────────────────────────────────────────────────────────────────
#
# The constants below were fitted to two lab scans of Kodak colour negative
# (2075x3130, the "Tess and Leo Ramos Wedding on Film" set). The procedure:
# take the flattest 512px crops in each frame, high-pass them at sigma 3 to
# strip scene content, and measure the residual's amplitude against tone, its
# autocorrelation along a row, and its per-channel correlation. Every number
# here is one of those three measurements. They describe THAT scanner and THAT
# stock; they are not a universal model of film.
#
# What the measurement overturned, in order of how much it mattered:
#
#   1. SCALE. The measured autocorrelation is [1.0, 0.21, -0.11, -0.02, ...] —
#      grain that decorrelates within two pixels and slightly overshoots
#      negative, i.e. a BAND-pass field. The previous implementation ran white
#      noise through filmlab.blur.gaussian_blur, which is three integer-radius
#      box passes and therefore cannot resolve a sigma below ~1px at all; at the
#      old default it produced [1.0, 0.96, 0.84, 0.67, ...], correlated over
#      ~6px. That is roughly five times coarser than the film, and low-pass
#      rather than band-pass. It is the single reason the old grain read as
#      blotchy mush instead of grain, and it is why _fine_gaussian below exists
#      rather than reusing gaussian_blur.
#
#   2. THE FLOOR. Grain does not vanish at the ends of the tone scale: the scans
#      hold ~3/255 in the deepest shadows and ~2.9/255 just short of white,
#      against a midtone peak of ~8/255. The old weight was 4L(1-L), which goes
#      to zero at both ends and so left dark frames — exactly the ones that
#      motivated this work — plasticky and clean where the film is not.
#
#   3. CHROMA. The three channels measured 0.75–0.89 correlated, not 1.00. The
#      old code forced a single monochrome field on the grounds that independent
#      per-channel noise reads as sensor speckle. That is true of INDEPENDENT
#      noise and the caution was right; the correction is that the layers are
#      mostly-but-not-fully shared, and forcing them identical is as wrong in
#      the other direction. Red measured grainiest, which is the order the
#      emulsion predicts.

# Amplitude relative to the midtone peak, at the ends of the tone scale.
GRAIN_TONE_FLOOR = 0.31
# Shape of the hump between those ends. The skew pulls the peak below mid-grey
# (measured ~0.40); the exponent sharpens the shoulders. Fitted jointly, mean
# absolute error 0.021 against the nine measured tone bins.
GRAIN_TONE_EXPONENT = 1.65
GRAIN_TONE_SKEW = 0.80

# Fraction of each channel's field shared with the other two.
GRAIN_CHROMA_SHARE = 0.80
# Per-channel amplitude. Red sits in the fastest, coarsest-grained layer.
GRAIN_CHANNEL_GAIN = (1.20, 1.00, 1.05)

# The band-pass: subtract a blur this many times wider than the grain itself.
# This is what puts the negative lobe in the autocorrelation and keeps the field
# from drifting into low-frequency blotching.
GRAIN_BANDPASS_RATIO = 4.0
# Below about a third of a pixel a Gaussian is indistinguishable from a delta,
# and the band-pass subtraction would cancel to zero. Grain finer than the
# sampling grid is white noise, which is the correct answer anyway.
GRAIN_MIN_SIGMA = 0.30


def _fine_gaussian(field, sigma: float):
    """Separable Gaussian with a TRUE float sigma, sub-pixel included.

    gaussian_blur is three box passes with an integer radius floored at 1, so
    every sigma below ~1.5 collapses onto the same ~2px kernel. Halation does
    not care — its sigmas are tens of pixels. Grain lives entirely inside that
    dead zone, so it needs a real kernel. Sigma is small here (a 3-sigma radius
    of 1–3 taps at the sizes grain actually uses), so an explicit FIR is cheap.
    """
    if sigma <= 0:
        return field

    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets * offsets) / np.float32(2.0 * sigma * sigma))
    kernel /= kernel.sum()

    out = field
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        padded = np.pad(out, padding, mode="reflect")
        accumulated = np.zeros_like(out)
        for index, coefficient in enumerate(kernel):
            window = [slice(None), slice(None)]
            window[axis] = slice(index, index + out.shape[axis])
            accumulated += np.float32(coefficient) * padded[tuple(window)]
        out = accumulated
    return out


def _shape_grain(field, sigma: float):
    """White noise -> a band-pass field with the measured correlation length.

    Two different blurs on purpose. The INNER one sets the grain's own size and
    is sub-pixel, so it has to be the exact FIR. The OUTER one only marks where
    the band-pass rolls off underneath the grain — a few percent of error in its
    width moves nothing visible — so it uses the O(n) box-blur, which is what
    keeps a 24MP export in seconds rather than minutes.
    """
    field = _fine_gaussian(field, sigma)
    return field - box_blur(field, sigma * GRAIN_BANDPASS_RATIO)


def _tone_weight(luma):
    """Grain amplitude as a function of lightness.

    A hump that peaks a little below mid-grey and settles onto GRAIN_TONE_FLOOR
    at both ends rather than reaching zero. See the fit note above.
    """
    skewed = np.power(np.clip(luma, 0.0, 1.0), np.float32(GRAIN_TONE_SKEW))
    hump = np.clip(np.float32(4.0) * skewed * (np.float32(1.0) - skewed), 0.0, 1.0)
    hump = np.power(hump, np.float32(GRAIN_TONE_EXPONENT))
    return (np.float32(GRAIN_TONE_FLOOR)
            + np.float32(1.0 - GRAIN_TONE_FLOOR) * hump).astype(np.float32)


def add_grain(rgb, intensity: float, size: float, seed: int = 0):
    """Film grain, in display (sRGB-encoded) space.

    `size` is a fraction of the short edge, so the look survives a resize:
    grain is fixed to the frame, as it is on the negative, not to the pixel
    grid. `intensity` is the standard deviation of the perturbation at the
    midtone peak, in display units — 0.032 reproduces the reference scans.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if intensity <= 0 or size <= 0:
        return rgb

    height, width = rgb.shape[:2]
    sigma = float(size) * min(height, width)
    # Clamp: a sigma larger than the frame leaves nothing for the band-pass to
    # subtract, and the renormalisation below would then divide by ~0.
    sigma = min(max(sigma, GRAIN_MIN_SIGMA), min(height, width) / 8.0)

    rng = np.random.default_rng(seed)
    shared_field = rng.standard_normal((height, width), dtype=np.float32)
    shared = np.float32(math.sqrt(GRAIN_CHROMA_SHARE))
    private = np.float32(math.sqrt(1.0 - GRAIN_CHROMA_SHARE))

    weight = _tone_weight(_luminance(rgb)) * np.float32(intensity)

    out = rgb.copy()
    scale = None
    for channel, gain in enumerate(GRAIN_CHANNEL_GAIN):
        # Each channel is mostly the shared field plus a little of its own. The
        # sqrt weights keep the sum at unit variance, so the channels differ in
        # correlation without differing in amplitude.
        field = shared_field * shared
        field += rng.standard_normal((height, width), dtype=np.float32) * private
        field = _shape_grain(field, sigma)

        # Band-passing costs variance. Restore it — once, from the first
        # channel, so the per-channel gains below survive rather than being
        # normalised away.
        if scale is None:
            deviation = float(field.std())
            scale = np.float32(1.0 / deviation if deviation > 1e-6 else 0.0)

        out[:, :, channel] += field * scale * np.float32(gain) * weight

    return np.clip(out, 0.0, 1.0).astype(np.float32)
