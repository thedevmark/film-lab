"""Rasterize the Film Lab mark (static/img/app-icon.svg) to PNG + ICO.

Kept in the repo because there is no SVG rasterizer in the dependency set —
this redraws the same geometry with PIL at 4x and downsamples, so the raster
assets stay reproducible from source rather than being opaque binaries.

The halo is not a hand-drawn gradient: it is a Gaussian blur of the frame mask
with the frame subtracted back out, which is the same blur-and-add operation
film.add_halation() performs on a real photograph. The icon renders itself.

    python tools/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 4  # supersample factor
N = 1024

REPO = Path(__file__).resolve().parent.parent
LOGO_PNG = REPO / "static" / "img" / "logo.png"
FAVICON = REPO / "static" / "favicon.ico"

SHELL_OUTER = "#0D1521"
SHELL_INNER = "#162334"
STRIP = "#0B1220"
SPROCKET = "#EEF4FA"

# Gold frame gradient, corner to corner.
GOLD_STOPS = [(0.00, (0xFF, 0xD3, 0x83)),
              (0.55, (0xFF, 0xB5, 0x47)),
              (1.00, (0xE8, 0x89, 0x1C))]

# Halation scatters long-wavelength light furthest, so the bloom runs red.
HALO_RGB = (0xFF, 0x7A, 0x33)
HALO_SIGMA = 34.0   # in 1024-space px
HALO_GAIN = 0.85

STRIP_BOX = (140, 200, 744, 624)   # x, y, w, h
STRIP_R = 28

# True 3:2 — a 35mm still frame, not a cinema frame.
FRAME = (221, 318, 582, 388)
FRAME_R = 16

SPROCKET_X = [182.9, 299.7, 416.6, 533.4, 650.3, 767.1]
SPROCKET_W, SPROCKET_H, SPROCKET_R = 74, 56, 15
SPROCKET_Y_TOP, SPROCKET_Y_BOT = 232, 736


def _hex(s: str):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _mask(draw_fn) -> np.ndarray:
    """Render a shape to a float mask in [0,1] at supersampled size."""
    m = Image.new("L", (N * S, N * S), 0)
    draw_fn(ImageDraw.Draw(m))
    return np.asarray(m, dtype=np.float32) / 255.0


def _rrect(xywh, radius):
    x, y, w, h = xywh
    box = [x * S, y * S, (x + w) * S, (y + h) * S]
    return lambda d: d.rounded_rectangle(box, radius=radius * S, fill=255)


def _gold_gradient() -> np.ndarray:
    """Diagonal gradient over the frame's bounding box, as a full-canvas RGB array."""
    x0, y0, w, h = FRAME
    u = np.clip((np.arange(N * S, dtype=np.float32) - x0 * S) / (w * S), 0.0, 1.0)[None, :]
    v = np.clip((np.arange(N * S, dtype=np.float32) - y0 * S) / (h * S), 0.0, 1.0)[:, None]
    t = np.clip((u + v) / 2.0, 0.0, 1.0)

    pos = np.array([p for p, _ in GOLD_STOPS], dtype=np.float32)
    out = np.zeros((N * S, N * S, 3), dtype=np.float32)
    for ch in range(3):
        vals = np.array([c[ch] for _, c in GOLD_STOPS], dtype=np.float32)
        out[:, :, ch] = np.interp(t, pos, vals)
    return out


def _halation(frame: np.ndarray) -> np.ndarray:
    """Blur the frame, subtract it back out — what's left is the light that escaped."""
    blurred = Image.fromarray((frame * 255).astype(np.uint8), mode="L")
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=HALO_SIGMA * S))
    bloom = np.asarray(blurred, dtype=np.float32) / 255.0
    return np.clip(bloom - frame, 0.0, 1.0) * HALO_GAIN


def _over(base: np.ndarray, rgb, alpha: np.ndarray) -> np.ndarray:
    """Source-over composite. rgb is a triple or an (H,W,3) array."""
    src = np.asarray(rgb, dtype=np.float32)
    if src.ndim == 1:
        src = src[None, None, :]
    return base * (1.0 - alpha[..., None]) + src * alpha[..., None]


def _sprockets(d):
    for x in SPROCKET_X:
        for y in (SPROCKET_Y_TOP, SPROCKET_Y_BOT):
            d.rounded_rectangle(
                [x * S, y * S, (x + SPROCKET_W) * S, (y + SPROCKET_H) * S],
                radius=SPROCKET_R * S, fill=255,
            )


def build() -> Image.Image:
    shell_outer = _mask(_rrect((0, 0, 1024, 1024), 224))
    shell_inner = _mask(_rrect((40, 40, 944, 944), 200))
    strip = _mask(_rrect(STRIP_BOX, STRIP_R))
    frame = _mask(_rrect(FRAME, FRAME_R))
    sprockets = _mask(_sprockets)

    canvas = np.zeros((N * S, N * S, 3), dtype=np.float32)
    canvas = _over(canvas, _hex(SHELL_OUTER), shell_outer)
    canvas = _over(canvas, _hex(SHELL_INNER), shell_inner)
    canvas = _over(canvas, _hex(STRIP), strip)
    # Bloom spills past the strip onto the panel, the way real halation does.
    canvas = _over(canvas, HALO_RGB, _halation(frame) * shell_outer)
    canvas = _over(canvas, _gold_gradient(), frame)
    canvas = _over(canvas, _hex(SPROCKET), sprockets)

    rgb = np.clip(canvas, 0, 255).astype(np.uint8)
    alpha = np.clip(shell_outer * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")
    return img.resize((N, N), Image.LANCZOS)


if __name__ == "__main__":
    icon = build()
    LOGO_PNG.parent.mkdir(parents=True, exist_ok=True)
    icon.save(LOGO_PNG, format="PNG")
    icon.save(FAVICON, format="ICO",
              sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"wrote {LOGO_PNG}")
    print(f"wrote {FAVICON}")
