"""Rasterize the Film Lab mark (static/img/app-icon.svg) to PNG + ICO.

The mark is a single 35mm frame: a dark film chip, a row of perforations top and
bottom, and a gold exposed window that nearly spans the strip — thin borders, the
way real film sits. The red-orange glow around the window is halation, produced
the way the pipeline produces it: blur the window, keep the light that escaped
its edge. No outer container — the film chip is the whole icon, on transparent.

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

STRIP = "#0B1220"
SPROCKET = "#EEF4FA"

GOLD_STOPS = [(0.00, (0xFF, 0xD3, 0x83)),
              (0.55, (0xFF, 0xB5, 0x47)),
              (1.00, (0xE8, 0x89, 0x1C))]

# Halation scatters long-wavelength light furthest, so the bloom runs red.
HALO_RGB = (0xFF, 0x7A, 0x33)
HALO_SIGMA = 26.0
HALO_GAIN = 0.9

# ── Geometry (1024 space) ────────────────────────────────────────────────────
# The film chip fills the canvas with a small margin. The gold window nearly
# reaches the chip's left/right edges — thin borders — with slim perforation
# bands top and bottom.
STRIP_BOX = (64, 214, 896, 596)   # x, y, w, h
STRIP_R = 26

FRAME = (120, 300, 784, 424)      # ~1.85:1, close to the chip edges
FRAME_R = 12

SPROCKET_W, SPROCKET_H, SPROCKET_R = 52, 40, 10
SPROCKET_Y_TOP, SPROCKET_Y_BOT = 244, 726
# 8 perforations, evenly spaced across the window's width.
SPROCKET_X = [120.0 + i * (784 - SPROCKET_W) / 7.0 for i in range(8)]


def _hex(s: str):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _mask(draw_fn) -> np.ndarray:
    m = Image.new("L", (N * S, N * S), 0)
    draw_fn(ImageDraw.Draw(m))
    return np.asarray(m, dtype=np.float32) / 255.0


def _rrect(xywh, radius):
    x, y, w, h = xywh
    box = [x * S, y * S, (x + w) * S, (y + h) * S]
    return lambda d: d.rounded_rectangle(box, radius=radius * S, fill=255)


def _gold_gradient() -> np.ndarray:
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
    blurred = Image.fromarray((frame * 255).astype(np.uint8), mode="L")
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=HALO_SIGMA * S))
    bloom = np.asarray(blurred, dtype=np.float32) / 255.0
    return np.clip(bloom - frame, 0.0, 1.0) * HALO_GAIN


def _over(base: np.ndarray, rgb, alpha: np.ndarray) -> np.ndarray:
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
    strip = _mask(_rrect(STRIP_BOX, STRIP_R))
    frame = _mask(_rrect(FRAME, FRAME_R))
    sprockets = _mask(_sprockets)

    canvas = np.zeros((N * S, N * S, 3), dtype=np.float32)
    canvas = _over(canvas, _hex(STRIP), strip)
    # Keep the glow on the film chip, around the window.
    canvas = _over(canvas, HALO_RGB, _halation(frame) * strip)
    canvas = _over(canvas, _gold_gradient(), frame)
    canvas = _over(canvas, _hex(SPROCKET), sprockets)

    rgb = np.clip(canvas, 0, 255).astype(np.uint8)
    # The chip is the whole icon: its silhouette is the alpha. No outer container.
    alpha = np.clip(strip * 255.0, 0, 255).astype(np.uint8)
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
