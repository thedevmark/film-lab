"""Image loading into linear light, carrying a state tag.

A RAW and a JPEG are not the same kind of data, and the tone map downstream has
to know which one it is holding. That is what the tag is for.

SCENE   - true scene-linear, with highlight headroom. Needs a scene-to-display
          render before it can be shown.
DISPLAY - already rendered by the camera and merely linearised. The S-curve is
          baked in and cannot be undone; applying a second scene-to-display
          transform here would stack two tone curves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from filmlab.tone import srgb_decode

SCENE = "scene"
DISPLAY = "display"

RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".dng", ".rw2", ".pef"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def _fit(pil: Image.Image, max_long_edge: int) -> Image.Image:
    """Downscale on the PIL object — converting to float32 first would cost 4x
    the uint8 size at full resolution, before the cap could apply."""
    width, height = pil.size
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return pil
    scale = max_long_edge / long_edge
    return pil.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS
    )


def _fit_raw16(rgb: np.ndarray, max_long_edge: int) -> np.ndarray:
    """Downscale a (H, W, 3) uint16 array before the float32 conversion.

    PIL has no native mode for 3-channel 16-bit data — only single-channel
    'I;16' (see PIL.Image._fromarray_typemap) — so unlike the 8-bit _fit()
    above this cannot hand the array straight to Image.fromarray(..., mode=
    "RGB"); that raises "Cannot handle this data type: (1, 1, 3), <u2" the
    moment a real 16-bit rawpy.postprocess() output reaches it. Each channel
    is resized on its own as a single-band image instead, then restacked.
    """
    height, width = rgb.shape[:2]
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return rgb
    scale = max_long_edge / long_edge
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    channels = [
        np.asarray(
            Image.fromarray(rgb[:, :, c]).resize(new_size, Image.LANCZOS),
            dtype=np.uint16,
        )
        for c in range(3)
    ]
    return np.stack(channels, axis=-1)


def load_image(path: Path, max_long_edge: int = 6000):
    """Load a photo into linear light. Returns (float32 (H,W,3), state)."""
    suffix = path.suffix.lower()

    if suffix in RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError:
            raise RuntimeError("rawpy is not installed — RAW file support unavailable.")

        with rawpy.imread(str(path)) as raw:
            # gamma=(1,1) and no_auto_bright=True are what make this scene-linear.
            # The defaults apply a BT.709 gamma and an automatic brightness
            # stretch — a full camera rendering — which is exactly the thing we
            # must not have here.
            rgb = raw.postprocess(
                use_camera_wb=True,
                gamma=(1, 1),
                no_auto_bright=True,
                output_bps=16,
            )
        # rawpy already applies orientation, so no exif_transpose here.
        rgb = _fit_raw16(rgb, max_long_edge)
        linear = np.asarray(rgb, dtype=np.float32) / np.float32(65535.0)
        return linear, SCENE

    # A vertically-held shot is stored in landscape layout plus an Orientation
    # tag. Without this the photo comes back rotated 90 degrees, since the output
    # JPEG carries no EXIF to compensate.
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    pil = _fit(pil, max_long_edge)
    encoded = np.asarray(pil, dtype=np.float32) / np.float32(255.0)
    return srgb_decode(encoded), DISPLAY
