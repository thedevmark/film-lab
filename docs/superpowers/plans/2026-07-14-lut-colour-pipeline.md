# LUT Colour Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `film.py`'s hand-tuned colour maths with a real 3D LUT (HaldCLUT, tetrahedral interpolation) applied at the correct point in a linear-light pipeline that branches on input state.

**Architecture:** A new `filmlab/` package holds the image maths as focused modules; `film.py` keeps `process_photo`, presets, and the Flask routes and becomes a thin composition layer. Photos load into **linear light** carrying a state tag (`"scene"` for RAW, `"display"` for already-rendered JPEG/PNG) — because a RAW and a JPEG are not the same kind of data, and the current code's core bug is pretending they are. Exposure and halation happen in linear; a neutral tone map lands the image in sRGB; the LUT applies there; grain is last.

**Tech Stack:** Python 3, numpy, Pillow, Flask, rawpy (optional). Tests via `unittest` (there is no pytest in this repo — do not introduce one).

## Global Constraints

- Tests run with `python -m unittest discover -s tests`. Every task must leave the full suite green.
- No new runtime dependencies. `scipy` is NOT available — blurs go through `filmlab/blur.py`.
- Python floats throughout the image maths are `float32`. Convert once, at load.
- The pre-LUT image must be a **neutral, standard-contrast sRGB render**. No look-contrast before the LUT. This is a hard constraint from the spec — the CLUTs were authored against neutral input.
- Grain is **always last**. Contrast applies after the LUT but before grain.
- Halation is **additive** and red-dominant (`s_R ≫ s_G > s_B ≈ 0`). Never a normalised mix — that conserves the local mean and cannot add density around a highlight.
- Halation and grain sizes are **fractions of the long edge**, never pixel counts, or the look changes between preview and export.
- Existing public names `process_photo`, `register_film_routes`, `coerce_params`, `BUILTIN_PRESETS` must keep working — `app.py` and the existing tests import them.

---

## File Structure

**Create:**
- `filmlab/__init__.py` — package marker, empty
- `filmlab/blur.py` — float Gaussian blur (moved out of `film.py`)
- `filmlab/tone.py` — sRGB transfer functions, hue-preserving highlight rolloff
- `filmlab/lut.py` — HaldCLUT loading, tetrahedral interpolation
- `filmlab/loader.py` — image loading, returns `(linear_rgb, state)`
- `filmlab/effects.py` — halation, grain
- `tools/make_hald.py` — identity HaldCLUT generator (for LUT extraction)
- `docs/extracting-a-lut.md` — how to extract a LUT from a licensed editor
- `tests/test_tone.py`, `tests/test_lut.py`, `tests/test_loader.py`, `tests/test_effects.py`

**Modify:**
- `film.py` — delete `apply_film_color`, `add_grain`, `add_halation`, `_load_image`, `_gaussian_blur`, `_box_blur_axis`; rewrite `process_photo`; re-point presets
- `tests/test_film.py` — remove tests for the deleted maths
- `tests/test_defects.py` — retarget the halation/EXIF tests at their new homes
- `.gitignore` — add `luts/private/`

**Rationale for the split:** `film.py` is at ~500 lines and holds maths, HTTP, and persistence. The maths is the part being rewritten and the part that needs dense unit tests. Splitting it out means each module is small enough to hold in context, and the Flask layer stops being coupled to numpy internals.

---

### Task 1: Package scaffold, blur, and sRGB transfer functions

**Files:**
- Create: `filmlab/__init__.py`, `filmlab/blur.py`, `filmlab/tone.py`
- Create: `tests/test_tone.py`
- Modify: `film.py` (remove `_box_blur_axis`, `_gaussian_blur`; import from `filmlab.blur`)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `filmlab.blur.gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray` — 2D float32 in, 2D float32 out
  - `filmlab.tone.srgb_decode(x: np.ndarray) -> np.ndarray` — sRGB-encoded [0,1] → linear
  - `filmlab.tone.srgb_encode(x: np.ndarray) -> np.ndarray` — linear → sRGB-encoded [0,1]

- [ ] **Step 1: Write the failing test**

Create `tests/test_tone.py`:

```python
import unittest

import numpy as np

from filmlab import tone


class TestSrgbTransfer(unittest.TestCase):
    def test_decode_encode_round_trips(self):
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)

        out = tone.srgb_encode(tone.srgb_decode(x))

        np.testing.assert_allclose(out, x, atol=1e-5)

    def test_known_anchors(self):
        # sRGB is linear below 0.04045, a 2.4 power above it.
        x = np.array([0.0, 0.04045, 0.5, 1.0], dtype=np.float32)

        linear = tone.srgb_decode(x)

        self.assertAlmostEqual(float(linear[0]), 0.0, places=6)
        self.assertAlmostEqual(float(linear[1]), 0.04045 / 12.92, places=5)
        self.assertAlmostEqual(float(linear[2]), 0.21404, places=4)  # mid-grey
        self.assertAlmostEqual(float(linear[3]), 1.0, places=5)

    def test_decode_is_monotonic(self):
        x = np.linspace(0.0, 1.0, 512, dtype=np.float32)

        linear = tone.srgb_decode(x)

        self.assertTrue(np.all(np.diff(linear) > 0))


class TestGaussianBlur(unittest.TestCase):
    def test_blur_conserves_energy(self):
        from filmlab.blur import gaussian_blur

        a = np.zeros((64, 64), dtype=np.float32)
        a[32, 32] = 1.0

        out = gaussian_blur(a, sigma=4.0)

        self.assertAlmostEqual(float(out.sum()), 1.0, places=3)

    def test_small_highlight_survives_in_float(self):
        """The uint8 blur path rounded a small highlight's spread peak to zero."""
        from filmlab.blur import gaussian_blur

        a = np.zeros((128, 128), dtype=np.float32)
        a[63:65, 63:65] = 1.0

        out = gaussian_blur(a, sigma=12.0)

        self.assertGreater(float(out.sum()), 3.0)
        self.assertGreater(len(np.unique(np.round(out, 6))), 100)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_tone -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'filmlab'`

- [ ] **Step 3: Write the implementation**

Create `filmlab/__init__.py` (empty file).

Create `filmlab/blur.py` — this is `_box_blur_axis` / `_gaussian_blur` lifted verbatim from `film.py`, where they already work and are already tested:

```python
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
```

Create `filmlab/tone.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_tone -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Remove the duplicated blur from `film.py`**

In `film.py`, delete the `_box_blur_axis` and `_gaussian_blur` function definitions entirely, and add near the other imports:

```python
from filmlab.blur import gaussian_blur
```

Then in `add_halation`, change the call `_gaussian_blur(red, float(radius))` to `gaussian_blur(red, float(radius))`.

- [ ] **Step 6: Run the full suite**

Run: `python -m unittest discover -s tests`
Expected: OK, 29 tests (24 existing + 5 new)

- [ ] **Step 7: Commit**

```bash
git add filmlab tests/test_tone.py film.py
git commit -m "Add filmlab package with float blur and sRGB transfer functions"
```

---

### Task 2: Hue-preserving highlight rolloff

**Files:**
- Modify: `filmlab/tone.py`
- Modify: `tests/test_tone.py`

**Interfaces:**
- Consumes: `filmlab.tone.srgb_encode`, `filmlab.tone.srgb_decode`
- Produces: `filmlab.tone.highlight_rolloff(rgb: np.ndarray, knee: float = 0.8) -> np.ndarray` — (H,W,3) linear in, (H,W,3) linear in [0,1] out

**Why this exists:** after a `× 2^EV` exposure push, linear values exceed 1.0. A naive per-channel `clip(0,1)` clips channels *in sequence* — the red channel of a warm specular hits 1.0 first, then green, and the highlight rotates hue toward yellow-white on its way to clipping. Instead, compress the **norm** of the RGB triplet and scale the triplet by the same factor. Ratios are preserved, so hue is preserved, and the shoulder is soft.

This same function serves both pipeline branches — the RAW path uses it as the scene-to-display shoulder, the JPEG path uses it to catch what the EV push pushed past 1.0. That is why there is no separate `filmic()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tone.py`, inside the file but as a new class:

```python
class TestHighlightRolloff(unittest.TestCase):
    def test_below_knee_is_untouched(self):
        rgb = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)

        out = tone.highlight_rolloff(rgb, knee=0.8)

        np.testing.assert_allclose(out, rgb, atol=1e-6)

    def test_output_never_exceeds_one(self):
        rgb = np.array([[[8.0, 4.0, 2.0], [100.0, 100.0, 100.0]]], dtype=np.float32)

        out = tone.highlight_rolloff(rgb, knee=0.8)

        self.assertTrue(np.all(out <= 1.0 + 1e-6))
        self.assertTrue(np.all(out >= 0.0))

    def test_hue_is_preserved_through_the_shoulder(self):
        """A per-channel clip would rotate this toward yellow-white. Ratios must hold."""
        rgb = np.array([[[4.0, 2.0, 1.0]]], dtype=np.float32)

        out = tone.highlight_rolloff(rgb, knee=0.8)[0, 0]

        # Input ratios are 4 : 2 : 1. They must survive.
        self.assertAlmostEqual(float(out[0] / out[1]), 2.0, places=4)
        self.assertAlmostEqual(float(out[1] / out[2]), 2.0, places=4)

    def test_is_monotonic_and_continuous_at_the_knee(self):
        ramp = np.linspace(0.0, 6.0, 2000, dtype=np.float32)
        rgb = np.stack([ramp, ramp, ramp], axis=-1)[None, :, :]

        out = tone.highlight_rolloff(rgb, knee=0.8)[0, :, 0]

        self.assertTrue(np.all(np.diff(out) >= -1e-7), "rolloff must be monotonic")
        # No step at the knee: the largest jump should be tiny.
        self.assertLess(float(np.abs(np.diff(out)).max()), 0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_tone -v`
Expected: FAIL with `AttributeError: module 'filmlab.tone' has no attribute 'highlight_rolloff'`

- [ ] **Step 3: Write the implementation**

Append to `filmlab/tone.py`:

```python
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

    return np.clip(rgb * scale, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_tone -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add filmlab/tone.py tests/test_tone.py
git commit -m "Add hue-preserving highlight rolloff

A per-channel clip crushes channels in sequence, so a warm specular rotates
toward yellow-white as it clips. Compress the RGB norm and scale the triplet
by the same factor: ratios hold, hue holds, and the shoulder is soft.

Serves both pipeline branches — the scene-to-display shoulder for RAW, and
the catch for what the exposure push sent past 1.0 on an already-rendered JPEG."
```

---

### Task 3: HaldCLUT loading and the identity generator

**Files:**
- Create: `filmlab/lut.py`, `tools/make_hald.py`, `tests/test_lut.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `filmlab.lut.identity_cube(size: int) -> np.ndarray` — returns `(S,S,S,3)` float32, `cube[r,g,b] == (r,g,b)/(S-1)`
  - `filmlab.lut.load_hald(path: Path) -> np.ndarray` — returns `(S,S,S,3)` float32 cube indexed `[r,g,b]`
  - `tools/make_hald.py` — CLI, writes a 16-bit identity Hald PNG

**The layout, stated exactly** (get this wrong and every colour is subtly displaced):
A Hald image of **level N** encodes a cube of **S = N²** entries per axis, laid out as an **N³ × N³** pixel image. Level 8 → 512×512 image → 64³ cube. Pixels are read in plain raster order into a flat `S³ × 3` table, where the index is:

```
i = r + S*g + S**2 * b        # red varies fastest, blue slowest
```

So `flat.reshape(S, S, S, 3)` yields an array indexed `[b, g, r]`, and `.transpose(2, 1, 0, 3)` turns it into the `[r, g, b]` cube we want. Read 16-bit PNGs at 16 bits — Pat David's originals were authored at 16-bit, and reading them as 8 quantises the table before we interpolate it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_lut.py`:

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from filmlab import lut


class TestIdentityCube(unittest.TestCase):
    def test_identity_cube_maps_each_node_to_itself(self):
        cube = lut.identity_cube(8)

        self.assertEqual(cube.shape, (8, 8, 8, 3))
        np.testing.assert_allclose(cube[0, 0, 0], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(cube[7, 7, 7], [1.0, 1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(cube[7, 0, 0], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(cube[0, 7, 0], [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(cube[0, 0, 7], [0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(cube[3, 5, 2],
                                   [3 / 7.0, 5 / 7.0, 2 / 7.0], atol=1e-6)


class TestLoadHald(unittest.TestCase):
    def _write_identity_hald(self, level, directory, bits=16):
        size = level * level
        cube = lut.identity_cube(size)
        # Back to raster order: [r,g,b] -> [b,g,r] -> flat -> square image.
        flat = cube.transpose(2, 1, 0, 3).reshape(-1, 3)
        side = level ** 3
        img = flat.reshape(side, side, 3)

        path = Path(directory) / f"identity_{level}_{bits}.png"
        if bits == 16:
            Image.fromarray((img * 65535).round().astype(np.uint16)).save(path)
        else:
            Image.fromarray((img * 255).round().astype(np.uint8)).save(path)
        return path

    def test_round_trips_an_identity_hald_at_16_bit(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write_identity_hald(8, d, bits=16)

            cube = lut.load_hald(path)

            self.assertEqual(cube.shape, (64, 64, 64, 3))
            np.testing.assert_allclose(cube, lut.identity_cube(64), atol=1e-4)

    def test_round_trips_an_identity_hald_at_8_bit(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write_identity_hald(8, d, bits=8)

            cube = lut.load_hald(path)

            np.testing.assert_allclose(cube, lut.identity_cube(64), atol=1e-2)

    def test_infers_level_from_image_size(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write_identity_hald(4, d)  # 64x64 image -> 16^3 cube

            cube = lut.load_hald(path)

            self.assertEqual(cube.shape, (16, 16, 16, 3))

    def test_rejects_a_non_hald_image(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "not_a_hald.png"
            Image.new("RGB", (100, 100)).save(path)

            with self.assertRaises(ValueError):
                lut.load_hald(path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_lut -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'filmlab.lut'`

- [ ] **Step 3: Write the implementation**

Create `filmlab/lut.py`:

```python
"""HaldCLUT loading and application.

A Hald image of level N encodes a cube of S = N**2 entries per axis, laid out as
an N**3 x N**3 image. Level 8 -> 512x512 -> 64**3. Pixels are read in raster
order into a flat table indexed  i = r + S*g + S**2 * b  (red fastest).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def identity_cube(size: int) -> np.ndarray:
    """The cube that maps every colour to itself. cube[r, g, b] == (r,g,b)/(size-1)."""
    axis = np.arange(size, dtype=np.float32) / np.float32(size - 1)
    r = axis[:, None, None]
    g = axis[None, :, None]
    b = axis[None, None, :]
    shape = (size, size, size)
    return np.stack([
        np.broadcast_to(r, shape),
        np.broadcast_to(g, shape),
        np.broadcast_to(b, shape),
    ], axis=-1).astype(np.float32)


def load_hald(path: Path) -> np.ndarray:
    """Load a HaldCLUT PNG into a (S,S,S,3) float32 cube indexed [r, g, b]."""
    img = Image.open(path)
    if img.mode not in ("RGB", "I;16", "RGB;16"):
        img = img.convert("RGB")

    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"{path.name}: not an RGB image")
    arr = arr[:, :, :3]

    height, width = arr.shape[:2]
    if height != width:
        raise ValueError(f"{path.name}: HaldCLUT must be square, got {width}x{height}")

    # width == level**3, so level is its cube root.
    level = int(round(width ** (1.0 / 3.0)))
    if level ** 3 != width:
        raise ValueError(
            f"{path.name}: {width}px is not a cube number — not a HaldCLUT"
        )

    size = level * level
    scale = np.float32(65535.0) if arr.dtype == np.uint16 else np.float32(255.0)
    flat = arr.reshape(-1, 3).astype(np.float32) / scale

    if flat.shape[0] != size ** 3:
        raise ValueError(f"{path.name}: expected {size ** 3} entries, got {flat.shape[0]}")

    # Raster order is [b, g, r]; we want [r, g, b].
    return np.ascontiguousarray(flat.reshape(size, size, size, 3).transpose(2, 1, 0, 3))
```

Create `tools/make_hald.py`:

```python
"""Write an identity HaldCLUT PNG.

Render this image through a colour tool and the result IS that tool's colour
transform, captured as a LUT. See docs/extracting-a-lut.md.

    python tools/make_hald.py --level 8 --out identity_hald_8.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from filmlab.lut import identity_cube


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an identity HaldCLUT PNG.")
    parser.add_argument("--level", type=int, default=8,
                        help="Hald level; 8 -> 512x512 image, 64^3 cube (default: 8)")
    parser.add_argument("--out", type=Path, default=Path("identity_hald_8.png"))
    args = parser.parse_args()

    size = args.level ** 2
    side = args.level ** 3

    cube = identity_cube(size)
    # [r,g,b] -> raster order [b,g,r] -> flat -> square.
    flat = cube.transpose(2, 1, 0, 3).reshape(-1, 3)
    img = (flat.reshape(side, side, 3) * 65535).round().astype(np.uint16)

    Image.fromarray(img).save(args.out)
    print(f"wrote {args.out}  ({side}x{side}, level {args.level}, {size}^3 cube)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_lut -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Verify the generator produces a loadable Hald**

Run:
```bash
python tools/make_hald.py --level 8 --out /tmp/identity_hald_8.png
python -c "
from pathlib import Path
import numpy as np
from filmlab import lut
cube = lut.load_hald(Path('/tmp/identity_hald_8.png'))
np.testing.assert_allclose(cube, lut.identity_cube(64), atol=1e-4)
print('round trip OK', cube.shape)
"
```
Expected: `round trip OK (64, 64, 64, 3)`

- [ ] **Step 6: Commit**

```bash
git add filmlab/lut.py tools/make_hald.py tests/test_lut.py
git commit -m "Add HaldCLUT loader and identity Hald generator

A Hald of level N encodes an S=N^2 cube as an N^3 x N^3 image, read in raster
order with red varying fastest. Read 16-bit PNGs at 16 bits: the originals were
authored at 16-bit, and reading them as 8 quantises the table before we ever
interpolate it."
```

---

### Task 4: Tetrahedral interpolation

**Files:**
- Modify: `filmlab/lut.py`, `tests/test_lut.py`

**Interfaces:**
- Consumes: `filmlab.lut.identity_cube`, `filmlab.lut.load_hald`
- Produces: `filmlab.lut.apply_lut(rgb: np.ndarray, cube: np.ndarray, strength: float = 1.0) -> np.ndarray` — (H,W,3) sRGB-encoded [0,1] in and out

**Why tetrahedral and not trilinear:** in the standard six-tetrahedra decomposition, the P000–P111 diagonal is a shared edge of *every* tetrahedron. So an input with `R == G == B` interpolates only from lattice nodes that are themselves on the neutral diagonal — **greys stay grey by construction**. Trilinear draws from all eight corners, most of them off-axis, and tints neutrals. Tetrahedral is also cheaper: four lattice fetches instead of eight.

The six cases, selected by the ordering of the fractional offsets `(dr, dg, db)`:

```
dr > dg > db :  (1-dr)·c000 + (dr-dg)·c100 + (dg-db)·c110 + db·c111
dr > db > dg :  (1-dr)·c000 + (dr-db)·c100 + (db-dg)·c101 + dg·c111
db > dr > dg :  (1-db)·c000 + (db-dr)·c001 + (dr-dg)·c101 + dg·c111
db > dg > dr :  (1-db)·c000 + (db-dg)·c001 + (dg-dr)·c011 + dr·c111
dg > db > dr :  (1-dg)·c000 + (dg-db)·c010 + (db-dr)·c011 + dr·c111
dg > dr > db :  (1-dg)·c000 + (dg-dr)·c010 + (dr-db)·c110 + db·c111
```

When `dr == dg == db` every case collapses to `(1-d)·c000 + d·c111` — a blend of two neutral nodes. That is the guarantee.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lut.py`:

```python
class TestApplyLut(unittest.TestCase):
    def test_identity_lut_is_a_no_op(self):
        cube = lut.identity_cube(64)
        rng = np.random.default_rng(0)
        rgb = rng.random((32, 32, 3), dtype=np.float32)

        out = lut.apply_lut(rgb, cube)

        np.testing.assert_allclose(out, rgb, atol=1e-5)

    def test_greys_stay_grey(self):
        """The whole reason for tetrahedral over trilinear.

        A LUT that is neutral on its diagonal must leave R==G==B inputs neutral.
        Trilinear pulls from off-axis corners and tints them; tetrahedral cannot,
        because the neutral diagonal is a shared edge of all six tetrahedra.
        """
        cube = lut.identity_cube(16)
        # Warp the cube off-axis, but keep the neutral diagonal exactly neutral.
        rng = np.random.default_rng(1)
        cube = np.clip(cube + rng.normal(0, 0.05, cube.shape), 0, 1).astype(np.float32)
        for i in range(16):
            cube[i, i, i] = np.float32(i / 15.0)

        grey = np.linspace(0, 1, 256, dtype=np.float32)
        rgb = np.stack([grey, grey, grey], axis=-1)[None, :, :]

        out = lut.apply_lut(rgb, cube)

        np.testing.assert_allclose(out[..., 0], out[..., 1], atol=1e-5)
        np.testing.assert_allclose(out[..., 1], out[..., 2], atol=1e-5)

    def test_hits_lattice_nodes_exactly(self):
        size = 8
        cube = lut.identity_cube(size)
        cube[2, 3, 4] = np.array([0.9, 0.1, 0.5], dtype=np.float32)

        rgb = np.array([[[2 / 7.0, 3 / 7.0, 4 / 7.0]]], dtype=np.float32)
        out = lut.apply_lut(rgb, cube)

        np.testing.assert_allclose(out[0, 0], [0.9, 0.1, 0.5], atol=1e-5)

    def test_strength_blends_toward_the_input(self):
        size = 8
        cube = np.zeros((size, size, size, 3), dtype=np.float32)  # maps everything to black
        rgb = np.full((4, 4, 3), 0.6, dtype=np.float32)

        half = lut.apply_lut(rgb, cube, strength=0.5)
        none = lut.apply_lut(rgb, cube, strength=0.0)
        full = lut.apply_lut(rgb, cube, strength=1.0)

        np.testing.assert_allclose(none, rgb, atol=1e-6)
        np.testing.assert_allclose(full, 0.0, atol=1e-6)
        np.testing.assert_allclose(half, rgb * 0.5, atol=1e-6)

    def test_out_of_range_input_is_clipped_not_wrapped(self):
        cube = lut.identity_cube(16)
        rgb = np.array([[[-0.5, 1.5, 0.5]]], dtype=np.float32)

        out = lut.apply_lut(rgb, cube)

        np.testing.assert_allclose(out[0, 0], [0.0, 1.0, 0.5], atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_lut -v`
Expected: FAIL with `AttributeError: module 'filmlab.lut' has no attribute 'apply_lut'`

- [ ] **Step 3: Write the implementation**

Append to `filmlab/lut.py`:

```python
def apply_lut(rgb, cube, strength: float = 1.0):
    """Apply a 3D LUT with tetrahedral interpolation.

    Input and output are sRGB-encoded [0,1]. Values outside the range are
    clipped, not wrapped — the LUT is only defined on the unit cube.

    Tetrahedral rather than trilinear because the neutral diagonal is a shared
    edge of all six tetrahedra: an R==G==B input therefore interpolates only
    between lattice nodes that are themselves neutral, so greys stay grey by
    construction. Trilinear weights all eight corners, most of them off-axis,
    and tints what should be a pure neutral. It is also cheaper: four fetches
    instead of eight.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if strength <= 0.0:
        return rgb.copy()

    size = cube.shape[0]
    clipped = np.clip(rgb, 0.0, 1.0)

    scaled = clipped * np.float32(size - 1)
    base = np.clip(np.floor(scaled), 0, size - 2).astype(np.int32)
    frac = (scaled - base).astype(np.float32)

    ir, ig, ib = base[..., 0], base[..., 1], base[..., 2]
    dr, dg, db = frac[..., 0], frac[..., 1], frac[..., 2]

    def node(orr, og, ob):
        return cube[ir + orr, ig + og, ib + ob]

    c000 = node(0, 0, 0)
    c111 = node(1, 1, 1)

    # Six tetrahedra, selected by the ordering of (dr, dg, db). Each case is a
    # weighted sum of four nodes: c000, c111, and two edge-adjacent corners.
    out = np.empty(rgb.shape, dtype=np.float32)

    w = lambda x: x[..., None]  # noqa: E731 - broadcast a weight over RGB

    cond_rg = dr > dg
    cond_gb = dg > db
    cond_rb = dr > db

    # dr > dg > db
    m = cond_rg & cond_gb
    out = np.where(w(m),
                   w(1 - dr) * c000 + w(dr - dg) * node(1, 0, 0)
                   + w(dg - db) * node(1, 1, 0) + w(db) * c111,
                   0.0).astype(np.float32)

    # dr > db >= dg
    m = cond_rg & ~cond_gb & cond_rb
    out += np.where(w(m),
                    w(1 - dr) * c000 + w(dr - db) * node(1, 0, 0)
                    + w(db - dg) * node(1, 0, 1) + w(dg) * c111,
                    0.0).astype(np.float32)

    # db >= dr > dg
    m = cond_rg & ~cond_gb & ~cond_rb
    out += np.where(w(m),
                    w(1 - db) * c000 + w(db - dr) * node(0, 0, 1)
                    + w(dr - dg) * node(1, 0, 1) + w(dg) * c111,
                    0.0).astype(np.float32)

    # db > dg >= dr
    m = ~cond_rg & cond_gb
    out += np.where(w(m),
                    w(1 - db) * c000 + w(db - dg) * node(0, 0, 1)
                    + w(dg - dr) * node(0, 1, 1) + w(dr) * c111,
                    0.0).astype(np.float32)

    # dg >= db > dr
    m = ~cond_rg & ~cond_gb & ~cond_rb
    out += np.where(w(m),
                    w(1 - dg) * c000 + w(dg - db) * node(0, 1, 0)
                    + w(db - dr) * node(0, 1, 1) + w(dr) * c111,
                    0.0).astype(np.float32)

    # dg >= dr > db  (the remaining case; also where dr == dg == db lands)
    m = ~cond_rg & ~cond_gb & cond_rb
    out += np.where(w(m),
                    w(1 - dg) * c000 + w(dg - dr) * node(0, 1, 0)
                    + w(dr - db) * node(1, 1, 0) + w(db) * c111,
                    0.0).astype(np.float32)

    if strength >= 1.0:
        return out
    s = np.float32(strength)
    return (s * out + (np.float32(1.0) - s) * rgb).astype(np.float32)
```

**Note on the case split:** the six masks must be mutually exclusive and cover
everything. With `cond_rg = dr > dg`, `cond_gb = dg > db`, `cond_rb = dr > db`,
the combination `~cond_rg & cond_gb & ...` is impossible when `dr <= dg` and
`dg > db` unless `dr > db` or not — both are covered by the last two cases. When
all three deltas are equal, every comparison is `False`, so the final case fires
and collapses to `(1-d)·c000 + d·c111`. That is the neutral-axis guarantee, and
`test_greys_stay_grey` is what pins it.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_lut -v`
Expected: PASS, 10 tests

If `test_greys_stay_grey` fails, the case masks are not partitioning correctly — print `m.sum()` for each of the six and confirm they sum to exactly the pixel count with no overlap.

- [ ] **Step 5: Commit**

```bash
git add filmlab/lut.py tests/test_lut.py
git commit -m "Add tetrahedral LUT interpolation

Tetrahedral, not trilinear: the neutral diagonal is a shared edge of all six
tetrahedra, so an R==G==B input interpolates only between nodes that are
themselves neutral and greys stay grey by construction. Trilinear weights all
eight corners, most off-axis, and tints neutrals. It is also cheaper — four
lattice fetches instead of eight."
```

---

### Task 5: The loader, and the state tag

**Files:**
- Create: `filmlab/loader.py`, `tests/test_loader.py`
- Modify: `film.py` (delete `_load_image`)
- Modify: `tests/test_defects.py` (retarget the EXIF test)

**Interfaces:**
- Consumes: `filmlab.tone.srgb_decode`
- Produces:
  - `filmlab.loader.SCENE = "scene"`, `filmlab.loader.DISPLAY = "display"`
  - `filmlab.loader.load_image(path: Path, max_long_edge: int = 6000) -> tuple[np.ndarray, str]` — returns `(linear_rgb_float32, state)`

**This is the task that fixes the real bug.** The current `_load_image` calls `rawpy.postprocess(use_camera_wb=True, output_bps=8)` — which applies BT.709 gamma, **auto-brightness**, highlight clipping, and returns 8 bits. That is a complete camera rendering. Linearising it recovers a linearised camera render, not scene light. Passing `gamma=(1,1), no_auto_bright=True, output_bps=16` is what actually yields scene-linear data with highlight headroom.

A JPEG, by contrast, is *irreducibly* display-referred — the camera already applied its S-curve and we cannot undo it. So it gets tagged `DISPLAY` and never sees a scene-to-display transform. Hence the tag.

- [ ] **Step 1: Write the failing test**

Create `tests/test_loader.py`:

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from filmlab import loader


class TestLoadImage(unittest.TestCase):
    def _jpeg(self, directory, size=(40, 20), colour=(128, 64, 32), orientation=None):
        pil = Image.new("RGB", size, colour)
        path = Path(directory) / "photo.jpg"
        if orientation is not None:
            exif = pil.getexif()
            exif[274] = orientation
            pil.save(path, exif=exif)
        else:
            pil.save(path)
        return path

    def test_rendered_input_is_tagged_display(self):
        with tempfile.TemporaryDirectory() as d:
            arr, state = loader.load_image(self._jpeg(d))

        self.assertEqual(state, loader.DISPLAY)

    def test_rendered_input_is_linearised(self):
        """Mid-grey sRGB 0.5 must come back as linear ~0.214, not 0.5."""
        with tempfile.TemporaryDirectory() as d:
            path = self._jpeg(d, colour=(128, 128, 128))
            arr, _ = loader.load_image(path)

        self.assertAlmostEqual(float(arr.mean()), 0.2158, delta=0.01)

    def test_exif_orientation_is_applied(self):
        """A vertically-held shot is stored landscape plus Orientation=6."""
        with tempfile.TemporaryDirectory() as d:
            path = self._jpeg(d, size=(40, 20), orientation=6)
            arr, _ = loader.load_image(path)

        self.assertEqual(arr.shape[:2], (40, 20))

    def test_downscales_past_the_long_edge_cap(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._jpeg(d, size=(800, 400))
            arr, _ = loader.load_image(path, max_long_edge=100)

        self.assertEqual(max(arr.shape[:2]), 100)

    def test_extreme_aspect_ratio_does_not_collapse_to_zero(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "wide.png"
            Image.new("RGB", (2000, 2)).save(path)

            arr, _ = loader.load_image(path, max_long_edge=100)

        self.assertGreaterEqual(arr.shape[0], 1)
        self.assertGreaterEqual(arr.shape[1], 1)

    def test_output_is_float32_in_zero_one(self):
        with tempfile.TemporaryDirectory() as d:
            arr, _ = loader.load_image(self._jpeg(d))

        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.all((arr >= 0.0) & (arr <= 1.0)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_loader -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'filmlab.loader'`

- [ ] **Step 3: Write the implementation**

Create `filmlab/loader.py`:

```python
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
        pil = Image.fromarray(rgb, mode="RGB")  # rawpy already applies orientation
        pil = _fit(pil, max_long_edge)
        linear = np.asarray(pil, dtype=np.float32) / np.float32(65535.0)
        return linear, SCENE

    # A vertically-held shot is stored in landscape layout plus an Orientation
    # tag. Without this the photo comes back rotated 90 degrees, since the output
    # JPEG carries no EXIF to compensate.
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    pil = _fit(pil, max_long_edge)
    encoded = np.asarray(pil, dtype=np.float32) / np.float32(255.0)
    return srgb_decode(encoded), DISPLAY
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_loader -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Remove the old loader from `film.py`**

Delete the `_load_image` function from `film.py` entirely. Leave `process_photo` calling it for now — it will break, which is correct: Task 8 rewires it. To keep the suite green in the meantime, add at the top of `film.py`:

```python
from filmlab.loader import load_image as _load_new_image
```

and change `process_photo`'s first line from `img = _load_image(input_path)` to:

```python
    linear, state = _load_new_image(input_path)
    # Interim shim: the old maths below is display-referred, so re-encode.
    # Task 8 removes this entirely.
    from filmlab.tone import srgb_encode
    img = srgb_encode(linear) if state == "display" else srgb_encode(linear)
```

Move the EXIF test out of `tests/test_defects.py` (it now lives in `tests/test_loader.py`) by deleting the `TestExifOrientation` class from `test_defects.py`.

- [ ] **Step 6: Run the full suite**

Run: `python -m unittest discover -s tests`
Expected: OK

- [ ] **Step 7: Commit**

```bash
git add filmlab/loader.py tests/test_loader.py tests/test_defects.py film.py
git commit -m "Load photos into linear light, tagged by input state

rawpy.postprocess with its defaults applies a BT.709 gamma, an automatic
brightness stretch, and highlight clipping, then hands back 8 bits — a complete
camera rendering. Linearising that recovers a linearised camera render, not
scene light, and any tone map on top of it is a second rendering. Pass
gamma=(1,1), no_auto_bright=True, output_bps=16 and it is actually scene-linear.

A JPEG is irreducibly display-referred, so it is tagged as such and must never
see a scene-to-display transform. The tag is what the tone map dispatches on."
```

---

### Task 6: Halation, rewritten

**Files:**
- Create: `filmlab/effects.py`, `tests/test_effects.py`
- Modify: `film.py` (delete `add_halation`)
- Modify: `tests/test_defects.py` (delete `TestHalationPrecision` — superseded)
- Modify: `tests/test_film.py` (delete `test_add_halation_boosts_red_highlight_bloom`)

**Interfaces:**
- Consumes: `filmlab.blur.gaussian_blur`
- Produces: `filmlab.effects.add_halation(linear_rgb, intensity: float, radius: float) -> np.ndarray` — operates in **linear light**; `radius` is a **fraction of the long edge**, not pixels

**What changed and why.** The old signature took `radius` in pixels, which meant the look changed between a downscaled preview and a full-resolution export. It also ran in gamma-encoded space. And the model it should have had — the one the review corrected — is *additive*: light punched through the emulsion, reflected off the base, and **re-exposed** the film. A normalised mix (`(1-s)·I + s·blur(I)`) conserves the local mean and therefore cannot add density around a highlight; it just softens the frame. The per-channel asymmetry is in **amount**, not radius: the red-sensitive layer sits deepest, so it catches most of the reflected light. Blue is essentially untouched.

- [ ] **Step 1: Write the failing test**

Create `tests/test_effects.py`:

```python
import unittest

import numpy as np

from filmlab import effects


class TestHalation(unittest.TestCase):
    def _highlight(self, shape=(200, 300), size=3):
        img = np.full(shape + (3,), 0.02, dtype=np.float32)
        cy, cx = shape[0] // 2, shape[1] // 2
        img[cy - size:cy + size, cx - size:cx + size] = 1.0
        return img

    def test_zero_intensity_is_identity(self):
        img = self._highlight()

        out = effects.add_halation(img, intensity=0.0, radius=0.01)

        np.testing.assert_allclose(out, img)

    def test_halation_adds_energy_rather_than_softening(self):
        """The rejected formula was a normalised mix, which conserves the local
        mean. Halation re-exposes the film: it must ADD."""
        img = self._highlight()

        out = effects.add_halation(img, intensity=0.5, radius=0.03)

        self.assertGreater(float(out.sum()), float(img.sum()))

    def test_bloom_is_red_dominant_and_spares_blue(self):
        img = self._highlight()

        bloom = effects.add_halation(img, intensity=0.5, radius=0.03) - img

        self.assertGreater(float(bloom[:, :, 0].sum()), float(bloom[:, :, 1].sum()))
        self.assertAlmostEqual(float(bloom[:, :, 2].sum()), 0.0, places=5)

    def test_small_highlight_still_blooms(self):
        """A 2px specular used to produce exactly zero halation."""
        img = self._highlight(size=1)

        bloom = effects.add_halation(img, intensity=0.5, radius=0.02) - img

        self.assertGreater(float(bloom[:, :, 0].sum()), 0.0)

    def test_bloom_is_not_posterised(self):
        img = self._highlight()

        bloom = effects.add_halation(img, intensity=0.5, radius=0.03) - img

        self.assertGreater(len(np.unique(np.round(bloom[:, :, 0], 6))), 100)

    def test_radius_is_resolution_independent(self):
        """The same preset must look the same on a preview and on an export."""
        small = self._highlight(shape=(200, 300), size=2)
        large = self._highlight(shape=(400, 600), size=4)

        b_small = effects.add_halation(small, 0.5, radius=0.05) - small
        b_large = effects.add_halation(large, 0.5, radius=0.05) - large

        # Bloom extent, as a fraction of the frame, must match.
        frac_small = float((b_small[:, :, 0] > 1e-4).sum()) / small[:, :, 0].size
        frac_large = float((b_large[:, :, 0] > 1e-4).sum()) / large[:, :, 0].size

        self.assertAlmostEqual(frac_small, frac_large, delta=0.03)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_effects -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'filmlab.effects'`

- [ ] **Step 3: Write the implementation**

Create `filmlab/effects.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_effects -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Delete the superseded code and tests**

- Delete `add_halation` from `film.py`.
- Delete the `TestHalationPrecision` class from `tests/test_defects.py`.
- Delete `test_add_halation_boosts_red_highlight_bloom` from `tests/test_film.py`.
- In `film.py`'s `process_photo`, change the `add_halation(...)` call to import from the new module: add `from filmlab.effects import add_halation` at the top and delete the old local definition. The `radius` param is now a fraction — Task 8 fixes the preset values; until then the halation will be enormous, which is expected and is why Task 8 immediately follows.

- [ ] **Step 6: Run the full suite**

Run: `python -m unittest discover -s tests`
Expected: OK

- [ ] **Step 7: Commit**

```bash
git add filmlab/effects.py tests/test_effects.py film.py tests/test_defects.py tests/test_film.py
git commit -m "Rewrite halation: additive, in linear light, resolution-independent

Halation is light that punched through the emulsion, reflected off the base,
and re-exposed the film. It is strictly additive. A normalised mix conserves
the local mean and therefore cannot add density around a highlight — it just
softens the frame and lays colour fringes on every edge, including dark ones.
That is veiling glare.

The per-channel asymmetry is in the amount, not the radius: the red layer sits
deepest and catches the reflected light, and the anti-halation backing spares
blue. Radius is now a fraction of the long edge, so the look no longer changes
between a preview and an export."
```

---

### Task 7: Grain, rewritten

**Files:**
- Modify: `filmlab/effects.py`, `tests/test_effects.py`
- Modify: `film.py` (delete `add_grain`)
- Modify: `tests/test_film.py`, `tests/test_defects.py` (delete superseded grain tests)

**Interfaces:**
- Consumes: `filmlab.blur.gaussian_blur`
- Produces: `filmlab.effects.add_grain(rgb, intensity: float, size: float, seed: int = 0) -> np.ndarray` — operates in **display (sRGB-encoded) space**; `size` is a **fraction of the short edge**

**Three things were wrong.** `np.random.normal(size=(h, w, 3))` generates *independent noise per channel* — that is chroma speckle, and it reads as a noisy digital sensor, not film. `np.repeat(np.repeat(...))` produces square, axis-aligned blocks rather than an isotropic grain spectrum. And `weight = 1.0 - luma * 0.85` puts **maximum grain in the blacks**, monotonically — backwards, since real RMS granularity peaks in the midtones. It was also unseeded, so re-processing one photo gave a different result every time and a batch would boil.

The midtone weighting here is `4·L·(1-L)`, a deliberate simplification of darktable's `paper_resp` model (where noise perturbs *exposure* and is pushed through a paper S-curve, so the same noise produces a large ΔL on the steep part of the curve and little on the shoulders). The parabola peaks at mid-grey and vanishes at both ends, which is the behaviour that matters.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_effects.py`:

```python
class TestGrain(unittest.TestCase):
    def test_zero_intensity_is_identity(self):
        img = np.full((32, 32, 3), 0.5, dtype=np.float32)

        out = effects.add_grain(img, intensity=0.0, size=0.01)

        np.testing.assert_allclose(out, img)

    def test_grain_is_monochrome_not_chroma_speckle(self):
        """Independent per-channel noise reads as a noisy sensor, not film.
        The three channels must receive the SAME perturbation."""
        img = np.full((64, 64, 3), 0.5, dtype=np.float32)

        grain = effects.add_grain(img, intensity=0.2, size=0.02, seed=7) - img

        np.testing.assert_allclose(grain[:, :, 0], grain[:, :, 1], atol=1e-6)
        np.testing.assert_allclose(grain[:, :, 1], grain[:, :, 2], atol=1e-6)

    def test_grain_peaks_in_the_midtones(self):
        """The old weight (1 - luma*0.85) put maximum grain in the blacks."""
        shadow = np.full((96, 96, 3), 0.05, dtype=np.float32)
        midtone = np.full((96, 96, 3), 0.50, dtype=np.float32)
        highlight = np.full((96, 96, 3), 0.97, dtype=np.float32)

        def amplitude(img):
            return float((effects.add_grain(img, 0.2, 0.02, seed=3) - img).std())

        self.assertGreater(amplitude(midtone), amplitude(shadow))
        self.assertGreater(amplitude(midtone), amplitude(highlight))

    def test_is_deterministic_under_a_seed(self):
        img = np.full((32, 32, 3), 0.5, dtype=np.float32)

        a = effects.add_grain(img, 0.2, 0.02, seed=42)
        b = effects.add_grain(img, 0.2, 0.02, seed=42)
        c = effects.add_grain(img, 0.2, 0.02, seed=43)

        np.testing.assert_allclose(a, b)
        self.assertFalse(np.allclose(a, c))

    def test_size_is_resolution_independent(self):
        small = np.full((100, 150, 3), 0.5, dtype=np.float32)
        large = np.full((200, 300, 3), 0.5, dtype=np.float32)

        g_small = effects.add_grain(small, 0.2, size=0.05, seed=1) - small
        g_large = effects.add_grain(large, 0.2, size=0.05, seed=1) - large

        # Same relative grain size => similar amplitude, not wildly different.
        self.assertAlmostEqual(float(g_small.std()), float(g_large.std()), delta=0.02)

    def test_absurd_size_does_not_explode(self):
        img = np.full((64, 64, 3), 0.5, dtype=np.float32)

        out = effects.add_grain(img, 0.05, size=1000.0, seed=1)

        self.assertEqual(out.shape, img.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_output_stays_in_range(self):
        img = np.full((64, 64, 3), 0.99, dtype=np.float32)

        out = effects.add_grain(img, 0.5, 0.02, seed=1)

        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_effects -v`
Expected: FAIL with `AttributeError: module 'filmlab.effects' has no attribute 'add_grain'`

- [ ] **Step 3: Write the implementation**

Append to `filmlab/effects.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_effects -v`
Expected: PASS, 13 tests

- [ ] **Step 5: Delete the superseded code and tests**

- Delete `add_grain` from `film.py`; import from `filmlab.effects` instead.
- Delete `test_add_grain_zero_intensity_is_identity` from `tests/test_film.py`.
- Delete `test_absurd_grain_size_does_not_allocate_the_universe` from `tests/test_defects.py` (superseded by `test_absurd_size_does_not_explode`).

- [ ] **Step 6: Run the full suite**

Run: `python -m unittest discover -s tests`
Expected: OK

- [ ] **Step 7: Commit**

```bash
git add filmlab/effects.py tests/test_effects.py film.py tests/test_film.py tests/test_defects.py
git commit -m "Rewrite grain: monochrome, midtone-peaked, isotropic, seeded

Three things were wrong. The noise was generated independently per channel,
which is chroma speckle and reads as a noisy sensor rather than film. It was
np.repeat-ed into square axis-aligned blocks instead of being filtered to an
isotropic spectrum. And the weight (1 - luma*0.85) put maximum grain in the
blacks, monotonically — backwards, since real RMS granularity peaks in the
midtones.

It is also seeded now, so one photo through one preset is reproducible and a
batch does not boil. Size is a fraction of the short edge."
```

---

### Task 8: Wire the pipeline, and pull the trigger on `apply_film_color`

**Files:**
- Modify: `film.py` (rewrite `process_photo`, delete `apply_film_color` / `apply_exposure` / `apply_contrast`, re-point `BUILTIN_PRESETS`, delete the provenance comment)
- Modify: `tests/test_film.py` (delete tests for the deleted maths; rewrite the pipeline-order test)
- Modify: `.gitignore`
- Create: `luts/README.md`, `docs/extracting-a-lut.md`

**Interfaces:**
- Consumes: everything from Tasks 1–7
- Produces: `film.process_photo(input_path: Path, params: dict) -> bytes` (unchanged signature)

**This is the phase that is allowed to fail.** Step 8 is a visual A/B gate. If the LUT does not land closer to your DxO finals than the hand-tuned maths did, **stop** — the LUT is wrong, and the correct response is to fix the LUT, not to press on and tune the sliders until it looks acceptable.

**Pipeline order, exactly:**

```
load -> (linear, state)
exposure     linear x 2**EV
halation     linear, additive
[if state == SCENE]  grey-point scale
rolloff      hue-preserving, both branches
srgb_encode  + clip [0,1]
LUT          tetrahedral, strength blend
contrast     post-LUT user finish, default 0
grain        LAST
-> JPEG
```

Contrast is after the LUT because the CLUTs were authored against a **neutral, standard-contrast** render — putting look-contrast before the LUT double-counts it. Grain is after contrast because grain is the texture the emulsion leaves on a *finished* frame, not a signal that gets graded.

- [ ] **Step 1: Write the failing test**

Replace the pipeline-order test in `tests/test_film.py` with:

```python
class TestPipeline(unittest.TestCase):
    def test_pipeline_runs_in_the_documented_order(self):
        import film
        from filmlab import loader

        order = []
        source = np.full((8, 8, 3), 0.25, dtype=np.float32)

        def record(name, fn):
            def inner(*args, **kwargs):
                order.append(name)
                return fn(*args, **kwargs)
            return inner

        with patch.object(film, "load_image",
                          return_value=(source.copy(), loader.DISPLAY)), \
             patch.object(film, "add_halation",
                          side_effect=record("halation", lambda img, **k: img)), \
             patch.object(film, "apply_lut",
                          side_effect=record("lut", lambda img, cube, strength=1.0: img)), \
             patch.object(film, "add_grain",
                          side_effect=record("grain", lambda img, **k: img)):
            out = film.process_photo(Path("x.jpg"), {"grain_intensity": 0.05})

        self.assertEqual(order, ["halation", "lut", "grain"])
        self.assertTrue(out.startswith(b"\xff\xd8"))

    def test_exposure_is_multiplicative_not_additive(self):
        """+1 EV must double the linear light, preserving channel ratios (and so
        hue). The old `img + bias` was a black-level lift in gamma space."""
        import film

        linear = np.full((4, 4, 3), 0.1, dtype=np.float32)
        linear[:, :, 0] = 0.2  # a colour, so a hue shift would show

        out = film.apply_exposure(linear, 1.0)

        np.testing.assert_allclose(out[:, :, 0], 0.4, atol=1e-5)
        np.testing.assert_allclose(out[:, :, 1], 0.2, atol=1e-5)
        # Ratio held => hue held.
        np.testing.assert_allclose(out[:, :, 0] / out[:, :, 1],
                                   linear[:, :, 0] / linear[:, :, 1], atol=1e-5)

    def test_apply_film_color_is_gone(self):
        import film
        self.assertFalse(hasattr(film, "apply_film_color"),
                         "the hand-tuned colour maths should have been deleted")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_film -v`
Expected: FAIL — `apply_film_color` still exists, `apply_exposure` is still additive

- [ ] **Step 3: Rewrite `film.py`'s pipeline section**

Delete from `film.py`: `apply_film_color`, `apply_contrast`, the old `apply_exposure`, the provenance comment block (the `# Derived from a scan of 45 shoot folders...` lines and the two scene-family paragraphs), and any leftover imports of `ImageFilter`.

Replace the pipeline section with:

```python
from pathlib import Path

import numpy as np

from filmlab.effects import add_grain, add_halation
from filmlab.loader import SCENE, load_image
from filmlab.lut import apply_lut, identity_cube, load_hald
from filmlab.tone import highlight_rolloff, srgb_encode

LUT_DIR = Path(__file__).parent / "luts"
GREY_SCENE = 0.18   # linear scene middle grey
GREY_DISPLAY = 0.18  # where we want it to land before the LUT

_LUT_CACHE: dict[str, np.ndarray] = {}


def get_lut(name: str) -> np.ndarray:
    """Load a LUT by name, from luts/private/ first, then luts/open/."""
    if name in _LUT_CACHE:
        return _LUT_CACHE[name]

    for folder in ("private", "open"):
        path = LUT_DIR / folder / f"{name}.png"
        if path.exists():
            cube = load_hald(path)
            _LUT_CACHE[name] = cube
            return cube

    # No LUT installed: fall through to a no-op rather than failing the render.
    cube = identity_cube(2)
    _LUT_CACHE[name] = cube
    return cube


def apply_exposure(linear_rgb, ev: float):
    """Exposure is a multiply in linear light.

    Doubling the photons doubles the linear value. That preserves the ratios
    between channels, and so preserves hue and saturation exactly. Adding a
    constant to gamma-encoded values — which is what this used to do — is a
    black-level lift whose effective gain varies per channel, which is why it
    shifted hue.
    """
    if ev == 0.0:
        return linear_rgb
    return linear_rgb * np.float32(2.0 ** ev)


def apply_contrast(rgb, strength: float):
    """Post-LUT user finish. Never baked into a film preset — the CLUT is the look."""
    if strength == 0.0:
        return rgb
    return np.clip(0.5 + (rgb - 0.5) * np.float32(1.0 + strength), 0.0, 1.0)


def process_photo(input_path: Path, params: dict) -> bytes:
    """load -> exposure -> halation -> render -> LUT -> contrast -> grain -> JPEG."""
    params = coerce_params(params)

    linear, state = load_image(input_path)

    linear = apply_exposure(linear, params["exposure_bias"])
    linear = add_halation(
        linear,
        intensity=params["halation_intensity"],
        radius=params["halation_radius"],
    )

    if state == SCENE:
        # Land scene middle grey where the CLUTs expect it. This is the whole of
        # the scene-to-display render: no look, no contrast curve. The CLUTs were
        # authored against a neutral render, and anything opinionated here would
        # be counted twice.
        linear = linear * np.float32(GREY_DISPLAY / GREY_SCENE)

    # Both branches: catch whatever the exposure push sent past 1.0, without
    # rotating hue the way a per-channel clip would.
    linear = highlight_rolloff(linear)

    rgb = np.clip(srgb_encode(linear), 0.0, 1.0)

    rgb = apply_lut(rgb, get_lut(params["lut"]), strength=params["grade_strength"])
    rgb = apply_contrast(rgb, params["contrast_strength"])
    rgb = add_grain(
        rgb,
        intensity=params["grain_intensity"],
        size=params["grain_size"],
        seed=params["seed"],
    )

    pil_out = Image.fromarray((rgb * 255).clip(0, 255).astype("uint8"))
    buf = io.BytesIO()
    pil_out.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf.read()
```

- [ ] **Step 4: Update `PARAM_SPEC`, `DEFAULT_PARAMS`, and `coerce_params`**

`grain_size` and `halation_radius` are now **fractions**, and there are two new keys. Replace the spec block in `film.py`:

```python
PARAM_SPEC = {
    "grade_strength":     (float, 0.0,   1.0),
    "exposure_bias":      (float, -5.0,  5.0),    # now EV stops, not an offset
    "contrast_strength":  (float, -1.0,  1.0),
    "grain_intensity":    (float, 0.0,   1.0),
    "grain_size":         (float, 0.0,   0.05),   # fraction of the short edge
    "halation_intensity": (float, 0.0,   1.0),
    "halation_radius":    (float, 0.0,   0.10),   # fraction of the long edge
    "seed":               (int,   0,     2 ** 31 - 1),
}

DEFAULT_PARAMS = {
    "grade_strength":     0.85,
    "exposure_bias":      0.0,
    "contrast_strength":  0.0,
    "grain_intensity":    0.055,
    "grain_size":         0.0015,
    "halation_intensity": 0.45,
    "halation_radius":    0.010,
    "seed":               0,
}
```

`lut` is a *string*, so it cannot go through the numeric `PARAM_SPEC`. Add it explicitly in `coerce_params`, just before the `return clean`:

```python
    lut_name = params.get("lut", "kodak_gold_200")
    if not isinstance(lut_name, str) or not lut_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("lut: expected a simple name")
    clean["lut"] = lut_name
```

and add `"lut": "kodak_gold_200"` to `DEFAULT_PARAMS`.

- [ ] **Step 5: Re-point the presets, and delete the provenance claim**

Replace `BUILTIN_PRESETS` in `film.py`:

```python
# Built-in presets. These are aesthetic starting points, not measurements.
#
# An earlier version of this file claimed they were "derived from a scan of 45
# shoot folders / 2,602 raw-to-edited pairs". That claim did not survive review:
# mean luminance, per-channel means, saturation and shadow percentile carry no
# information at all about grain amplitude, grain size, halation radius, or
# grade strength — five of the seven parameters were never derivable from the
# statistics they were attributed to. The two that were have an unexcluded
# alternative explanation in crop bias (photographers crop toward the subject
# and away from blown edges, which moves mean luminance on its own).
#
# exposure_bias and contrast_strength are deliberately 0 in every preset. The
# LUT carries the colour; exposure is per-photo and always was.

BUILTIN_PRESETS = {
    "Ambient Film": {
        "lut":                "kodak_gold_200",
        "grade_strength":     0.85,
        "exposure_bias":      0.0,
        "contrast_strength":  0.0,
        "grain_intensity":    0.065,
        "grain_size":         0.0018,
        "halation_intensity": 0.50,
        "halation_radius":    0.010,
    },
    "Flash Film": {
        "lut":                "kodak_gold_200",
        "grade_strength":     0.50,
        "exposure_bias":      0.0,
        "contrast_strength":  0.0,
        "grain_intensity":    0.040,
        "grain_size":         0.0012,
        "halation_intensity": 0.30,
        "halation_radius":    0.012,
    },
}
```

- [ ] **Step 6: Add the LUT directories and documentation**

```bash
mkdir -p luts/open luts/private
```

Append to `.gitignore`:

```
# Your own extracted LUTs. Never committed — they may be derived from
# commercial software you licensed, and that licence is yours, not the repo's.
luts/private/
```

Create `luts/README.md`:

```markdown
# LUTs

Film colour renderings live here as HaldCLUT PNGs.

- `open/` — openly-licensed LUTs, shipped with the repo. Check each one's
  attribution before redistributing.
- `private/` — **gitignored.** Your own LUTs, including any you extract from an
  editor you have licensed. They stay on your disk.

A LUT is selected by name: `{"lut": "kodak_gold_200"}` loads
`luts/private/kodak_gold_200.png` if it exists, otherwise
`luts/open/kodak_gold_200.png`. If neither exists the colour stage is a no-op,
so the app still runs with no LUTs installed.

To make your own, see [../docs/extracting-a-lut.md](../docs/extracting-a-lut.md).
```

Create `docs/extracting-a-lut.md`:

```markdown
# Extracting a LUT from an editor you have licensed

A 3D LUT is just a mapping from every input colour to an output colour. So if
you render an image *containing every colour* through a colour tool, the result
**is** that tool's colour transform. That image is called an identity HaldCLUT.

This is for your own use, on software you have licensed. Do not redistribute the
result — the colour science in it is not yours to give away.

## 1. Generate the identity Hald

```bash
python tools/make_hald.py --level 8 --out identity_hald_8.png
```

That is a 512x512 16-bit PNG containing all 262,144 colours of a 64³ cube.

## 2. Render it through your editor

Open it, apply **only** the film emulation preset, and export it — same size, no
resize, no crop, PNG.

**Turn everything else off.** This matters more than it sounds:

- **Grain — off.** This is the trap. Grain is *noise*, and the Hald's pixels are
  the LUT's lattice nodes. Leaving grain on gives every node an independent noise
  sample: you get a LUT that is noise-corrupted at every entry, and it will look
  "filmic" enough that you will not notice for weeks.
- **Halation, vignetting, blur — off.** A 3D LUT is a *pointwise* map. It cannot
  represent any effect that depends on neighbouring pixels, so these cannot be
  captured; they would only smear the table. Film Lab renders them itself.
- **Auto-corrections — off.** Smart Lighting, ClearView, auto-exposure, lens
  corrections, denoising. Each one contaminates the table with a transform that
  has nothing to do with the film.
- **Sharpening — off.** Same reason as halation.

## 3. Install it

Save the exported PNG as `luts/private/<name>.png` and select it with
`{"lut": "<name>"}`. `luts/private/` is gitignored.

## 4. Check it

Apply it to a photo at `grade_strength: 1.0` and look at a neutral grey card or
a white wall. If it has taken on a colour cast, something was left on in step 2.
```

- [ ] **Step 7: Run the full suite**

Run: `python -m unittest discover -s tests`
Expected: OK. Fix any test in `test_film.py` / `test_defects.py` that still references deleted functions or the old pixel-based param ranges.

- [ ] **Step 8: THE GATE — visual A/B against real photos**

This is the acceptance test. Nothing above proves the output is *good*.

1. Extract a Kodak Gold LUT per `docs/extracting-a-lut.md`, into `luts/private/kodak_gold_200.png`.
2. Pick 8–10 photos you already have DxO finals for, spanning daylight and flash.
3. Run each through the new pipeline at `grade_strength: 1.0`, `exposure_bias: 0`.
4. Put the new output, the DxO final, and the old pipeline's output side by side.

Run:
```bash
python app.py
```
and process them through the UI, or script `film.process_photo` directly.

**Pass:** the new output is *closer to the DxO final* than the old pipeline's was. Skin tones and neutrals especially — check a grey card or white wall for a cast.

**Fail:** if it is not closer, **stop here.** Do not tune sliders to compensate. A cast on neutrals means the LUT is contaminated (something was left on during extraction — go back to step 2 of the extraction doc). A tone mismatch means the grey-point scale is wrong for your camera. Fix the input, not the output.

- [ ] **Step 9: Commit**

```bash
git add film.py tests luts docs .gitignore
git commit -m "Replace the hand-tuned colour maths with a real 3D LUT

apply_film_color was per-channel gamma powers, a Gaussian midtone-warmth mask,
a shadow crossover term and an orange saturation hack. It was a guess. It is
gone, and a measured LUT carries the colour instead.

Exposure is now a multiply in linear light, so it preserves channel ratios and
therefore hue — the old additive offset in gamma space was a black-level lift
whose effective gain varied per channel. Contrast moves after the LUT and out
of every preset: the CLUTs were authored against a neutral, standard-contrast
render, so look-contrast before them is counted twice. Grain is last.

The provenance comment goes with it. Five of the seven preset parameters were
never derivable from the statistics they claimed, and the two that were have an
unexcluded explanation in crop bias."
```

---

## Self-Review

**Spec coverage.** Loader with state tag → Task 5. Multiplicative exposure → Task 8. Additive per-channel halation → Task 6. Neutral render + rolloff → Tasks 2, 8. sRGB encode + clip → Tasks 1, 8. Tetrahedral LUT → Tasks 3, 4. Monochrome midtone-peaked seeded grain → Task 7. Contrast after LUT → Task 8. Presets with neutral exposure/contrast, provenance deleted → Task 8. `luts/open` + `luts/private` + extraction doc → Task 8. **Batch → deliberately deferred to its own plan** (see the scope note at the top). **The calibration constant** (measuring rawpy vs. a DxO-neutral render, spec's "one honest number") is **not** in this plan — it is an optimisation of the grey-point scale in Task 8, and it should only be attempted if the Step 8 gate reveals a systematic tone offset. Recorded here so it is not silently dropped.

**Type consistency.** `load_image` returns `(np.ndarray, str)` and is consumed that way in Task 8. `apply_lut(rgb, cube, strength)` matches its call site. `add_halation(linear_rgb, intensity, radius)` and `add_grain(rgb, intensity, size, seed)` match theirs. `identity_cube(size)` takes cube size `S`, not Hald level `N` — `load_hald` and `make_hald.py` both convert with `size = level ** 2`, and this is the single easiest thing in the plan to get backwards.

**Risk resolved.** The spec listed "the default LUT may not exist" as the top open risk. It does exist — see Task 9, which was added after verifying spektrafilm's repository directly. `get_lut` still degrades to an identity cube when a LUT is missing, so the app runs with none installed.

---

### Task 9: Bake and ship the default LUT

**Files:**
- Create: `luts/open/kodak_gold_200.png` (binary, CC BY-SA 4.0)
- Create: `luts/open/LICENSE`, `luts/open/ATTRIBUTION.md`
- Create: `docs/baking-the-default-lut.md`
- Modify: `README.md` (licence carve-out)

**Interfaces:**
- Consumes: `filmlab.lut.load_hald` (Task 3)
- Produces: a real film LUT at the default name, so `process_photo` does something out of the box

**Why this is possible at all.** `spektrafilm` (formerly agx-emulsion) ships `kodak_gold_200.json` — a measured profile: 81 wavelengths of spectral sensitivity, per-layer base density, 256-point density curves, Status M densitometry, D55 reference. It has a first-class HaldCLUT exporter, and **sRGB→sRGB is its supported path** (its scene-linear input spaces are deliberately disabled, because a uniform [0,1] cube domain cannot represent scene-linear highlights past diffuse white — the same reason this plan does not let the LUT be the tone map). It is exactly the tool for this.

**The licence, stated plainly, because getting this wrong is the expensive mistake.** spektrafilm's code is GPLv3, but its profiles are CC BY-SA 4.0, and `SPEKTRAFILM_LICENSE.txt` pre-answers our exact question:

> "This license applies to the original spektrafilm profiles … **and to all direct derivatives of the profiles, such as copies in other projects, LUTs, or any other format that encodes the same content.** LUTs and similar artifacts are interpreted as direct encodings of the information in the original profiles."

Consequences, all of which the tasks below honour:
- The baked PNG **is CC BY-SA 4.0**. It cannot be relicensed MIT. It ships as a per-directory carve-out with attribution — normal, permissible, and it must be *said*.
- **The bake script does not live in this repo.** A script that does `import spektrafilm` is a GPL derivative, and this repo is MIT. The command is *documented* instead, so anyone can reproduce the asset without this repo carrying GPL code.
- Images produced *by applying* the LUT are yours. No copyleft on your photographs.

- [ ] **Step 1: Install spektrafilm in a throwaway environment**

Outside this repo — the point is that spektrafilm never becomes a dependency of it.

```bash
python -m venv /tmp/spektra && /tmp/spektra/Scripts/pip install spektrafilm
```

- [ ] **Step 2: Bake the Hald PNG**

Three gotchas, each of which will otherwise cost an hour:
- The CLI has **no `--format` flag**. Hald PNG is reachable only through the Python API.
- The CLI's default resolution is **33, which is not a perfect square**, so the Hald writer raises `ValueError`. Use 64.
- A bundle **requires a print profile**. Gold 200's own `target_print` field names `kodak_portra_endura`.

Save as `/tmp/bake.py` (NOT in this repo — it imports a GPL library):

```python
from pathlib import Path

from spektrafilm_lut_creator import BundleBuilder
from spektrafilm_lut_creator.formats import get_format

bundle = BundleBuilder(
    film="kodak_gold_200",
    print="kodak_portra_endura",   # from the profile's own target_print field
    input_space="srgb",            # display-referred in...
    output_space="srgb",           # ...display-referred out
    resolution=64,                 # a perfect square: level 8, 64^3 cube
    topology="1lut",
).build()

get_format("hald_png").write(bundle.lut, Path("kodak_gold_200.png"))
print("baked kodak_gold_200.png")
```

Run: `/tmp/spektra/Scripts/python /tmp/bake.py`

- [ ] **Step 3: Verify the LUT before trusting it**

A contaminated LUT looks "filmic" enough that you will not notice for weeks. Check the neutral axis, which is the property tetrahedral interpolation is *supposed* to preserve — if the LUT itself tints greys, no amount of correct interpolation saves it.

```bash
python -c "
from pathlib import Path
import numpy as np
from filmlab import lut

cube = lut.load_hald(Path('luts/open/kodak_gold_200.png'))
print('shape', cube.shape)

# The LUT must not be an identity — it should actually do something.
ident = lut.identity_cube(cube.shape[0])
print('mean deviation from identity: %.4f' % float(np.abs(cube - ident).mean()))

# Neutral axis: how far does a grey drift off-neutral?
d = np.array([cube[i, i, i] for i in range(cube.shape[0])])
spread = (d.max(axis=1) - d.min(axis=1))
print('max neutral drift: %.4f' % float(spread.max()))
"
```

Expected: shape `(64, 64, 64, 3)`; mean deviation comfortably above `0.01` (it is not an identity); max neutral drift small. A *film* LUT legitimately warms neutrals, so some drift is real — but if it is large and one-sided, the bake picked up a cast.

- [ ] **Step 4: Write the attribution, because the licence requires it**

Create `luts/open/LICENSE` containing the full CC BY-SA 4.0 text (fetch from <https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt>).

Create `luts/open/ATTRIBUTION.md`:

```markdown
# Attribution

## kodak_gold_200.png

Baked from the **spektrafilm** film profile `kodak_gold_200`, a spectral
photochemical simulation built from measured film datasheet density curves.

- **Author:** Andrea Volpato
- **Source:** <https://github.com/andreavolpato/spektrafilm>
- **Licence:** [CC BY-SA 4.0](LICENSE)
- **Modified:** yes — baked to a level-8 HaldCLUT PNG (sRGB in, sRGB out,
  print profile `kodak_portra_endura`) by the Film Lab project. Not modified
  otherwise.

spektrafilm's licence states that LUTs are direct encodings of its profiles and
so carry the profile licence. This file is therefore **CC BY-SA 4.0**, not MIT,
and share-alike applies to it. The rest of this repository is MIT. Photographs
you produce by applying this LUT are yours, with no copyleft attached.

To reproduce this file, see [../../docs/baking-the-default-lut.md](../../docs/baking-the-default-lut.md).
```

- [ ] **Step 5: Document the bake, since the script cannot ship here**

Create `docs/baking-the-default-lut.md` containing the explanation from this task: why the script is not in the repo (it imports GPL code, this repo is MIT), the three gotchas, the `bake.py` source from Step 2, and the verification from Step 3.

- [ ] **Step 6: State the carve-out in the README**

Replace the README's licence section:

```markdown
## License

Code is [MIT](LICENSE).

**One exception:** `luts/open/kodak_gold_200.png` is baked from
[spektrafilm](https://github.com/andreavolpato/spektrafilm)'s measured film
profile and is **[CC BY-SA 4.0](luts/open/LICENSE)**, not MIT — spektrafilm's
licence treats a LUT as a direct encoding of its profile data, so share-alike
follows it. See [luts/open/ATTRIBUTION.md](luts/open/ATTRIBUTION.md).

Photographs you make with it are yours. No copyleft reaches your images.

Any LUTs you add yourself carry their own licences — check them.

Film stock names are trademarks of their respective owners. Nothing here is
affiliated with, endorsed by, or derived from Kodak, Fujifilm, or DxO.
```

- [ ] **Step 7: Confirm the pipeline renders with it**

Run: `python -m unittest discover -s tests` → OK

Then process a real photo and look at it. This is the same gate as Task 8 Step 8; if the LUT is installed before that gate, run them together.

- [ ] **Step 8: Commit**

```bash
git add luts docs/baking-the-default-lut.md README.md
git commit -m "Ship a real Kodak Gold 200 LUT, baked from spektrafilm

spektrafilm carries a measured kodak_gold_200 profile — spectral sensitivities,
per-layer density curves, Status M densitometry — and a HaldCLUT exporter whose
supported path is sRGB in, sRGB out, which is exactly what this pipeline needs.
So the default look is now physically derived rather than guessed, and no
extraction from commercial software is required to use this project.

The LUT is CC BY-SA 4.0, not MIT: spektrafilm's licence explicitly treats a LUT
as a direct encoding of its profile data. It ships as a per-directory carve-out
with attribution. The bake script is NOT in this repo — it imports GPL code and
this repo is MIT — so the command is documented instead."
```

**Note on exposure mapping, the one knob you will actually have to dial.** Feeding an already-rendered sRGB image into a film model forces a decision about what encoded 1.0 means in scene-exposure terms. spektrafilm exposes this as `stops_above_midgray`, defaulting to 4.0, and is candid that "this is an aesthetic interpretation, not a measurement" — the film's native headroom is ~2.47 stops, and +4 is chosen so encoded 1.0 lands on the shoulder and the rolloff engages. Practical effect: **the LUT will render slightly brighter than the source** unless tuned. If the Task 8 gate shows a systematic brightness offset, this is the first thing to adjust — not the grey-point scale, and definitely not the sliders.
