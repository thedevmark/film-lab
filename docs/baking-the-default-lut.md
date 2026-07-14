# Baking the default LUT

`luts/open/kodak_gold_200.png` is generated from
[spektrafilm](https://github.com/andreavolpato/spektrafilm), a spectral
photochemical film simulation built from measured datasheet density curves. This
is how to reproduce it.

## Why the script isn't in this repo

It imports `spektrafilm`, which is **GPLv3**. This repo is **MIT**. Shipping a
script that imports GPL code would make that script a GPL derivative and muddy
the licensing of a repository whose whole point is to be simple to reuse. So the
command lives here, in prose, and you run it in a throwaway environment.

The *output* is a different matter: the baked LUT is CC BY-SA 4.0 and ships in
`luts/open/` as a per-directory carve-out. See
[../luts/open/ATTRIBUTION.md](../luts/open/ATTRIBUTION.md).

## 1. Install spektrafilm somewhere disposable

It is not on PyPI, and it pulls in napari, Qt, scipy, and numba — none of which
this project wants anywhere near its dependency set.

```bash
python -m venv /tmp/spektra
/tmp/spektra/bin/pip install "git+https://github.com/andreavolpato/spektrafilm.git"
```

(On Windows: `/tmp/spektra/Scripts/pip.exe`.)

## 2. Bake

Save as `/tmp/bake_gold.py` — **not** inside this repo:

```python
from pathlib import Path

from spektrafilm_lut_creator.builders import BundleBuilder, BundleSpec
from spektrafilm_lut_creator.formats import get_format

spec = BundleSpec(
    film_profile="kodak_gold_200",
    print_profiles=("kodak_portra_endura",),  # the profile's own target_print field
    input_color_space="srgb",                 # display-referred in...
    output_color_space="srgb",                # ...display-referred out
    resolution=64,                            # level 8, 64^3 cube
    topology="1lut",
    stops_above_midgray=2.47,                 # see below — this one matters
)

bundle = BundleBuilder(spec).build()
name, lut = bundle.luts[0]                    # luts is a list of (filename, Lut)

out = Path("kodak_gold_200.png")
get_format("hald_png").write(lut, out)
print("wrote", out)
```

Run it, and move the PNG to `luts/open/kodak_gold_200.png`.

## The four things that will waste your afternoon

**`stops_above_midgray` defaults to 4.0 and will brighten every photo by over a
stop.** This is the one that matters. spektrafilm's default (`"auto"`) resolves
to 4.0, chosen so that encoded 1.0 lands on the film's shoulder and the rolloff
engages on SDR sources — its docs call this "an aesthetic interpretation, not a
measurement." Measured through the baked LUT:

| `stops_above_midgray` | mid-grey 0.500 becomes |
| --- | --- |
| `"auto"` (= 4.0) | **0.779** |
| 3.5 | 0.719 |
| 3.0 | 0.637 |
| **2.47** | **0.506** |

2.47 is `log2(1.0 / 0.18)` — sRGB's actual headroom above mid-grey, and the
physically honest value for display-referred input. Film Lab does its own
exposure and highlight rolloff *before* the LUT, so the LUT's job is colour, not
exposure. Use 2.47.

**`resolution` defaults to 33, which is not a perfect square**, so the Hald
writer raises `ValueError`. A Hald of level *N* is an *N*³×*N*³ image holding an
*N*²-per-axis cube, so the resolution must be a perfect square: 16, 25, 36, 49,
**64**. Use 64 — that is level 8, the size every HaldCLUT consumer expects.

**There is no `--format` flag on the CLI.** `spektrafilm-lut build` writes
`.cube`. Hald PNG is reachable only through the Python API, via
`get_format("hald_png")`.

**A bundle requires a print profile.** Film alone is a negative; you need the
paper it gets printed onto. Gold 200's own profile names
`kodak_portra_endura` in its `target_print` field — use that.

## 3. Check it before trusting it

A wrong LUT still looks "filmic". Verify it numerically:

```bash
python -c "
from pathlib import Path
import numpy as np
from filmlab import lut

cube = lut.load_hald(Path('luts/open/kodak_gold_200.png'))
n = cube.shape[0]
print('shape        ', cube.shape)
print('mid-grey     ', cube[n//2, n//2, n//2])   # expect ~0.51 — NOT ~0.78
print('black        ', cube[0, 0, 0])            # expect ~0.05 — film base fog
print('white        ', cube[-1, -1, -1])         # expect ~0.82 — the shoulder
print('vs identity  ', float(np.abs(cube - lut.identity_cube(n)).mean()))  # expect ~0.09
"
```

If mid-grey comes back near 0.78, `stops_above_midgray` is still at its default.
If the deviation from identity is ~0, the LUT is an identity and did not bake.
