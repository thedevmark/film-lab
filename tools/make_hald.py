"""Write an identity HaldCLUT PNG.

Render this image through a colour tool and the result IS that tool's colour
transform, captured as a LUT. See docs/extracting-a-lut.md.

    python tools/make_hald.py --level 8 --out identity_hald_8.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# `python tools/make_hald.py` (as invoked above) puts this script's own
# directory on sys.path, not the repo root -- so the sibling `filmlab`
# package isn't importable without this. `python -m tools.make_hald` doesn't
# need it (that form puts the cwd on sys.path instead), but the plain form is
# what the CLI is documented to use above.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filmlab.lut import _write_png16_rgb, identity_cube  # noqa: E402


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

    # Not Image.fromarray(img).save(...): Pillow's fromarray has no type-map
    # entry for a 3-channel uint16 array (only single-channel 16-bit
    # grayscale), so it can't construct a 16-bit RGB image at all. See
    # filmlab/lut.py's module docstring for the matching read-side problem.
    _write_png16_rgb(args.out, img)
    print(f"wrote {args.out}  ({side}x{side}, level {args.level}, {size}^3 cube)")


if __name__ == "__main__":
    main()
