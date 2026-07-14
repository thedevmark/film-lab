"""HaldCLUT loading and application.

A Hald image of level N encodes a cube of S = N**2 entries per axis, laid out as
an N**3 x N**3 image. Level 8 -> 512x512 -> 64**3. Pixels are read in raster
order into a flat table indexed  i = r + S*g + S**2 * b  (red fastest).

Pillow's "RGB" mode is hard-wired to 8 bits per channel: opening a 16-bit
colour PNG silently keeps only the high byte of every sample (see
`PIL.PngImagePlugin._MODES[(16, 2)] == ("RGB", "RGB;16B")` -- the unpacker
throws the low byte away while decoding into Pillow's 8-bit "RGB" buffer).
`Image.fromarray` has the matching problem in the other direction: its
type map has no entry for a 3-channel uint16 array, so it cannot even
construct a 16-bit RGB image to save. HaldCLUTs are authored at 16-bit
specifically to avoid quantising the table before it's interpolated, so
this module reads and writes 16-bit RGB PNGs itself (stdlib `struct` +
`zlib` only -- no new dependency) instead of going through Pillow for that
one case. 8-bit PNGs are unaffected and go through Pillow as normal.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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


# --- 16-bit PNG I/O -----------------------------------------------------
#
# Pillow can't round-trip this, so we speak PNG directly for the 16-bit
# case. Scoped deliberately narrow: RGB or RGBA, 8 or 16 bit samples,
# non-interlaced, no palette -- exactly what HaldCLUT PNGs are.

def _iter_png_chunks(data: bytes):
    pos = len(_PNG_SIGNATURE)
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        cdata = data[pos + 8:pos + 8 + length]
        yield ctype, cdata
        pos += 12 + length  # length(4) + type(4) + data + crc(4)


def _paeth_predictor(a, b, c):
    """PNG Paeth predictor (int32 arrays); a=left, b=above, c=above-left."""
    p = a + b - c
    pa = np.abs(p - a)
    pb = np.abs(p - b)
    pc = np.abs(p - c)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))


def _unfilter_scanlines(raw: bytes, width: int, height: int, bpp: int) -> np.ndarray:
    """Reverse PNG per-scanline filtering. Returns (height, width, bpp) int32."""
    stride = width * bpp
    prev = np.zeros((width, bpp), dtype=np.int32)
    out = np.empty((height, width, bpp), dtype=np.int32)
    pos = 0

    for y in range(height):
        ftype = raw[pos]
        row = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=pos + 1)
        row = row.astype(np.int32).reshape(width, bpp)
        pos += 1 + stride

        if ftype == 0:  # None
            recon = row
        elif ftype == 1:  # Sub: left neighbour, same scanline
            recon = np.cumsum(row, axis=0) % 256
        elif ftype == 2:  # Up: pixel directly above
            recon = (row + prev) % 256
        elif ftype in (3, 4):  # Average / Paeth: horizontal + vertical recurrence
            recon = np.empty_like(row)
            left = np.zeros(bpp, dtype=np.int32)
            zeros = np.zeros(bpp, dtype=np.int32)
            for x in range(width):
                above = prev[x]
                above_left = prev[x - 1] if x > 0 else zeros
                if ftype == 3:
                    predictor = (left + above) // 2
                else:
                    predictor = _paeth_predictor(left, above, above_left)
                recon[x] = (row[x] + predictor) % 256
                left = recon[x]
        else:
            raise ValueError(f"unsupported PNG filter type {ftype}")

        out[y] = recon
        prev = recon

    return out


def _png_bit_depth(path: Path):
    """Peek at a PNG's IHDR bit depth without decoding it. None if not a PNG."""
    with open(path, "rb") as f:
        header = f.read(29)
    if header[:8] != _PNG_SIGNATURE or header[12:16] != b"IHDR":
        return None
    return header[24]


def _read_png16_rgb(path: Path) -> np.ndarray:
    """Decode a 16-bit-per-channel RGB(A) PNG at full precision. Returns (H,W,3) uint16."""
    data = path.read_bytes()
    if data[:8] != _PNG_SIGNATURE:
        raise ValueError(f"{path.name}: not a PNG file")

    ihdr = None
    idat = bytearray()
    for ctype, cdata in _iter_png_chunks(data):
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filt, interlace = \
                struct.unpack(">IIBBBBB", cdata)
            ihdr = (width, height, bit_depth, color_type, interlace)
        elif ctype == b"IDAT":
            idat += cdata
        elif ctype == b"PLTE":
            raise ValueError(f"{path.name}: palette-based PNGs are not supported")

    if ihdr is None:
        raise ValueError(f"{path.name}: missing IHDR chunk")
    width, height, bit_depth, color_type, interlace = ihdr

    if interlace != 0:
        raise ValueError(f"{path.name}: interlaced PNGs are not supported")
    if color_type not in (2, 6):
        raise ValueError(f"{path.name}: expected an RGB(A) colour type, got {color_type}")

    channels = 3 if color_type == 2 else 4
    bpp = channels * 2  # 16-bit samples: 2 bytes per channel

    raw = zlib.decompress(bytes(idat))
    unfiltered = _unfilter_scanlines(raw, width, height, bpp)  # (H, W, channels*2)

    hi = unfiltered[:, :, 0::2].astype(np.uint16)
    lo = unfiltered[:, :, 1::2].astype(np.uint16)
    samples = (hi << 8) | lo
    return samples[:, :, :3]


def _write_png16_rgb(path: Path, arr: np.ndarray) -> None:
    """Write a 16-bit-per-channel RGB PNG from an (H, W, 3) uint16 array."""
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) array")
    height, width, _ = arr.shape

    be = arr.astype(">u2")
    raw = b"".join(b"\x00" + row.tobytes() for row in be)  # filter type 0 (None) per row
    compressed = zlib.compress(raw, 9)

    def chunk(tag: bytes, cdata: bytes) -> bytes:
        return struct.pack(">I", len(cdata)) + tag + cdata + struct.pack(">I", zlib.crc32(tag + cdata))

    ihdr = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(_PNG_SIGNATURE)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))


# --- Hald cube loading ---------------------------------------------------

def load_hald(path: Path) -> np.ndarray:
    """Load a HaldCLUT PNG into a (S,S,S,3) float32 cube indexed [r, g, b]."""
    path = Path(path)

    if _png_bit_depth(path) == 16:
        arr = _read_png16_rgb(path)
        scale = np.float32(65535.0)
    else:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"{path.name}: not an RGB image")
        arr = arr[:, :, :3]
        scale = np.float32(255.0)

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
    flat = arr.reshape(-1, 3).astype(np.float32) / scale

    if flat.shape[0] != size ** 3:
        raise ValueError(f"{path.name}: expected {size ** 3} entries, got {flat.shape[0]}")

    # Raster order is [b, g, r]; we want [r, g, b].
    return np.ascontiguousarray(flat.reshape(size, size, size, 3).transpose(2, 1, 0, 3))


# --- Tetrahedral interpolation -------------------------------------------

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
    #
    # cond_rg / cond_gb / cond_rb only reach six of the eight possible
    # (T/F, T/F, T/F) combinations: (T,T,F) and (F,F,T) are algebraically
    # impossible (dr>dg>db>dr is inconsistent, and its mirror). The six masks
    # below are the six reachable combinations, each attached to the formula
    # for its matching ordering:
    #
    #   (T,T,T) dr>dg>db | (T,F,T) dr>db>dg | (T,F,F) db>dr>dg
    #   (F,F,F) db>dg>dr | (F,T,F) dg>db>dr | (F,T,T) dg>dr>db
    #
    # Getting a mask's booleans out of order silently sends pixels through the
    # wrong formula (or lets two masks overlap, or leaves a gap) without
    # raising anything -- see test_lut.py's TestApplyLutSixTetrahedra for the
    # verification that these six partition every pixel exactly once.
    out = np.empty(rgb.shape, dtype=np.float32)

    w = lambda x: x[..., None]  # noqa: E731 - broadcast a weight over RGB

    cond_rg = dr > dg
    cond_gb = dg > db
    cond_rb = dr > db

    # dr > dg > db  ->  (T,T,T)
    m = cond_rg & cond_gb
    out = np.where(w(m),
                   w(1 - dr) * c000 + w(dr - dg) * node(1, 0, 0)
                   + w(dg - db) * node(1, 1, 0) + w(db) * c111,
                   0.0).astype(np.float32)

    # dr > db > dg  ->  (T,F,T)
    m = cond_rg & ~cond_gb & cond_rb
    out += np.where(w(m),
                    w(1 - dr) * c000 + w(dr - db) * node(1, 0, 0)
                    + w(db - dg) * node(1, 0, 1) + w(dg) * c111,
                    0.0).astype(np.float32)

    # db > dr > dg  ->  (T,F,F)
    m = cond_rg & ~cond_gb & ~cond_rb
    out += np.where(w(m),
                    w(1 - db) * c000 + w(db - dr) * node(0, 0, 1)
                    + w(dr - dg) * node(1, 0, 1) + w(dg) * c111,
                    0.0).astype(np.float32)

    # db > dg > dr  ->  (F,F,F)  (also where dr == dg == db lands: all three
    # comparisons are False when the deltas are equal, and this formula then
    # collapses to (1-d)*c000 + d*c111 -- the neutral-axis guarantee.)
    m = ~cond_rg & ~cond_gb & ~cond_rb
    out += np.where(w(m),
                    w(1 - db) * c000 + w(db - dg) * node(0, 0, 1)
                    + w(dg - dr) * node(0, 1, 1) + w(dr) * c111,
                    0.0).astype(np.float32)

    # dg > db > dr  ->  (F,T,F)
    m = ~cond_rg & cond_gb & ~cond_rb
    out += np.where(w(m),
                    w(1 - dg) * c000 + w(dg - db) * node(0, 1, 0)
                    + w(db - dr) * node(0, 1, 1) + w(dr) * c111,
                    0.0).astype(np.float32)

    # dg > dr > db  ->  (F,T,T)
    m = ~cond_rg & cond_gb & cond_rb
    out += np.where(w(m),
                    w(1 - dg) * c000 + w(dg - dr) * node(0, 1, 0)
                    + w(dr - db) * node(1, 1, 0) + w(db) * c111,
                    0.0).astype(np.float32)

    if strength >= 1.0:
        return out
    s = np.float32(strength)
    return (s * out + (np.float32(1.0) - s) * rgb).astype(np.float32)
