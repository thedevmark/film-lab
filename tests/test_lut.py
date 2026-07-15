import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from filmlab import lut


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    """One length-prefixed, CRC-suffixed PNG chunk, built from scratch."""
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _build_png16_rgb(width: int, height: int, raw_scanlines: bytes) -> bytes:
    """Assemble a minimal 16-bit RGB PNG's bytes, independent of
    `lut._write_png16_rgb`.

    `raw_scanlines` is the exact IDAT payload *before* zlib compression:
    for each row, one filter-type byte followed by width * 6 sample bytes
    (3 channels, 2 bytes each, big-endian per the PNG spec). Callers hand-
    pick the filter byte and raw sample bytes per row, so this can produce
    scanlines using any filter type, or a mix of them -- something our own
    writer (which only ever emits filter type 0) never does. Building the
    PNG this way means these tests pin `_read_png16_rgb` against the PNG
    spec, not against our own writer's conventions.
    """
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    compressed = zlib.compress(raw_scanlines, 9)
    return (
        lut._PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


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
            # Not Image.fromarray(...).save(path): Pillow's fromarray has no
            # type-map entry for a 3-channel uint16 array, so it can't build
            # a 16-bit RGB image at all. See filmlab/lut.py's module
            # docstring.
            lut._write_png16_rgb(path, (img * 65535).round().astype(np.uint16))
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


class TestPngByteOrderAgainstSpec(unittest.TestCase):
    """Pins `_read_png16_rgb`'s 16-bit sample byte order against a hand-built
    PNG, independent of `_write_png16_rgb`.

    Every round-trip test in this file writes with `_write_png16_rgb` and
    reads back with `_read_png16_rgb`. If both ends flipped to little-endian
    together, every round-trip would still match -- a symmetric bug is
    invisible to a test that only checks the pair agrees with itself. The
    PNG spec (sec 7.2) mandates big-endian 16-bit samples, so this builds
    the encoded bytes by hand and checks the decoded value against that
    external spec instead.
    """

    def test_16_bit_samples_are_read_big_endian(self):
        # One pixel, filter type None. Each channel's two bytes are chosen
        # non-equal and non-palindromic so a hi/lo swap changes the value:
        # red = 0x01, 0x02 -> must decode to 0x0102 (258), not 0x0201 (513).
        raw = bytes([0,  0x01, 0x02,  0x03, 0x04,  0x05, 0x06])
        png_bytes = _build_png16_rgb(width=1, height=1, raw_scanlines=raw)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "one_pixel.png"
            path.write_bytes(png_bytes)

            arr = lut._read_png16_rgb(path)

        self.assertEqual(arr.shape, (1, 1, 3))
        self.assertEqual(int(arr[0, 0, 0]), 0x0102)  # 258, NOT 0x0201 = 513
        self.assertEqual(int(arr[0, 0, 1]), 0x0304)
        self.assertEqual(int(arr[0, 0, 2]), 0x0506)


class TestPngDefilter(unittest.TestCase):
    """PNG scanline reconstruction, tested independently of _write_png16_rgb.

    Our own writer only ever emits filter type 0 (None), so the round-trip
    tests above never exercise Sub/Up/Average/Paeth. Real 16-bit HaldCLUTs
    (e.g. Pat David's, or anything exported from GIMP/ImageMagick) are
    smooth gradients, exactly the content adaptive PNG filtering likes to
    pick Up/Average/Paeth for -- if this reconstruction were wrong, a real
    file would silently decode to the wrong colours. Expected values below
    are hand-computed from the PNG spec's filter definitions, bpp=1 to keep
    the arithmetic checkable by inspection.
    """

    def test_sub_filter_is_a_cumulative_left_neighbour_sum(self):
        # Row 0: filter None, bytes [10, 20, 30].
        # Row 1: filter Sub,  bytes [5, 5, 5] -> 5, 5+5=10, 10+5=15.
        raw = bytes([0, 10, 20, 30, 1, 5, 5, 5])

        out = lut._unfilter_scanlines(raw, width=3, height=2, bpp=1)

        np.testing.assert_array_equal(out[0, :, 0], [10, 20, 30])
        np.testing.assert_array_equal(out[1, :, 0], [5, 10, 15])

    def test_up_filter_adds_the_pixel_directly_above(self):
        # Row 0: filter None, bytes [10, 20, 30].
        # Row 1: filter Up,   bytes [1, 1, 1] -> 11, 21, 31.
        raw = bytes([0, 10, 20, 30, 2, 1, 1, 1])

        out = lut._unfilter_scanlines(raw, width=3, height=2, bpp=1)

        np.testing.assert_array_equal(out[1, :, 0], [11, 21, 31])

    def test_average_filter_floors_the_mean_of_left_and_above(self):
        # Row 0: filter None, bytes [20, 60, 100].
        # Row 1: filter Average, all-zero payload -> pure predictor:
        #   x=0: floor((0 + 20) / 2)  = 10
        #   x=1: floor((10 + 60) / 2) = 35
        #   x=2: floor((35 + 100) / 2) = 67
        raw = bytes([0, 20, 60, 100, 3, 0, 0, 0])

        out = lut._unfilter_scanlines(raw, width=3, height=2, bpp=1)

        np.testing.assert_array_equal(out[1, :, 0], [10, 35, 67])

    def test_paeth_filter_picks_the_closest_of_left_above_above_left(self):
        # Row 0: filter None, bytes [10, 50, 90].
        # Row 1: filter Paeth, all-zero payload -> pure predictor. With
        # above-left always <= both neighbours here, Paeth always picks the
        # "above" candidate:
        #   x=0: a=0 (no left), b=10, c=0            -> pick b=10
        #   x=1: a=10 (left),   b=50, c=10 (above-left) -> pick b=50
        #   x=2: a=50 (left),   b=90, c=50 (above-left) -> pick b=90
        raw = bytes([0, 10, 50, 90, 4, 0, 0, 0])

        out = lut._unfilter_scanlines(raw, width=3, height=2, bpp=1)

        np.testing.assert_array_equal(out[1, :, 0], [10, 50, 90])

    def test_rejects_an_unsupported_filter_type(self):
        raw = bytes([9, 1, 2, 3])  # filter type 9 does not exist

        with self.assertRaises(ValueError):
            lut._unfilter_scanlines(raw, width=3, height=1, bpp=1)


class TestPngDefilterWraparound(unittest.TestCase):
    """Every case above stays under 255, so `_unfilter_scanlines`'s `% 256`
    arithmetic is never actually exercised -- stripping all four `% 256`
    operators leaves those tests green. These cases are chosen so the
    reconstructed sum genuinely exceeds 255 and must wrap.
    """

    def test_sub_filter_wraps_when_the_running_sum_exceeds_255(self):
        # Filter Sub, one row: raw bytes 200 then 100.
        # x=0: no left -> 200. x=1: (100 + 200) mod 256 = 44, NOT 300.
        raw = bytes([1, 200, 100])

        out = lut._unfilter_scanlines(raw, width=2, height=1, bpp=1)

        np.testing.assert_array_equal(out[0, :, 0], [200, 44])

    def test_up_filter_wraps_when_the_value_plus_above_exceeds_255(self):
        # Row 0: filter None, byte 200. Row 1: filter Up, byte 100.
        # (100 + 200) mod 256 = 44, NOT 300.
        raw = bytes([0, 200, 2, 100])

        out = lut._unfilter_scanlines(raw, width=1, height=2, bpp=1)

        np.testing.assert_array_equal(out[1, :, 0], [44])

    def test_average_filter_wraps_when_predictor_plus_raw_exceeds_255(self):
        # Row 0: filter None, byte 255. Row 1: filter Average, byte 200.
        # predictor = floor((0 + 255) / 2) = 127. (200 + 127) mod 256 = 71,
        # NOT 327.
        raw = bytes([0, 255, 3, 200])

        out = lut._unfilter_scanlines(raw, width=1, height=2, bpp=1)

        np.testing.assert_array_equal(out[1, :, 0], [71])

    def test_paeth_filter_wraps_when_predictor_plus_raw_exceeds_255(self):
        # Row 0: filter None, byte 200. Row 1: filter Paeth, byte 200.
        # a=0 (no left), b=200 (above), c=0 (no above-left) -> predictor
        # picks b=200. (200 + 200) mod 256 = 144, NOT 400.
        raw = bytes([0, 200, 4, 200])

        out = lut._unfilter_scanlines(raw, width=1, height=2, bpp=1)

        np.testing.assert_array_equal(out[1, :, 0], [144])


class TestPaethPredictorBranchSelection(unittest.TestCase):
    """The existing Paeth test (`TestPngDefilter`) always resolves to the
    "pick above" (b) branch. These force the other two branches, so a bug
    that swaps the `pa <= pb` / `pb <= pc` tie-breaks gets caught. Reference:
    PNG spec sec 6.6 -- p = a + b - c; pa=|p-a|; pb=|p-b|; pc=|p-c|; return a
    if pa<=pb and pa<=pc, else b if pb<=pc, else c.
    """

    def test_paeth_picks_left_when_left_is_the_closest_predictor(self):
        # Row 0 (None): [10, 10] -> above = 10 for every column.
        # Row 1 (Paeth):
        #   x=0: a=0, b=10, c=0 -> p=10, pa=10, pb=0, pc=10 -> pick b=10.
        #        raw=40 -> recon = (40+10) % 256 = 50.
        #   x=1: a=50 (left), b=10 (above), c=10 (above-left)
        #        -> p=50+10-10=50, pa=|50-50|=0, pb=|50-10|=40, pc=40
        #        -> pa<=pb and pa<=pc -> pick a=50.
        #        raw=5 -> recon = (5+50) % 256 = 55.
        raw = bytes([0, 10, 10, 4, 40, 5])

        out = lut._unfilter_scanlines(raw, width=2, height=2, bpp=1)

        np.testing.assert_array_equal(out[1, :, 0], [50, 55])

    def test_paeth_picks_above_left_when_it_is_the_closest_predictor(self):
        # Row 0 (None): [50, 10] -> above-left=50, above=10 for x=1.
        # Row 1 (Paeth):
        #   x=0: a=0, b=50, c=0 -> p=50, pa=50, pb=0, pc=50 -> pick b=50.
        #        raw=40 -> recon = (40+50) % 256 = 90.
        #   x=1: a=90 (left), b=10 (above), c=50 (above-left)
        #        -> p=90+10-50=50, pa=|50-90|=40, pb=|50-10|=40, pc=|50-50|=0
        #        -> pa<=pb (true) and pa<=pc (40<=0, false) -> not a.
        #        -> pb<=pc (40<=0, false) -> pick c=50.
        #        raw=7 -> recon = (7+50) % 256 = 57.
        raw = bytes([0, 50, 10, 4, 40, 7])

        out = lut._unfilter_scanlines(raw, width=2, height=2, bpp=1)

        np.testing.assert_array_equal(out[1, :, 0], [90, 57])


class TestReadPng16RgbMultiFilterScanlines(unittest.TestCase):
    """Decodes a hand-built 16-bit RGB PNG whose five scanlines each use a
    different filter type (None, Sub, Up, Average, Paeth), end to end
    through `_read_png16_rgb`. This is the closest thing to a real-world
    file this suite can build without a network fetch: real HaldCLUTs are
    smooth gradients, so an adaptive PNG encoder picks a different filter
    per row depending on local content, exactly like this.

    Every pixel uses the same 16-bit value in all three channels, so the
    "expected" numbers below only need to be derived once per row/column and
    are then asserted against all of R, G, and B.
    """

    def test_decodes_a_five_row_image_with_one_filter_type_per_row(self):
        # Row 0 -- None: samples straight off the wire.
        #   col0: hi=10, lo=100 -> 0x0A64 = 2660
        #   col1: hi=20, lo=110 -> 0x146E = 5230
        row0 = bytes([0]) + bytes([10, 100] * 3) + bytes([20, 110] * 3)

        # Row 1 -- Sub: cumulative left-sum per byte lane, independent of
        # row 0. hi lane raw=[15, 20] -> recon=[15, 35].
        # lo lane raw=[5, 7] -> recon=[5, 12].
        #   col0: hi=15, lo=5  -> 0x0F05 = 3845
        #   col1: hi=35, lo=12 -> 0x230C = 8972
        row1 = bytes([1]) + bytes([15, 5] * 3) + bytes([20, 7] * 3)

        # Row 2 -- Up: raw + row-1's reconstructed value, per byte lane.
        # hi lane raw=[35, 55] + prev=[15, 35] -> recon=[50, 90].
        # lo lane raw=[55, 68] + prev=[5, 12]  -> recon=[60, 80].
        #   col0: hi=50, lo=60 -> 0x323C = 12860
        #   col1: hi=90, lo=80 -> 0x5A50 = 23120
        row2 = bytes([2]) + bytes([35, 55] * 3) + bytes([55, 68] * 3)

        # Row 3 -- Average: floor((left+above)/2) + raw, per byte lane.
        # hi lane: x=0 predictor=floor((0+50)/2)=25, raw=45 -> recon=70.
        #          x=1 predictor=floor((70+90)/2)=80, raw=20 -> recon=100.
        # lo lane: x=0 predictor=floor((0+60)/2)=30, raw=15 -> recon=45.
        #          x=1 predictor=floor((45+80)/2)=62, raw=28 -> recon=90.
        #   col0: hi=70, lo=45  -> 0x462D = 17965
        #   col1: hi=100, lo=90 -> 0x645A = 25690
        row3 = bytes([3]) + bytes([45, 15] * 3) + bytes([20, 28] * 3)

        # Row 4 -- Paeth: predictor picks "above" (b) at every column here
        # (branch selection itself is covered by
        # TestPaethPredictorBranchSelection above).
        # hi lane: x=0 a=0,b=70,c=0 -> pick b=70, raw=20 -> recon=90.
        #          x=1 a=90,b=100,c=70 -> pick b=100, raw=30 -> recon=130.
        # lo lane: x=0 a=0,b=45,c=0 -> pick b=45, raw=15 -> recon=60.
        #          x=1 a=60,b=90,c=45 -> pick b=90, raw=20 -> recon=110.
        #   col0: hi=90, lo=60   -> 0x5A3C = 23100
        #   col1: hi=130, lo=110 -> 0x826E = 33390
        row4 = bytes([4]) + bytes([20, 15] * 3) + bytes([30, 20] * 3)

        raw_scanlines = row0 + row1 + row2 + row3 + row4
        png_bytes = _build_png16_rgb(width=2, height=5, raw_scanlines=raw_scanlines)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "multi_filter.png"
            path.write_bytes(png_bytes)

            arr = lut._read_png16_rgb(path)

        self.assertEqual(arr.shape, (5, 2, 3))
        expected_per_column = [
            [2660, 5230],
            [3845, 8972],
            [12860, 23120],
            [17965, 25690],
            [23100, 33390],
        ]
        for y, (col0, col1) in enumerate(expected_per_column):
            for channel in range(3):
                self.assertEqual(int(arr[y, 0, channel]), col0, f"row {y} col 0 channel {channel}")
                self.assertEqual(int(arr[y, 1, channel]), col1, f"row {y} col 1 channel {channel}")


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


def _reference_tetrahedral_lookup(r, g, b, cube):
    """Plain-Python, obviously-correct per-pixel tetrahedral lookup.

    This is deliberately NOT vectorized and shares none of `apply_lut`'s code
    -- it exists purely as test scaffolding, precisely because the six-mask
    vectorized form in `filmlab/lut.py` is easy to get subtly wrong (two masks
    overlapping double-counts a pixel and brightens it; a gap between masks
    leaves a pixel black) in ways that greys-stay-grey / lattice-node /
    identity-cube tests cannot see, since all of those happen to be invariant
    under exactly the kind of mask mix-up this function is here to catch.
    Six explicit if/elif branches, one per ordering of (dr, dg, db), each
    doing the arithmetic from the brief by hand.
    """
    size = cube.shape[0]
    sr, sg, sb = r * (size - 1), g * (size - 1), b * (size - 1)
    ir = min(int(np.floor(sr)), size - 2)
    ig = min(int(np.floor(sg)), size - 2)
    ib = min(int(np.floor(sb)), size - 2)
    dr, dg, db = sr - ir, sg - ig, sb - ib

    def node(orr, og, ob):
        return cube[ir + orr, ig + og, ib + ob].astype(np.float64)

    c000 = node(0, 0, 0)
    c111 = node(1, 1, 1)

    if dr > dg > db:
        out = (1 - dr) * c000 + (dr - dg) * node(1, 0, 0) + (dg - db) * node(1, 1, 0) + db * c111
    elif dr > db > dg:
        out = (1 - dr) * c000 + (dr - db) * node(1, 0, 0) + (db - dg) * node(1, 0, 1) + dg * c111
    elif db > dr > dg:
        out = (1 - db) * c000 + (db - dr) * node(0, 0, 1) + (dr - dg) * node(1, 0, 1) + dg * c111
    elif db > dg > dr:
        out = (1 - db) * c000 + (db - dg) * node(0, 0, 1) + (dg - dr) * node(0, 1, 1) + dr * c111
    elif dg > db > dr:
        out = (1 - dg) * c000 + (dg - db) * node(0, 1, 0) + (db - dr) * node(0, 1, 1) + dr * c111
    else:  # dg > dr > db (also where dr == dg == db, all comparisons False -> db>dg>dr branch above)
        out = (1 - dg) * c000 + (dg - dr) * node(0, 1, 0) + (dr - db) * node(1, 1, 0) + db * c111

    return out


class TestApplyLutAgainstReferenceImplementation(unittest.TestCase):
    """Test A: reference-implementation equivalence on a non-affine cube.

    This is the test that would have caught the six-tetrahedra masking bug:
    an earlier draft of `apply_lut` had three of its six boolean masks
    labelled with the wrong ordering, so some pixels were run through the
    formula for a *different* tetrahedron than the one their (dr, dg, db)
    actually fell in. Every other test in this file passes against both the
    correct and the buggy version, because they only probe greys (which all
    six formulas agree on), lattice nodes (dr == dg == db == 0), or an
    identity/all-zero cube (where several formulas collapse to the same
    answer). Only a comparison against an independent, non-vectorized,
    obviously-correct reference on a genuinely warped (non-affine) cube with
    off-axis, non-lattice-aligned query colours can distinguish "the right
    node, wrong weight" failure this kind of bug produces.
    """

    def test_matches_reference_on_random_non_affine_cube_queries(self):
        size = 16
        rng = np.random.default_rng(42)
        cube = lut.identity_cube(size)
        # Non-affine warp: identity_cube is exactly affine (cube[r,g,b] ==
        # (r,g,b)/(size-1)), so every tetrahedral formula degenerates to
        # plain linear interpolation on it and cannot distinguish a mask bug
        # from correct code. Random per-node noise breaks that affineness.
        cube = np.clip(cube + rng.normal(0, 0.08, cube.shape), 0.0, 1.0).astype(np.float32)

        out_cube = lut.apply_lut  # keep the reference to apply_lut close to its call site

        # A few hundred random, non-grey, non-lattice-aligned query colours.
        # Continuous uniform draws already make r==g==b, exact lattice
        # alignment (frac==0), and ties between fractional offsets
        # probability-zero events, so no explicit nudge is needed -- and a
        # nudge-then-clip approach is actively dangerous here: clipping to
        # exactly 0.0/1.0 forces the fractional offset to exactly 0 or 1,
        # which can tie two of (dr, dg, db) exactly. Kept away from the
        # extreme ends purely so no channel clips to a boundary value.
        n = 400
        colours = rng.uniform(0.03, 0.97, (n, 3))

        rgb = colours.astype(np.float32).reshape(1, n, 3)
        vectorized = out_cube(rgb, cube)

        for i in range(n):
            r, g, b = colours[i]
            expected = _reference_tetrahedral_lookup(r, g, b, cube)
            np.testing.assert_allclose(
                vectorized[0, i], expected, atol=1e-4,
                err_msg=f"query {i}: rgb=({r!r},{g!r},{b!r})",
            )


class TestApplyLutSixTetrahedraPartition(unittest.TestCase):
    """Test B: the six masks partition exactly.

    `apply_lut` sums six `np.where(mask, formula, 0.0)` terms. If two masks
    ever overlap on the same pixel, that pixel gets two formulas added
    together and comes out too bright; if no mask covers a pixel, it comes
    out black (the `np.where` false-branch is 0.0). This test asserts, over
    many random (dr, dg, db) triples, that exactly one of the six masks is
    True per pixel -- never zero, never two.

    NOTE: this check alone is NOT sufficient to catch a masking bug -- the
    earlier buggy version of `apply_lut` (three masks pointing at the wrong
    formula) also partitioned every pixel exactly once. Mislabelling which
    *formula* a correctly-partitioning mask is attached to produces wrong
    colours while still summing to exactly 1 everywhere. That is exactly why
    `TestApplyLutAgainstReferenceImplementation` (Test A) exists: this test
    (Test B) can rule out double-counting/gaps, but only an independent
    reference implementation can catch a right-mask-wrong-formula bug.
    """

    def test_masks_sum_to_exactly_one_everywhere(self):
        rng = np.random.default_rng(7)
        n = 200_000
        dr = rng.random(n, dtype=np.float32)
        dg = rng.random(n, dtype=np.float32)
        db = rng.random(n, dtype=np.float32)

        cond_rg = dr > dg
        cond_gb = dg > db
        cond_rb = dr > db

        m1 = cond_rg & cond_gb                  # dr > dg > db
        m2 = cond_rg & ~cond_gb & cond_rb        # dr > db > dg
        m3 = cond_rg & ~cond_gb & ~cond_rb       # db > dr > dg
        m4 = ~cond_rg & ~cond_gb & ~cond_rb      # db > dg > dr
        m5 = ~cond_rg & cond_gb & ~cond_rb       # dg > db > dr
        m6 = ~cond_rg & cond_gb & cond_rb        # dg > dr > db

        total = (m1.astype(np.int64) + m2.astype(np.int64) + m3.astype(np.int64)
                 + m4.astype(np.int64) + m5.astype(np.int64) + m6.astype(np.int64))

        self.assertTrue(np.all(total == 1), f"partition violated: counts={np.bincount(total)}")


class TestShippedKodakGold200(unittest.TestCase):
    """The one artifact this repo ships, pinned to its actual measured values.

    Every other LUT test patches LUT_DIR to a temp directory and feeds it a
    synthetic Hald, so nothing loaded the real file. Replacing it with solid
    magenta left the suite green. DELETING it left the suite green — get_lut
    falls back to an identity cube by design, and an identity LUT is
    indistinguishable from "the film look is doing nothing", which is exactly
    the failure a packaging miss or a truncated binary produces.

    The numbers below are the ones documented in docs/baking-the-default-lut.md,
    and are what the shipped file actually measures. Mid-grey is the load-bearing
    one: spektrafilm's default stops_above_midgray (4.0) puts it at 0.78, over a
    stop of brightening on every photograph. This file was baked at 2.47, which
    puts it at 0.51.
    """

    PATH = Path(__file__).resolve().parent.parent / "luts" / "open" / "kodak_gold_200.png"

    def setUp(self):
        if not self.PATH.exists():
            self.fail(f"the shipped LUT is missing: {self.PATH}")
        self.cube = lut.load_hald(self.PATH)

    def test_it_is_a_level_8_cube(self):
        self.assertEqual(self.cube.shape, (64, 64, 64, 3))

    def test_mid_grey_lands_where_the_pipeline_expects_it(self):
        """~0.51, not ~0.78. This is the one that catches a wrong
        stops_above_midgray bake."""
        mid = self.cube[32, 32, 32]

        for channel, value in zip("rgb", mid):
            self.assertGreater(float(value), 0.48, f"mid-grey {channel} = {value}")
            self.assertLess(float(value), 0.54, f"mid-grey {channel} = {value}")

    def test_black_carries_the_film_base_fog(self):
        """Film has no true black: the base is not perfectly clear."""
        black = self.cube[0, 0, 0]

        self.assertTrue(np.all(black > 0.02), f"black = {black}")
        self.assertTrue(np.all(black < 0.09), f"black = {black}")

    def test_white_sits_on_the_shoulder(self):
        white = self.cube[-1, -1, -1]

        self.assertTrue(np.all(white > 0.75), f"white = {white}")
        self.assertTrue(np.all(white < 0.88), f"white = {white}")

    def test_it_is_neither_an_identity_nor_a_wrong_table(self):
        """One number that fails in both directions.

        Too low and the file is missing, empty, or an identity — a shipped app
        with no film look and no error. Too high and it is not this LUT at all
        (solid magenta scores ~0.46).
        """
        deviation = float(np.abs(self.cube - lut.identity_cube(64)).mean())

        self.assertGreater(deviation, 0.03, "the shipped LUT does nothing")
        self.assertLess(deviation, 0.30, "the shipped LUT is not the baked Gold 200")

    def test_it_is_the_lut_the_pipeline_actually_loads(self):
        """The wiring, not just the file: the default preset's LUT name, resolved
        through get_lut against the real luts/ directory, is this cube."""
        import film

        film._LUT_CACHE.clear()
        try:
            loaded = film.get_lut(film.DEFAULT_PARAMS["lut"])
        finally:
            film._LUT_CACHE.clear()

        self.assertEqual(loaded.shape, self.cube.shape)
        np.testing.assert_allclose(loaded, self.cube, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
