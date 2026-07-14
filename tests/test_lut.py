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


if __name__ == "__main__":
    unittest.main()
