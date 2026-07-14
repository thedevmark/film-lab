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


if __name__ == "__main__":
    unittest.main()
