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
