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
