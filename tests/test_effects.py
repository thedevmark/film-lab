import math
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
        """The same preset must look the same on a preview and on an export:
        `radius` is a fraction of the long edge, so a highlight's bloom must
        cover the same fraction of the frame — and the same *absolute pixel
        extent scaled by resolution* — whether the image is a 1x preview or
        a 2x export.

        radius=0.05 is chosen so the sigma it implies is a real, substantial
        blur at both sizes (15px on a 300px long edge, 30px on 600px) — not
        something the box-blur's radius=1 floor collapses to noise. A test
        that only compares two near-zero bloom fractions proves nothing;
        both the "substantial bloom" and the "absolute extent doubles"
        assertions below are needed to rule out a raw-pixel-count reading of
        `radius` (which would clamp to an identical few-pixel bloom at both
        resolutions instead of scaling with the frame).
        """
        # Same scene, same aspect ratio, 2x resolution, highlight scaled
        # proportionally (a 2px half-width highlight becomes 4px at 2x).
        small = self._highlight(shape=(200, 300), size=2)
        large = self._highlight(shape=(400, 600), size=4)

        b_small = effects.add_halation(small, 0.5, radius=0.05) - small
        b_large = effects.add_halation(large, 0.5, radius=0.05) - large

        mask_small = b_small[:, :, 0] > 1e-4
        mask_large = b_large[:, :, 0] > 1e-4

        frac_small = float(mask_small.sum()) / small[:, :, 0].size
        frac_large = float(mask_large.sum()) / large[:, :, 0].size

        # The bloom must be a substantial feature of the frame before its
        # extent means anything -- otherwise comparing two near-zero numbers
        # would pass regardless of how `radius` is interpreted.
        self.assertGreater(frac_small, 0.02)
        self.assertGreater(frac_large, 0.02)

        # Bloom extent, as a fraction of the frame, must match.
        self.assertAlmostEqual(frac_small, frac_large, delta=0.03)

        # And the absolute spatial extent, in pixels, must double along with
        # the frame -- the direct signature of sigma scaling with
        # resolution rather than being a fixed pixel count. Use the square
        # root of the bloomed-pixel count as a linear measure of extent.
        extent_small = math.sqrt(float(mask_small.sum()))
        extent_large = math.sqrt(float(mask_large.sum()))

        self.assertAlmostEqual(extent_large / extent_small, 2.0, delta=0.3)


class TestWhichEdgeEachSizeKeysOff(unittest.TestCase):
    """Grain keys off the SHORT edge; halation keys off the LONG edge.

    The resolution-independence tests above and below cannot see this: they
    compare (200,300) against (400,600), where BOTH dimensions scale by the same
    factor, so min() and max() move together and swapping one for the other
    changes nothing they measure. Both effects could be keying off the wrong edge
    and every one of those assertions would still hold.

    The three frames here separate them. A 200x800 frame and its transpose share
    a short edge (200) and a long edge (800) — so both effects must match across
    the pair whichever edge they use, which pins nothing on its own but does pin
    that neither effect cares about orientation. The 200x200 square is what
    separates: it has the same SHORT edge as the rectangles and a 4x smaller LONG
    edge. So grain must be the SAME size in the square as in the rectangles, and
    halation must be 4x SMALLER. Get min and max the wrong way round and those
    two expectations trade places.
    """

    SHORT = 200
    LONG = 800

    def _highlight(self, shape, size=2):
        img = np.full(shape + (3,), 0.02, dtype=np.float32)
        cy, cx = shape[0] // 2, shape[1] // 2
        img[cy - size:cy + size, cx - size:cx + size] = 1.0
        return img

    def _bloom_extent(self, shape, radius=0.02):
        """The bloom's standard deviation, in pixels, along the frame's long axis.

        A second moment rather than a thresholded pixel count: the count depends
        on the bloom's peak amplitude, which itself falls as the blur widens, so
        a mask area is only loosely proportional to sigma. The second moment is
        var(source) + sigma**2 exactly, whatever the amplitude.
        """
        img = self._highlight(shape)
        bloom = effects.add_halation(img, intensity=0.5, radius=radius) - img
        bloom = bloom[:, :, 0].astype(np.float64)

        long_axis = 0 if shape[0] > shape[1] else 1
        profile = bloom.sum(axis=1 - long_axis)  # the source is separable, so this is exact

        index = np.arange(profile.size, dtype=np.float64)
        total = profile.sum()
        centre = (index * profile).sum() / total
        return math.sqrt(float((profile * (index - centre) ** 2).sum() / total))

    def _grain_extent(self, shape, size=0.05, seed=1):
        """The grain's correlation length, in pixels.

        For a Gaussian-filtered white-noise field of width sigma,
        E[(dg/dx)^2] / E[g^2] == 1 / (2*sigma**2), so sigma == 1/sqrt(2r). This
        is immune to add_grain's renormalisation back to unit variance — which
        is exactly what makes a plain .std() comparison blind to grain size.
        """
        img = np.full(shape + (3,), 0.5, dtype=np.float32)
        grain = effects.add_grain(img, 0.1, size=size, seed=seed) - img
        field = grain[:, :, 0].astype(np.float64)

        delta = np.diff(field, axis=1)
        ratio = float((delta ** 2).mean()) / float((field ** 2).mean())
        return 1.0 / math.sqrt(2.0 * ratio)

    def test_grain_keys_off_the_short_edge(self):
        landscape = self._grain_extent((self.SHORT, self.LONG))
        portrait = self._grain_extent((self.LONG, self.SHORT))
        square = self._grain_extent((self.SHORT, self.SHORT))

        # Substantial, multi-pixel blobs — otherwise the ratios below compare
        # noise.
        self.assertGreater(square, 3.0)

        # The square's SHORT edge is the same 200 as the rectangle's, but its
        # LONG edge is a quarter of it. Same short edge => same grain. Keying off
        # the long edge would make the rectangle's grain 4x coarser.
        self.assertAlmostEqual(landscape / square, 1.0, delta=0.15,
                               msg="grain is not keyed off the short edge")

        # And orientation is irrelevant: the transpose shares both edges.
        self.assertAlmostEqual(landscape / portrait, 1.0, delta=0.15)

    def test_halation_keys_off_the_long_edge(self):
        landscape = self._bloom_extent((self.SHORT, self.LONG))
        portrait = self._bloom_extent((self.LONG, self.SHORT))
        square = self._bloom_extent((self.SHORT, self.SHORT))

        self.assertGreater(square, 3.0)

        # The square's LONG edge is a quarter of the rectangle's, so its bloom
        # must be about a quarter the size. (Not exactly 4x: gaussian_blur rounds
        # its box radius to a whole pixel, which costs a few percent at these
        # sigmas.) Keying off the short edge would make this ratio 1.0 — both
        # frames have a 200px short edge.
        self.assertGreater(landscape / square, 3.0,
                           "halation is not keyed off the long edge")
        self.assertLess(landscape / square, 4.5)

        # And orientation is irrelevant: the transpose shares both edges.
        self.assertAlmostEqual(landscape / portrait, 1.0, delta=0.05)


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
        """`size` is a fraction of the short edge, so the same relative grain
        BLOB SIZE should appear whether the frame is a preview or a full-res
        export. Global amplitude (.std() over the whole field) is NOT a valid
        proxy for this: add_grain renormalises the blurred noise field back to
        unit variance, so the overall amplitude comes out ~identical regardless
        of whether sigma actually scaled with resolution -- an amplitude-only
        comparison would pass even if `size` were misread as a raw pixel count.
        Instead, measure the spatial extent of a grain blob directly (mean
        same-sign run length along rows), the same technique used for
        halation's resolution-independence test, and require it to roughly
        double when the frame doubles -- and to be well above one pixel to
        begin with, so we are not comparing near-meaningless small numbers.
        """
        small = np.full((100, 150, 3), 0.5, dtype=np.float32)
        large = np.full((200, 300, 3), 0.5, dtype=np.float32)

        g_small = effects.add_grain(small, 0.2, size=0.05, seed=1) - small
        g_large = effects.add_grain(large, 0.2, size=0.05, seed=1) - large

        # Same relative grain size => similar global amplitude, not wildly
        # different (a coarse sanity check; the real assertion is extent below).
        self.assertAlmostEqual(float(g_small.std()), float(g_large.std()), delta=0.02)

        def mean_run_length(channel):
            """Average length, in pixels, of a same-sign run along each row."""
            signs = np.sign(channel).astype(np.int8)
            total_len = 0
            total_runs = 0
            for row in signs:
                boundaries = np.flatnonzero(np.diff(row) != 0) + 1
                run_lengths = np.diff(np.concatenate(([0], boundaries, [row.size])))
                total_len += row.size
                total_runs += run_lengths.size
            return total_len / total_runs

        extent_small = mean_run_length(g_small[:, :, 0])
        extent_large = mean_run_length(g_large[:, :, 0])

        # The grain blobs must be a substantial, multi-pixel feature -- not
        # single-pixel iid noise -- before their extent means anything.
        self.assertGreater(extent_small, 1.5)
        self.assertGreater(extent_large, 1.5)

        # And the extent, in pixels, must double along with the frame: the
        # direct signature of sigma scaling with resolution rather than being
        # a fixed pixel count.
        self.assertAlmostEqual(extent_large / extent_small, 2.0, delta=0.5)

    def test_absurd_size_does_not_explode(self):
        img = np.full((64, 64, 3), 0.5, dtype=np.float32)

        out = effects.add_grain(img, 0.05, size=1000.0, seed=1)

        self.assertEqual(out.shape, img.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_output_stays_in_range(self):
        img = np.full((64, 64, 3), 0.99, dtype=np.float32)

        out = effects.add_grain(img, 0.5, 0.02, seed=1)

        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))


if __name__ == "__main__":
    unittest.main()
