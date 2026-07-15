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

        # -1e-6, not -1e-7: float32 epsilon near 1.0 is ~1.2e-7, and the
        # divide-then-multiply round trip legitimately costs a few ULPs.
        # This is not real non-monotonicity, just float32 slack.
        self.assertTrue(np.all(np.diff(out) >= -1e-6), "rolloff must be monotonic")
        # No step at the knee: the largest jump should be tiny.
        self.assertLess(float(np.abs(np.diff(out)).max()), 0.01)

    def test_is_pointwise(self):
        """Each pixel must depend on itself alone.

        A previous implementation ran a cumulative maximum across the flattened
        image, so one bright pixel raised the scale factor for every pixel after
        it in raster order. Nothing in the suite caught it.
        """
        rng = np.random.default_rng(0)
        img = (rng.random((8, 8, 3), dtype=np.float32) * 3.0).astype(np.float32)

        whole = tone.highlight_rolloff(img)

        # Every pixel, processed alone, must give the same answer as it does
        # inside the full image.
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                alone = tone.highlight_rolloff(img[y:y + 1, x:x + 1, :])
                np.testing.assert_allclose(
                    alone[0, 0], whole[y, x], atol=1e-6,
                    err_msg=f"pixel ({y},{x}) changed depending on its neighbours",
                )

    def test_pixel_order_does_not_matter(self):
        """Reversing raster order must reverse the output, nothing more."""
        rng = np.random.default_rng(1)
        img = (rng.random((4, 6, 3), dtype=np.float32) * 3.0).astype(np.float32)

        forward = tone.highlight_rolloff(img)
        reversed_out = tone.highlight_rolloff(img[::-1, ::-1, :])

        np.testing.assert_allclose(reversed_out, forward[::-1, ::-1, :], atol=1e-6)


class TestHuePreservingClip(unittest.TestCase):
    """The DISPLAY-side highlight transform.

    A camera JPEG has already been rendered. The only correct transform in front
    of the LUT is the IDENTITY — and there is no smooth alternative: any
    monotonic map that is the identity on [0,1] and lands in [0,1] IS min(x, 1).
    So the only freedom left is in HOW the over-range values (which only an
    exposure push can produce) come back down, and that is done by scaling the
    triplet rather than clipping each channel on its own.
    """

    def test_is_the_identity_in_range(self):
        """The whole point. A rolloff here would crush white to grey."""
        rgb = np.array([[[0.0, 0.5, 1.0], [1.0, 1.0, 1.0], [0.81, 0.9, 0.999]]],
                       dtype=np.float32)

        out = tone.hue_preserving_clip(rgb)

        np.testing.assert_allclose(out, rgb, atol=1e-7)

    def test_white_stays_white(self):
        rgb = np.ones((2, 2, 3), dtype=np.float32)

        np.testing.assert_allclose(tone.hue_preserving_clip(rgb), 1.0, atol=1e-7)

    def test_over_range_keeps_channel_ratios(self):
        """A per-channel clip would flatten 4:2:1 to 1:1:1 and rotate the hue."""
        rgb = np.array([[[4.0, 2.0, 1.0]]], dtype=np.float32)

        out = tone.hue_preserving_clip(rgb)[0, 0]

        self.assertAlmostEqual(float(out[0]), 1.0, places=6)
        self.assertAlmostEqual(float(out[0] / out[1]), 2.0, places=5)
        self.assertAlmostEqual(float(out[1] / out[2]), 2.0, places=5)

    def test_output_never_exceeds_one(self):
        rgb = np.array([[[8.0, 4.0, 2.0], [100.0, 100.0, 100.0]]], dtype=np.float32)

        out = tone.hue_preserving_clip(rgb)

        self.assertTrue(np.all(out <= 1.0 + 1e-6))
        self.assertTrue(np.all(out >= 0.0))

    def test_black_does_not_divide_by_zero(self):
        rgb = np.zeros((2, 2, 3), dtype=np.float32)

        out = tone.hue_preserving_clip(rgb)

        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out, 0.0, atol=1e-7)

    def test_is_pointwise(self):
        """Each pixel depends on itself alone.

        An earlier transform in this file ran a cumulative maximum across the
        flattened image, so one bright pixel raised the scale factor of every
        pixel after it in raster order. Nothing like that here.
        """
        rng = np.random.default_rng(0)
        img = (rng.random((8, 8, 3), dtype=np.float32) * 3.0).astype(np.float32)

        whole = tone.hue_preserving_clip(img)

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                alone = tone.hue_preserving_clip(img[y:y + 1, x:x + 1, :])
                np.testing.assert_allclose(
                    alone[0, 0], whole[y, x], atol=1e-6,
                    err_msg=f"pixel ({y},{x}) changed depending on its neighbours",
                )

    def test_is_monotonic(self):
        ramp = np.linspace(0.0, 6.0, 2000, dtype=np.float32)
        rgb = np.stack([ramp, ramp, ramp], axis=-1)[None, :, :]

        out = tone.hue_preserving_clip(rgb)[0, :, 0]

        self.assertTrue(np.all(np.diff(out) >= -1e-6))


class TestRolloffIsNotAnIdentity(unittest.TestCase):
    """The reason the two transforms cannot be the same function.

    highlight_rolloff is asymptotic to 1.0, so it never reaches white: f(1.0) is
    0.926 in linear light. That is correct for a scene-linear render, where 1.0
    is just another scene value with headroom above it, and catastrophic for an
    already-rendered JPEG, where 1.0 is white and must stay white.
    """

    def test_rolloff_cannot_reach_white(self):
        white = np.ones((1, 1, 3), dtype=np.float32)

        out = tone.highlight_rolloff(white)

        self.assertLess(float(out.max()), 0.95)
        # ...whereas the display-side transform lands it exactly on white.
        self.assertAlmostEqual(float(tone.hue_preserving_clip(white).max()), 1.0,
                               places=6)


if __name__ == "__main__":
    unittest.main()
