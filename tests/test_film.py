import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from flask import Flask
from PIL import Image

import film


def _solid_hald(path: Path, colour):
    """An 8x8 PNG is a valid level-2 HaldCLUT (2**3 == 8 px, 4**3 == 64 entries).

    A solid one maps every input colour to `colour`, which makes it trivial to
    tell two LUTs apart without caring what either of them does.
    """
    Image.new("RGB", (8, 8), colour).save(path)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        film._LUT_CACHE.clear()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        film._LUT_CACHE.clear()
        self.tmp.cleanup()

    def _file(self, name: str, content: bytes = b"bytes on disk") -> Path:
        """A real file at a real path.

        process_photo hashes the input file to derive the grain seed, so the
        path has to exist even in a test that patches load_image — which is what
        the pipeline is always handed in production anyway.
        """
        path = Path(self.tmp.name) / name
        path.write_bytes(content)
        return path

    def _record_stages(self, state, params):
        """Run process_photo with every stage recorded, for one input state."""
        order = []
        seen = {}
        source = np.full((8, 8, 3), 0.25, dtype=np.float32)

        def record(name, fn):
            def inner(*args, **kwargs):
                order.append(name)
                return fn(*args, **kwargs)
            return inner

        def lut(img, cube, strength=1.0):
            seen["strength"] = strength
            return img

        def grain(img, **kwargs):
            seen["seed"] = kwargs.get("seed")
            return img

        suffix = ".arw" if state == "scene" else ".jpg"
        with patch.object(film, "load_image",
                          return_value=(source.copy(), state)), \
             patch.object(film, "apply_exposure",
                          side_effect=record("exposure", lambda img, ev: img)), \
             patch.object(film, "add_halation",
                          side_effect=record("halation", lambda img, **k: img)), \
             patch.object(film, "highlight_rolloff",
                          side_effect=record("rolloff", lambda img: img)), \
             patch.object(film, "hue_preserving_clip",
                          side_effect=record("clip", lambda img: img)), \
             patch.object(film, "apply_lut",
                          side_effect=record("lut", lut)), \
             patch.object(film, "apply_contrast",
                          side_effect=record("contrast", lambda img, strength: img)), \
             patch.object(film, "add_grain",
                          side_effect=record("grain", grain)):
            out = film.process_photo(self._file(f"x{suffix}"), params)

        return order, seen, out

    def test_pipeline_runs_in_the_documented_order(self):
        """Every stage of the pipeline is recorded here, and the params that
        reach them are captured.

        The highlight transform, grade_strength and seed are all in this test
        because each of them could be deleted, hardcoded or ignored without a
        single test failing: the order list only pinned the stages it happened to
        name, and nothing looked at what the stages were called WITH.
        """
        from filmlab import loader

        params = {"grain_intensity": 0.05, "grade_strength": 0.25, "seed": 4321}
        order, seen, out = self._record_stages(loader.DISPLAY, params)

        # A JPEG is already camera-rendered, so the highlight stage is a
        # hue-preserving CLIP — the identity in range. A shoulder here would be a
        # second tone curve in front of the LUT, and would make white unreachable.
        # Contrast is AFTER the LUT: the CLUTs were authored against a neutral,
        # standard-contrast render, so look-contrast before them is counted twice.
        # Grain is LAST: it is the texture on a finished frame, not a signal that
        # gets graded.
        self.assertEqual(
            order, ["exposure", "halation", "clip", "lut", "contrast", "grain"]
        )

        # grade_strength is the headline control of the whole feature, and seed
        # is the difference between a reproducible render and a lottery. Both
        # must actually arrive at the stage that uses them.
        self.assertEqual(seen["strength"], 0.25)
        self.assertEqual(seen["seed"], 4321)

        self.assertTrue(out.startswith(b"\xff\xd8"))

        decoded = Image.open(io.BytesIO(out))
        self.assertEqual(decoded.format, "JPEG")
        self.assertEqual(decoded.size, (8, 8))

    def test_scene_input_gets_the_shoulder_and_display_input_does_not(self):
        """The state tag exists to pick the highlight transform, and this is the
        only place the choice is visible.

        SCENE is scene-linear with real headroom above 1.0, so the soft shoulder
        IS its scene-to-display render. DISPLAY has already been rendered by the
        camera; a shoulder there would crush white and stack a second tone curve
        in front of the LUT.
        """
        from filmlab import loader

        params = {"grain_intensity": 0.0, "seed": 1}

        scene_order, _, _ = self._record_stages(loader.SCENE, params)
        display_order, _, _ = self._record_stages(loader.DISPLAY, params)

        self.assertIn("rolloff", scene_order)
        self.assertNotIn("clip", scene_order)

        self.assertIn("clip", display_order)
        self.assertNotIn("rolloff", display_order)

    def test_pure_white_survives_the_zeroed_pipeline(self):
        """With every control at zero, a white wall must come out white.

        The shoulder used to run on both states. It is asymptotic to 1.0, so
        f(1.0) = 0.926 in linear light, and 255 came out of the JPEG as 246 with
        the top 21 code values squashed into 13. Every photo with a blown sky or
        a white wall rendered it light grey, at every setting, with no way to
        turn it off. It also made the LUT's own calibration point — encoded 1.0,
        which is what stops_above_midgray=2.47 was baked against — unreachable.
        """
        source = Path(self.tmp.name) / "white.jpg"
        Image.new("RGB", (16, 16), (255, 255, 255)).save(source, quality=95)

        with patch.object(film, "LUT_DIR", Path(self.tmp.name) / "no-luts"):
            out = film.process_photo(source, {
                "lut":                "nothing_installed",
                "grade_strength":     0.0,
                "exposure_bias":      0.0,
                "contrast_strength":  0.0,
                "grain_intensity":    0.0,
                "halation_intensity": 0.0,
            })

        decoded = np.asarray(Image.open(io.BytesIO(out)), dtype=np.int16)
        self.assertEqual(int(decoded.min()), 255,
                         f"white came back as {int(decoded.min())}, not 255")

    def test_the_zeroed_display_pipeline_is_an_identity(self):
        """Not just white: with every control at zero and no LUT, a JPEG in must
        be the same JPEG out, to within JPEG quantisation. The pre-LUT image is
        supposed to be a neutral, standard-contrast sRGB render — the very space
        the CLUT was authored against — and any tone curve on this path is one
        the LUT will then count twice."""
        source = Path(self.tmp.name) / "ramp.jpg"
        # A ramp that reaches both 0 and 255 on every channel — the top of the
        # range is exactly where the shoulder did its damage.
        axis = (np.arange(32) * 255 // 31).astype(np.uint8)
        ramp = np.stack([
            np.repeat(axis[:, None], 32, axis=1),
            np.repeat(axis[None, :], 32, axis=0),
            np.repeat(axis[::-1, None], 32, axis=1),
        ], axis=-1)
        Image.fromarray(ramp).save(source, quality=95)
        original = np.asarray(Image.open(source), dtype=np.int16)

        with patch.object(film, "LUT_DIR", Path(self.tmp.name) / "no-luts"):
            out = film.process_photo(source, {
                "lut":                "nothing_installed",
                "grade_strength":     0.0,
                "exposure_bias":      0.0,
                "contrast_strength":  0.0,
                "grain_intensity":    0.0,
                "halation_intensity": 0.0,
            })

        decoded = np.asarray(Image.open(io.BytesIO(out)), dtype=np.int16)
        delta = np.abs(decoded - original)
        self.assertLessEqual(int(delta.max()), 6,
                             f"zeroed pipeline moved a pixel by {int(delta.max())} levels")
        self.assertLess(float(delta.mean()), 2.0)

    def test_grey_point_scale_runs_before_halation(self):
        """HALATION_THRESHOLD (0.70) and the rolloff knee (0.8) are both ABSOLUTE
        linear values. If the grey-point scale sits between them, the same RAW
        highlight is tested for halation at one exposure and rolled off at
        another. Masked today only because GREY_DISPLAY / GREY_SCENE happens to
        be 1.0."""
        from filmlab import loader

        source = np.full((4, 4, 3), 0.25, dtype=np.float32)
        seen = {}

        def halation(img, **kwargs):
            seen["halation_in"] = float(np.asarray(img, dtype=np.float32).mean())
            return img

        def rolloff(img):
            seen["rolloff_in"] = float(np.asarray(img, dtype=np.float32).mean())
            return img

        # Scene grey at 0.36 => the scale to a 0.18 display grey is a halving.
        with patch.object(film, "GREY_SCENE", 0.36), \
             patch.object(film, "load_image",
                          return_value=(source.copy(), loader.SCENE)), \
             patch.object(film, "add_halation", side_effect=halation), \
             patch.object(film, "highlight_rolloff", side_effect=rolloff):
            film.process_photo(self._file("x.arw"), {"grain_intensity": 0.0})

        # Both absolute thresholds must see the SAME exposure.
        self.assertAlmostEqual(seen["halation_in"], 0.125, places=5,
                               msg="halation ran before the grey-point scale")
        self.assertAlmostEqual(seen["rolloff_in"], 0.125, places=5)

    def test_blown_highlights_keep_their_hue_through_the_whole_pipeline(self):
        """The unmocked half of the rolloff's wiring.

        A +2 EV push sends linear values past 1.0. Without the rolloff, the clip
        after srgb_encode crushes each channel independently: the brightest
        channel of a warm highlight hits 1.0 first and the highlight drifts
        toward yellow-white on its way out. The rolloff compresses the whole
        triplet by one scale factor, so the linear channel RATIOS — and so the
        hue — come out of the JPEG intact.
        """
        from filmlab import loader
        from filmlab.tone import srgb_decode, srgb_encode

        # A warm highlight two stops over: ratios 1 : 0.7 : 0.35.
        linear = np.zeros((8, 8, 3), dtype=np.float32)
        linear[:, :, 0] = 2.0
        linear[:, :, 1] = 1.4
        linear[:, :, 2] = 0.7

        with patch.object(film, "load_image",
                          return_value=(linear.copy(), loader.DISPLAY)):
            out = film.process_photo(self._file("x.jpg"), {
                # Only the tone path under test: no LUT, no grain, no halation.
                "grade_strength": 0.0,
                "halation_intensity": 0.0,
                "grain_intensity": 0.0,
                "contrast_strength": 0.0,
            })

        # Sanity: red AND green really are over range on their way in, so a
        # per-channel clip really would flatten both of them to the same 1.0.
        self.assertGreater(float(srgb_encode(np.float32(2.0))), 1.0)
        self.assertGreater(float(srgb_encode(np.float32(1.4))), 1.0)

        decoded = np.asarray(Image.open(io.BytesIO(out)), dtype=np.float32) / 255.0
        result = srgb_decode(decoded)
        red = float(result[:, :, 0].mean())
        green = float(result[:, :, 1].mean())
        blue = float(result[:, :, 2].mean())

        # The over-range input came back into the display range with its channel
        # ratios intact. A per-channel clip would have sent both red and green to
        # 1.0, making green/red 1.0 instead of 0.7.
        self.assertAlmostEqual(green / red, 1.4 / 2.0, delta=0.02)
        self.assertAlmostEqual(blue / red, 0.7 / 2.0, delta=0.02)

    def test_the_lut_sees_srgb_encoded_data_not_linear_light(self):
        """The CLUT is a map on display-referred colour. Handing it linear light
        applies the film's colour to the wrong tone scale entirely."""
        from filmlab import loader
        from filmlab.tone import srgb_encode

        seen = {}
        source = np.full((4, 4, 3), 0.25, dtype=np.float32)

        def capture(img, cube, strength=1.0):
            seen["rgb"] = np.asarray(img, dtype=np.float32).copy()
            return img

        with patch.object(film, "load_image",
                          return_value=(source.copy(), loader.DISPLAY)), \
             patch.object(film, "apply_lut", side_effect=capture):
            film.process_photo(self._file("x.jpg"), {
                "halation_intensity": 0.0,
                "grain_intensity": 0.0,
            })

        encoded = float(srgb_encode(np.float32(0.25)))
        self.assertGreater(encoded, 0.4, "sanity: sRGB encoding really does lift 0.25")
        np.testing.assert_allclose(seen["rgb"], encoded, atol=1e-5)

    def test_grey_point_scale_applies_to_scene_input_only(self):
        """Scene-linear RAW needs middle grey placed where the CLUT expects it.
        A camera JPEG has already been rendered — scaling it again is a second
        tone transform on top of the one baked in."""
        from filmlab import loader
        from filmlab.tone import srgb_encode

        source = np.full((4, 4, 3), 0.25, dtype=np.float32)
        seen = []

        def capture(img, cube, strength=1.0):
            seen.append(float(np.asarray(img, dtype=np.float32).mean()))
            return img

        params = {"halation_intensity": 0.0, "grain_intensity": 0.0}

        # Scene grey at 0.36 => the scale to a 0.18 display grey is a halving.
        with patch.object(film, "GREY_SCENE", 0.36), \
             patch.object(film, "apply_lut", side_effect=capture):
            with patch.object(film, "load_image",
                              return_value=(source.copy(), loader.DISPLAY)):
                film.process_photo(self._file("x.jpg"), params)
            with patch.object(film, "load_image",
                              return_value=(source.copy(), loader.SCENE)):
                film.process_photo(self._file("x.arw"), params)

        display_value, scene_value = seen
        np.testing.assert_allclose(display_value, float(srgb_encode(np.float32(0.25))),
                                   atol=1e-5)
        self.assertLess(scene_value, display_value - 0.05,
                        "SCENE input did not go through the grey-point scale")

    def test_exposure_is_multiplicative_not_additive(self):
        """+1 EV must double the linear light, preserving channel ratios (and so
        hue). The old `img + bias` was a black-level lift in gamma space."""
        linear = np.full((4, 4, 3), 0.1, dtype=np.float32)
        linear[:, :, 0] = 0.2  # a colour, so a hue shift would show

        out = film.apply_exposure(linear, 1.0)

        np.testing.assert_allclose(out[:, :, 0], 0.4, atol=1e-5)
        np.testing.assert_allclose(out[:, :, 1], 0.2, atol=1e-5)
        # Ratio held => hue held.
        np.testing.assert_allclose(out[:, :, 0] / out[:, :, 1],
                                   linear[:, :, 0] / linear[:, :, 1], atol=1e-5)

    def test_exposure_keeps_highlight_headroom_above_one(self):
        """Clipping here would throw away the very headroom the rolloff exists
        to recover."""
        linear = np.full((2, 2, 3), 0.8, dtype=np.float32)

        out = film.apply_exposure(linear, 2.0)

        np.testing.assert_allclose(out, 3.2, atol=1e-5)

    def test_contrast_pivots_on_mid_grey(self):
        rgb = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)

        out = film.apply_contrast(rgb, 0.5)

        np.testing.assert_allclose(out, np.array([[[0.125, 0.5, 0.875]]], dtype=np.float32),
                                   atol=1e-5)
        np.testing.assert_allclose(film.apply_contrast(rgb, 0.0), rgb)

    def test_apply_film_color_is_gone(self):
        self.assertFalse(hasattr(film, "apply_film_color"),
                         "the hand-tuned colour maths should have been deleted")


class TestGrainSeed(unittest.TestCase):
    """seed = 0 is the "auto" sentinel, and auto means derived from the FILE.

    It used to mean literally zero: nothing in DEFAULT_PARAMS, the presets or the
    UI ever set it, so np.random.default_rng(0) drew the same field for every
    photo. Two photos of the same dimensions got bit-identical grain — fixed
    pattern noise across a whole shoot, plainly visible once the set is viewed
    together.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        film._LUT_CACHE.clear()

    def tearDown(self):
        film._LUT_CACHE.clear()
        self.tmp.cleanup()

    BASE = {
        "lut":                "nothing_installed",
        "grade_strength":     0.0,
        "halation_intensity": 0.0,
        "grain_intensity":    0.05,
        "grain_size":         0.001,  # sigma < 0.5px: the raw draw, unblurred
    }

    def _grain_field(self, source: Path, params=None):
        """The grain the pipeline lays on `source`, isolated by subtracting the
        same render with grain switched off."""
        base = {**self.BASE, **(params or {})}

        with patch.object(film, "LUT_DIR", self.dir / "no-luts"):
            grainy = film.process_photo(source, base)
            clean = film.process_photo(source, {**base, "grain_intensity": 0.0})

        a = np.asarray(Image.open(io.BytesIO(grainy)), dtype=np.float32)
        b = np.asarray(Image.open(io.BytesIO(clean)), dtype=np.float32)
        return a - b

    def _seed_reaching_grain(self, source: Path, params=None):
        seen = {}

        def grain(img, **kwargs):
            seen["seed"] = kwargs.get("seed")
            return img

        with patch.object(film, "LUT_DIR", self.dir / "no-luts"), \
             patch.object(film, "add_grain", side_effect=grain):
            film.process_photo(source, {**self.BASE, **(params or {})})
        return seen["seed"]

    def _photo(self, name: str, colour) -> Path:
        path = self.dir / name
        Image.new("RGB", (32, 32), colour).save(path, quality=95)
        return path

    def test_two_photos_do_not_share_one_grain_field(self):
        first = self._grain_field(self._photo("a.jpg", (128, 128, 128)))
        second = self._grain_field(self._photo("b.jpg", (127, 128, 129)))

        # Same size, same settings, different FILE — so a different field.
        self.assertEqual(first.shape, second.shape)
        self.assertGreater(float(np.abs(first - second).mean()), 1.0,
                           "two different photos got the same grain field")

    def test_the_same_photo_twice_gets_the_same_grain(self):
        source = self._photo("a.jpg", (128, 128, 128))

        first = self._grain_field(source)
        second = self._grain_field(source)

        np.testing.assert_array_equal(first, second)

    def test_an_explicit_seed_still_wins(self):
        """Reproducibility is not negotiable: a non-zero seed overrides the file
        hash, so two different photos rendered with the same explicit seed get
        the same field."""
        a = self._photo("a.jpg", (128, 128, 128))
        b = self._photo("b.jpg", (200, 40, 60))

        self.assertEqual(self._seed_reaching_grain(a, {"seed": 99}), 99)
        self.assertEqual(self._seed_reaching_grain(b, {"seed": 99}), 99)

        # ...and with the sentinel, the seed that reaches the grain is the file's.
        self.assertEqual(self._seed_reaching_grain(a), film._seed_from_file(a))
        self.assertNotEqual(self._seed_reaching_grain(a),
                            self._seed_reaching_grain(b))

    def test_the_derived_seed_is_in_range_and_stable(self):
        source = self._photo("a.jpg", (128, 128, 128))

        seed = film._seed_from_file(source)

        self.assertIsInstance(seed, int)
        self.assertGreaterEqual(seed, 0)
        self.assertEqual(seed, film._seed_from_file(source))
        self.assertNotEqual(seed, film._seed_from_file(self._photo("b.jpg", (10, 20, 30))))


class TestLutSelection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lut_dir = Path(self.tmp.name)
        film._LUT_CACHE.clear()

    def tearDown(self):
        film._LUT_CACHE.clear()
        self.tmp.cleanup()

    def test_missing_lut_degrades_to_an_identity_cube(self):
        """The repo ships with an empty luts/ directory. A missing LUT must be a
        no-op colour stage, not a failed render."""
        from filmlab.lut import apply_lut

        with patch.object(film, "LUT_DIR", self.lut_dir):
            cube = film.get_lut("nothing_installed")

        probe = np.array([[[0.1, 0.5, 0.9], [0.0, 1.0, 0.35]]], dtype=np.float32)
        np.testing.assert_allclose(apply_lut(probe, cube), probe, atol=1e-6)

    def test_process_photo_renders_with_no_luts_installed(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "photo.jpg"
            Image.new("RGB", (16, 12), (180, 120, 60)).save(source)

            with patch.object(film, "LUT_DIR", self.lut_dir):
                out = film.process_photo(source, {"lut": "nothing_installed"})

        self.assertTrue(out.startswith(b"\xff\xd8"))
        decoded = Image.open(io.BytesIO(out))
        self.assertEqual(decoded.size, (16, 12))

    def test_private_lut_wins_over_the_open_one_of_the_same_name(self):
        (self.lut_dir / "open").mkdir()
        (self.lut_dir / "private").mkdir()
        _solid_hald(self.lut_dir / "open" / "kodak_gold_200.png", (0, 0, 255))
        _solid_hald(self.lut_dir / "private" / "kodak_gold_200.png", (255, 0, 0))

        with patch.object(film, "LUT_DIR", self.lut_dir):
            cube = film.get_lut("kodak_gold_200")

        np.testing.assert_allclose(cube[..., 0], 1.0, atol=1e-5)  # red, not blue
        np.testing.assert_allclose(cube[..., 2], 0.0, atol=1e-5)

    def test_open_lut_is_used_when_no_private_one_exists(self):
        (self.lut_dir / "open").mkdir()
        _solid_hald(self.lut_dir / "open" / "kodak_gold_200.png", (0, 0, 255))

        with patch.object(film, "LUT_DIR", self.lut_dir):
            cube = film.get_lut("kodak_gold_200")

        np.testing.assert_allclose(cube[..., 2], 1.0, atol=1e-5)
        np.testing.assert_allclose(cube[..., 0], 0.0, atol=1e-5)

    def test_a_lut_installed_after_a_miss_is_picked_up_without_a_restart(self):
        """docs/extracting-a-lut.md says: extract a LUT, drop it into
        luts/private/, re-render. The whole workflow depends on that.

        The cache used to be keyed on the LUT's NAME and used to store the
        identity fallback under the name of the LUT that was MISSING — so the
        first render before the file existed poisoned the cache for the life of
        the process, and the freshly extracted LUT did nothing. A LUT doing
        nothing looks exactly like a LUT that is wrong, and the acceptance check
        in that doc tells you a wrong-looking result means go re-extract it.

        Note the deliberate absence of a cache clear between the two calls.
        """
        from filmlab.lut import apply_lut

        (self.lut_dir / "private").mkdir()
        probe = np.array([[[0.1, 0.5, 0.9], [0.0, 1.0, 0.35]]], dtype=np.float32)

        with patch.object(film, "LUT_DIR", self.lut_dir):
            # Nothing installed yet: a no-op colour stage.
            cube = film.get_lut("freshly_extracted")
            np.testing.assert_allclose(apply_lut(probe, cube), probe, atol=1e-6)

            # The user drops their extracted Hald in, and re-renders.
            _solid_hald(self.lut_dir / "private" / "freshly_extracted.png", (255, 0, 0))
            cube = film.get_lut("freshly_extracted")

        np.testing.assert_allclose(cube[..., 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(cube[..., 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(cube[..., 2], 0.0, atol=1e-5)


class TestParams(unittest.TestCase):
    def test_defaults_cover_every_spec_key_plus_the_lut(self):
        clean = film.coerce_params({})

        for key in film.PARAM_SPEC:
            self.assertIn(key, clean)
        self.assertEqual(clean["lut"], "kodak_gold_200")

    # A test asserting that coerce_params({"grain_size": 3}) yields <= 0.05 used
    # to live here. It was vacuous: the clamp makes that true for ANY input, and
    # the value it was passing lands exactly ON the maximum — which is the precise
    # disaster _migrate_legacy_units exists to prevent, asserted as if it were the
    # desired outcome. TestLegacyPresetUnits below pins the real behaviour (v1
    # pixel values fall back to the DEFAULTS, not the maxima), and
    # test_defects.TestParamValidation pins the boundary clamp itself.

    def test_exposure_bias_is_ev_stops(self):
        self.assertEqual(film.coerce_params({"exposure_bias": -3.0})["exposure_bias"], -3.0)
        self.assertEqual(film.coerce_params({"exposure_bias": 99})["exposure_bias"], 5.0)

    def test_lut_name_cannot_escape_the_lut_directory(self):
        for bad in ("../../etc/passwd", "a/b", "..", "", "with space", "semi;colon"):
            with self.assertRaises(ValueError, msg=f"{bad!r} should be rejected"):
                film.coerce_params({"lut": bad})

        for good in ("kodak_gold_200", "portra-400", "Cinestill800T"):
            self.assertEqual(film.coerce_params({"lut": good})["lut"], good)

    def test_non_string_lut_is_rejected(self):
        with self.assertRaises(ValueError):
            film.coerce_params({"lut": 7})

    def test_presets_carry_no_exposure_or_contrast(self):
        """The LUT carries the colour. Exposure is per-photo, and look-contrast
        baked into a preset would be counted twice against the CLUT."""
        for name, preset in film.BUILTIN_PRESETS.items():
            self.assertEqual(preset["exposure_bias"], 0.0, name)
            self.assertEqual(preset["contrast_strength"], 0.0, name)
            self.assertIn("lut", preset, name)
            # Presets must survive the boundary they are fed through unchanged.
            clean = film.coerce_params(preset)
            for key, value in preset.items():
                self.assertAlmostEqual(clean[key], value, msg=f"{name}.{key} was clamped")


class TestLegacyPresetUnits(unittest.TestCase):
    """grain_size and halation_radius used to be PIXEL counts.

    They are now fractions of an edge, capped at 0.05 and 0.10. A preset saved
    before that change (grain_size: 3, halation_radius: 38) does not fail on
    load — it CLAMPS, to the maximum of each, which on a 4000x3000 photo is a
    150px grain sigma and a 400px halation sigma. The render is destroyed and
    nothing says so.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.presets_file = Path(self.tmp.name) / "film_presets.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, presets):
        self.presets_file.write_text(json.dumps(presets), encoding="utf-8")

    def test_v1_pixel_values_fall_back_to_the_defaults_not_the_maxima(self):
        self._write({
            "Old Look": {
                "lut":                "kodak_gold_200",
                "grade_strength":     0.70,
                "grain_intensity":    0.06,
                "grain_size":         3,     # v1: PIXELS
                "halation_intensity": 0.50,
                "halation_radius":    38,    # v1: PIXELS
            }
        })

        with self.assertLogs("film", level="WARNING") as logs:
            loaded = film._load_user_presets(self.presets_file)

        params = loaded["Old Look"]

        # The old pixel count cannot be converted into a fraction without the
        # dimensions of the photo it was tuned on, so the honest answer is the
        # default — NOT the clamped maximum.
        self.assertEqual(params["grain_size"], film.DEFAULT_PARAMS["grain_size"])
        self.assertEqual(params["halation_radius"], film.DEFAULT_PARAMS["halation_radius"])

        # And it survives the boundary clamp, which is where the damage used to
        # be done.
        clean = film.coerce_params(params)
        self.assertEqual(clean["grain_size"], film.DEFAULT_PARAMS["grain_size"])
        self.assertEqual(clean["halation_radius"], film.DEFAULT_PARAMS["halation_radius"])
        self.assertNotEqual(clean["grain_size"], film.PARAM_SPEC["grain_size"][2])
        self.assertNotEqual(clean["halation_radius"], film.PARAM_SPEC["halation_radius"][2])

        # Everything the user chose that still means what it meant is untouched.
        self.assertEqual(params["grade_strength"], 0.70)
        self.assertEqual(params["grain_intensity"], 0.06)

        # Silently rewriting someone's preset is its own defect.
        self.assertTrue(any("grain_size" in line for line in logs.output))
        self.assertTrue(any("halation_radius" in line for line in logs.output))

    def test_current_presets_are_left_alone(self):
        current = {
            "New Look": {
                "lut":                "kodak_gold_200",
                "grade_strength":     0.85,
                "grain_intensity":    0.055,
                "grain_size":         0.0015,
                "halation_intensity": 0.45,
                "halation_radius":    0.010,
            }
        }
        self._write(current)

        loaded = film._load_user_presets(self.presets_file)

        self.assertEqual(loaded, current)


class TestFilmRoutes(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.temp_dir.name)
        self.presets_file = self.state_dir / "film_presets.json"

        self.flask_app = Flask(__name__)
        film.register_film_routes(self.flask_app, self.presets_file)
        self.client = self.flask_app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_presets_merges_builtin_and_user_presets(self):
        presets_file = self.state_dir / "film_presets.json"
        presets_file.write_text(json.dumps({"Custom Lab": {"grade_strength": 0.25}}), encoding="utf-8")

        response = self.client.get("/api/film/presets")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        names = {item["name"] for item in payload}
        self.assertIn("Ambient Film", names)
        self.assertIn("Flash Film", names)
        self.assertIn("Custom Lab", names)

        custom = next(item for item in payload if item["name"] == "Custom Lab")
        self.assertFalse(custom["builtin"])
        self.assertEqual(custom["params"]["grade_strength"], 0.25)

    def test_save_preset_rejects_builtin_name(self):
        response = self.client.post("/api/film/presets", json={
            "name": "Ambient Film",
            "params": {"grade_strength": 0.1},
        })

        self.assertEqual(response.status_code, 400)
        self.assertIn("Cannot overwrite", response.get_json()["error"])

    def test_save_and_delete_user_preset_round_trips(self):
        response = self.client.post("/api/film/presets", json={
            "name": "My Night Look",
            "params": {"grade_strength": 0.33, "grain_size": 0.002},
        })
        self.assertEqual(response.status_code, 200)

        presets_file = self.state_dir / "film_presets.json"
        saved = json.loads(presets_file.read_text(encoding="utf-8"))
        self.assertIn("My Night Look", saved)

        delete_response = self.client.delete("/api/film/presets/My%20Night%20Look")
        self.assertEqual(delete_response.status_code, 200)

        remaining = json.loads(presets_file.read_text(encoding="utf-8"))
        self.assertNotIn("My Night Look", remaining)

    def test_process_route_rejects_unsupported_extension(self):
        response = self.client.post(
            "/api/film/process",
            data={"file": (io.BytesIO(b"not-an-image"), "bad.txt")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.get_json()["error"])

    def test_process_route_returns_jpeg_attachment(self):
        with patch.object(film, "process_photo", return_value=b"\xff\xd8\xff\xd9") as mock_process:
            response = self.client.post(
                "/api/film/process",
                data={
                    "file": (io.BytesIO(b"fake-image-bytes"), "sample.jpg"),
                    "params": json.dumps({"grade_strength": 0.5}),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        self.assertIn("attachment", response.headers.get("Content-Disposition", ""))
        mock_process.assert_called_once()


if __name__ == "__main__":
    unittest.main()
