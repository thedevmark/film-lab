import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ModuleNotFoundError as exc:
    np = None
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None

try:
    from flask import Flask
except ModuleNotFoundError as exc:
    Flask = None
    FLASK_IMPORT_ERROR = exc
else:
    FLASK_IMPORT_ERROR = None

try:
    from PIL import Image
except ModuleNotFoundError as exc:
    Image = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None

try:
    import film
except Exception as exc:
    film = None
    FILM_IMPORT_ERROR = exc
else:
    FILM_IMPORT_ERROR = None


FILM_MATH_READY = film is not None and np is not None and Image is not None
FILM_ROUTE_READY = film is not None and Flask is not None


def _skip_reason(*errors):
    for error in errors:
        if error is not None:
            return str(error)
    return "required Film Lab test dependencies are unavailable"


def _solid_hald(path: Path, colour):
    """An 8x8 PNG is a valid level-2 HaldCLUT (2**3 == 8 px, 4**3 == 64 entries).

    A solid one maps every input colour to `colour`, which makes it trivial to
    tell two LUTs apart without caring what either of them does.
    """
    Image.new("RGB", (8, 8), colour).save(path)


@unittest.skipUnless(FILM_MATH_READY, _skip_reason(FILM_IMPORT_ERROR, NUMPY_IMPORT_ERROR, PIL_IMPORT_ERROR))
class TestPipeline(unittest.TestCase):
    def setUp(self):
        film._LUT_CACHE.clear()

    def tearDown(self):
        film._LUT_CACHE.clear()

    def test_pipeline_runs_in_the_documented_order(self):
        from filmlab import loader

        order = []
        source = np.full((8, 8, 3), 0.25, dtype=np.float32)

        def record(name, fn):
            def inner(*args, **kwargs):
                order.append(name)
                return fn(*args, **kwargs)
            return inner

        with patch.object(film, "load_image",
                          return_value=(source.copy(), loader.DISPLAY)), \
             patch.object(film, "apply_exposure",
                          side_effect=record("exposure", lambda img, ev: img)), \
             patch.object(film, "add_halation",
                          side_effect=record("halation", lambda img, **k: img)), \
             patch.object(film, "apply_lut",
                          side_effect=record("lut", lambda img, cube, strength=1.0: img)), \
             patch.object(film, "apply_contrast",
                          side_effect=record("contrast", lambda img, strength: img)), \
             patch.object(film, "add_grain",
                          side_effect=record("grain", lambda img, **k: img)):
            out = film.process_photo(Path("x.jpg"), {"grain_intensity": 0.05})

        # Contrast is AFTER the LUT: the CLUTs were authored against a neutral,
        # standard-contrast render, so look-contrast before them is counted
        # twice. Grain is LAST: it is the texture on a finished frame, not a
        # signal that gets graded.
        self.assertEqual(order, ["exposure", "halation", "lut", "contrast", "grain"])
        self.assertTrue(out.startswith(b"\xff\xd8"))

        decoded = Image.open(io.BytesIO(out))
        self.assertEqual(decoded.format, "JPEG")
        self.assertEqual(decoded.size, (8, 8))

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
            film.process_photo(Path("x.jpg"), {
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
                film.process_photo(Path("x.jpg"), params)
            with patch.object(film, "load_image",
                              return_value=(source.copy(), loader.SCENE)):
                film.process_photo(Path("x.arw"), params)

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


@unittest.skipUnless(FILM_MATH_READY, _skip_reason(FILM_IMPORT_ERROR, NUMPY_IMPORT_ERROR, PIL_IMPORT_ERROR))
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


@unittest.skipUnless(film is not None, _skip_reason(FILM_IMPORT_ERROR))
class TestParams(unittest.TestCase):
    def test_defaults_cover_every_spec_key_plus_the_lut(self):
        clean = film.coerce_params({})

        for key in film.PARAM_SPEC:
            self.assertIn(key, clean)
        self.assertEqual(clean["lut"], "kodak_gold_200")

    def test_grain_size_and_halation_radius_are_fractions_now(self):
        clean = film.coerce_params({"grain_size": 3, "halation_radius": 38})

        # The old pixel-valued sliders must clamp into the fraction range rather
        # than being taken at face value.
        self.assertLessEqual(clean["grain_size"], 0.05)
        self.assertLessEqual(clean["halation_radius"], 0.10)

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


@unittest.skipUnless(FILM_ROUTE_READY, _skip_reason(FILM_IMPORT_ERROR, FLASK_IMPORT_ERROR))
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
