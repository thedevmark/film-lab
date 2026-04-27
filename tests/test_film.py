import io
import json
import sys
import tempfile
import types
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
    return "required Film Studio test dependencies are unavailable"


@unittest.skipUnless(FILM_MATH_READY, _skip_reason(FILM_IMPORT_ERROR, NUMPY_IMPORT_ERROR, PIL_IMPORT_ERROR))
class TestFilmMath(unittest.TestCase):
    def test_apply_exposure_clips_into_valid_range(self):
        img = np.array([[[0.05, 0.5, 0.95]]], dtype=np.float32)

        brighter = film.apply_exposure(img, 0.10)
        darker = film.apply_exposure(img, -0.20)

        np.testing.assert_allclose(brighter, np.array([[[0.15, 0.60, 1.0]]], dtype=np.float32))
        np.testing.assert_allclose(darker, np.array([[[0.0, 0.30, 0.75]]], dtype=np.float32))

    def test_apply_film_color_strength_zero_is_identity(self):
        img = np.array([[[0.42, 0.38, 0.51], [0.20, 0.25, 0.30]]], dtype=np.float32)

        out = film.apply_film_color(img, 0.0)

        np.testing.assert_allclose(out, img)

    def test_apply_film_color_pushes_midtones_warmer(self):
        img = np.full((4, 4, 3), [0.42, 0.38, 0.40], dtype=np.float32)

        out = film.apply_film_color(img, 1.0)

        self.assertGreater(float(out[:, :, 0].mean()), float(img[:, :, 0].mean()))
        self.assertLess(float(out[:, :, 2].mean()), float(img[:, :, 2].mean()))
        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))

    def test_add_grain_zero_intensity_is_identity(self):
        img = np.full((6, 6, 3), 0.5, dtype=np.float32)

        out = film.add_grain(img, 0.0, 3)

        np.testing.assert_allclose(out, img)

    def test_add_halation_boosts_red_highlight_bloom(self):
        img = np.zeros((9, 9, 3), dtype=np.float32)
        img[4, 4] = [1.0, 1.0, 1.0]

        out = film.add_halation(img, intensity=0.75, radius=2)
        delta = out - img

        self.assertGreater(float(delta[:, :, 0].sum()), float(delta[:, :, 1].sum()))
        self.assertGreater(float(delta[:, :, 1].sum()), float(delta[:, :, 2].sum()))
        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))

    def test_process_photo_applies_pipeline_in_expected_order(self):
        order = []
        input_path = Path("dummy.jpg")
        source = np.full((4, 4, 3), 0.5, dtype=np.float32)

        def _record(name, delta):
            def inner(img, value):
                order.append(name)
                return np.clip(img + delta, 0.0, 1.0)
            return inner

        with patch.object(film, "_load_image", return_value=source.copy()), \
             patch.object(film, "apply_exposure", side_effect=_record("exposure", 0.01)), \
             patch.object(film, "apply_film_color", side_effect=_record("grade", 0.01)), \
             patch.object(film, "apply_contrast", side_effect=_record("contrast", 0.01)), \
             patch.object(film, "add_halation", side_effect=lambda img, intensity, radius: (order.append("halation"), np.clip(img + 0.01, 0.0, 1.0))[1]), \
             patch.object(film, "add_grain", side_effect=lambda img, intensity, size: (order.append("grain"), np.clip(img + 0.01, 0.0, 1.0))[1]):
            output_bytes = film.process_photo(input_path, {
                "exposure_bias": 0.1,
                "grade_strength": 0.8,
                "contrast_strength": 0.2,
                "halation_intensity": 0.4,
                "halation_radius": 12,
                "grain_intensity": 0.05,
                "grain_size": 3,
            })

        self.assertEqual(order, ["exposure", "grade", "contrast", "halation", "grain"])
        self.assertTrue(output_bytes.startswith(b"\xff\xd8"))

        out = Image.open(io.BytesIO(output_bytes))
        self.assertEqual(out.format, "JPEG")
        self.assertEqual(out.size, (4, 4))


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
            "params": {"grade_strength": 0.33, "grain_size": 4},
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
