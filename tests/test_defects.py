"""Regression tests for defects found in adversarial review.

Each test here failed before its fix. They are separated from test_film.py so
the thing each one pins down stays legible.
"""

import io
import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from flask import Flask
from PIL import Image

import film


class TestHalationPrecision(unittest.TestCase):
    """add_halation used to quantize the bloom to uint8 *before* blurring.

    A Gaussian spreads a highlight's energy over ~radius^2 px, so a small
    highlight's blurred peak fell below 1/255 and rounded to zero — producing
    no halation at all for catchlights, streetlamps, and sun sparkle, which is
    the entire reason the feature exists.
    """

    def test_small_highlight_still_produces_halation(self):
        img = np.zeros((128, 128, 3), dtype=np.float32)
        img[63:65, 63:65] = 1.0  # 2px specular highlight

        out = film.add_halation(img, intensity=0.5, radius=25)
        bloom = out - img

        self.assertGreater(
            float(bloom[:, :, 0].sum()), 0.0,
            "a 2px highlight produced zero red halation — bloom was quantized away",
        )

    def test_bloom_is_not_posterized(self):
        img = np.zeros((128, 128, 3), dtype=np.float32)
        img[60:68, 60:68] = 1.0

        out = film.add_halation(img, intensity=0.5, radius=38)
        levels = np.unique(np.round((out - img)[:, :, 0], 6))

        # The uint8 path collapsed the halo to a handful of levels, which reads
        # as visible banding rings.
        self.assertGreater(len(levels), 32, f"bloom posterized to {len(levels)} levels")

    def test_halation_is_red_dominant_and_leaves_blue_alone(self):
        img = np.zeros((64, 64, 3), dtype=np.float32)
        img[32, 32] = 1.0

        bloom = film.add_halation(img, intensity=0.8, radius=8) - img

        self.assertGreater(float(bloom[:, :, 0].sum()), float(bloom[:, :, 1].sum()))
        self.assertAlmostEqual(float(bloom[:, :, 2].sum()), 0.0, places=6)


class TestExifOrientation(unittest.TestCase):
    """Portrait photos came back rotated 90 degrees.

    Cameras store a vertically-held shot in landscape layout plus an
    Orientation tag. _load_image never applied it, and the JPEG was written
    with no EXIF, so the compensating tag was gone too.
    """

    def test_portrait_photo_is_uprighted_on_load(self):
        # 40 wide x 20 tall stored, Orientation=6 => displays as 20x40 portrait.
        pil = Image.new("RGB", (40, 20), (128, 64, 32))
        exif = pil.getexif()
        exif[274] = 6  # Orientation: rotate 90 CW

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "portrait.jpg"
            pil.save(path, exif=exif)

            arr = film._load_image(path)

        self.assertEqual(
            arr.shape[:2], (40, 20),
            "EXIF orientation ignored — a portrait photo loaded sideways",
        )


class TestParamValidation(unittest.TestCase):
    """Unvalidated params reached numpy and the filesystem.

    grain_size floor-divides the image down to 1x1 and then np.repeat expands
    back up by `size`, so the intermediate is size x size x 3 regardless of the
    input image — 100000 asked for 112 GiB from a 64x64 photo.
    """

    def test_absurd_grain_size_does_not_allocate_the_universe(self):
        img = np.full((64, 64, 3), 0.5, dtype=np.float32)

        out = film.add_grain(img, intensity=0.05, size=100000)

        self.assertEqual(out.shape, img.shape)

    def test_coerce_params_clamps_and_rejects(self):
        clean = film.coerce_params({"grain_size": 100000, "halation_radius": -5})
        self.assertLessEqual(clean["grain_size"], 64)
        self.assertGreaterEqual(clean["halation_radius"], 0)

        for bad in ({"exposure_bias": None}, {"grain_size": "abc"}, {"grain_size": 1e999}):
            with self.assertRaises(ValueError, msg=f"{bad!r} should be rejected"):
                film.coerce_params(bad)

    def test_non_dict_params_rejected(self):
        with self.assertRaises(ValueError):
            film.coerce_params([1, 2, 3])


class TestPresetPersistence(unittest.TestCase):
    """The presets file could be silently wiped, and the API reported success.

    _save_user_presets truncated then wrote, non-atomically. A torn write left
    invalid JSON; _load_user_presets swallowed the parse error and returned {};
    the next save wrote that empty dict back over the file.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.presets_file = Path(self.tmp.name) / "film_presets.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_corrupt_file_is_not_silently_overwritten(self):
        self.presets_file.write_text("{ this is not json", encoding="utf-8")

        with self.assertRaises(Exception):
            film._load_user_presets(self.presets_file, tolerate_corrupt=False)

    def test_concurrent_saves_do_not_lose_updates(self):
        app = Flask(__name__)
        film.register_film_routes(app, self.presets_file)
        client_lock = threading.Lock()
        errors = []

        def save(i):
            try:
                with client_lock:
                    c = app.test_client()
                c.post("/api/film/presets",
                       json={"name": f"p{i}", "params": {"grain_size": 3}})
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=save, args=(i,)) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        saved = json.loads(self.presets_file.read_text(encoding="utf-8"))
        self.assertEqual(len(saved), 40, f"lost updates: only {len(saved)}/40 persisted")


class TestRouteHardening(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.presets_file = Path(self.tmp.name) / "film_presets.json"
        self.app = Flask(__name__)
        film.register_film_routes(self.app, self.presets_file)
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def test_bad_params_are_400_not_500(self):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="JPEG")
        buf.seek(0)

        r = self.client.post("/api/film/process", data={
            "file": (buf, "x.jpg"),
            "params": json.dumps({"grain_size": "abc"}),
        }, content_type="multipart/form-data")

        self.assertEqual(r.status_code, 400)

    def test_internal_errors_do_not_leak_paths(self):
        r = self.client.post("/api/film/process", data={
            "file": (io.BytesIO(b"not really a jpeg"), "x.jpg"),
        }, content_type="multipart/form-data")

        body = r.get_data(as_text=True)
        self.assertNotIn("Temp", body)
        self.assertNotIn("Users", body)
        self.assertNotIn(tempfile.gettempdir(), body)

    def test_preset_write_requires_json_content_type(self):
        """get_json(force=True) parsed any Content-Type, so a text/plain
        cross-origin POST was a CORS simple request with no preflight — any
        site could write presets while Film Lab ran on localhost."""
        r = self.client.post("/api/film/presets",
                             data=json.dumps({"name": "csrf", "params": {}}),
                             content_type="text/plain")

        self.assertEqual(r.status_code, 415)

    def test_preset_name_with_slash_is_rejected(self):
        r = self.client.post("/api/film/presets",
                             json={"name": "a/b", "params": {"grain_size": 3}})

        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
