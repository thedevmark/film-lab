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

from flask import Flask
from PIL import Image

import film


class TestParamValidation(unittest.TestCase):
    """Unvalidated params reached numpy and the filesystem.

    Both grain_size and halation_radius scale a Gaussian kernel, so an
    unbounded value asks for arbitrary work from an arbitrarily small photo.
    They are clamped at the boundary, before any of them reach numpy.
    """

    def test_coerce_params_clamps_and_rejects(self):
        clean = film.coerce_params({"grain_size": 100000, "halation_radius": -5})
        self.assertLessEqual(clean["grain_size"], 0.05)
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
