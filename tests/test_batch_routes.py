"""HTTP surface for batch processing."""

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from flask import Flask
from PIL import Image

import film


class TestBatchRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "in"
        self.dst = self.root / "out"
        self.src.mkdir()
        for n in ("a.jpg", "b.jpg"):
            Image.new("RGB", (16, 16), (120, 90, 60)).save(self.src / n)

        self.app = Flask(__name__)
        film.register_film_routes(self.app, self.root / "presets.json")
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def _drain(self, job_id, timeout=15.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.client.get(f"/api/film/batch/{job_id}").get_json()
            if state["status"] != "running":
                return state
            time.sleep(0.02)
        self.fail("batch did not finish in time")

    def test_start_processes_a_folder_end_to_end(self):
        r = self.client.post("/api/film/batch",
                             json={"source": str(self.src), "dest": str(self.dst),
                                   "params": {"grade_strength": 0.0}})
        self.assertEqual(r.status_code, 200)
        job_id = r.get_json()["id"]

        state = self._drain(job_id)

        self.assertEqual(state["status"], "done")
        self.assertEqual(state["done"], 2)
        self.assertTrue((self.dst / "a_film.jpg").exists())
        self.assertTrue((self.dst / "b_film.jpg").exists())

    def test_start_requires_json(self):
        r = self.client.post("/api/film/batch",
                             data="{}", content_type="text/plain")
        self.assertEqual(r.status_code, 415)

    def test_missing_folders_are_rejected(self):
        r = self.client.post("/api/film/batch", json={"source": str(self.src)})
        self.assertEqual(r.status_code, 400)

    def test_nonexistent_source_is_400(self):
        r = self.client.post("/api/film/batch",
                             json={"source": str(self.root / "nope"), "dest": str(self.dst)})
        self.assertEqual(r.status_code, 400)

    def test_bad_params_are_rejected_before_the_job_starts(self):
        r = self.client.post("/api/film/batch",
                             json={"source": str(self.src), "dest": str(self.dst),
                                   "params": {"grain_size": "abc"}})
        self.assertEqual(r.status_code, 400)

    def test_status_of_unknown_job_is_404(self):
        r = self.client.get("/api/film/batch/job999")
        self.assertEqual(r.status_code, 404)

    def test_cancel_actually_stops_a_running_job(self):
        # Hold each render on a gate so the job is genuinely mid-run when we cancel.
        # process_photo is resolved as a module global at call time, so patching it
        # reaches the batch worker's closure.
        gate = threading.Event()
        for i in range(20):
            Image.new("RGB", (16, 16)).save(self.src / f"x{i:02d}.jpg")

        def blocking(path, params):
            gate.wait(3.0)
            return b"\xff\xd8\xff\xd9"

        with patch.object(film, "process_photo", side_effect=blocking):
            r = self.client.post("/api/film/batch",
                                 json={"source": str(self.src), "dest": str(self.dst)})
            job_id = r.get_json()["id"]
            time.sleep(0.2)  # let the worker enter the first render
            c = self.client.post(f"/api/film/batch/{job_id}/cancel")
            self.assertEqual(c.status_code, 200)
            self.assertTrue(c.get_json()["cancelled"])
            gate.set()
            state = self._drain(job_id)

        self.assertEqual(state["status"], "cancelled")
        self.assertLess(state["done"] + state["skipped"], state["total"])

    def test_dest_that_is_an_existing_file_is_400(self):
        dest_file = self.root / "outfile"
        dest_file.write_bytes(b"file")
        r = self.client.post("/api/film/batch",
                             json={"source": str(self.src), "dest": str(dest_file)})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
