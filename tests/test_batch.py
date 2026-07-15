"""Batch processing: folder in, folder out, resumable, cancellable."""

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from filmlab.batch import BatchManager, OUTPUT_SUFFIX


def _touch(directory: Path, name: str) -> Path:
    p = directory / name
    p.write_bytes(b"x")
    return p


class TestListing(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.src = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _manager(self, process_fn=lambda s, o, p: o.write_bytes(b"out")):
        return BatchManager({".jpg", ".arw"}, process_fn)

    def test_only_supported_extensions_are_listed(self):
        _touch(self.src, "a.jpg")
        _touch(self.src, "b.arw")
        _touch(self.src, "notes.txt")
        _touch(self.src, "c.png")  # not in this manager's set

        files = self._manager()._list_files(self.src)

        self.assertEqual({p.name for p in files}, {"a.jpg", "b.arw"})

    def test_previous_outputs_are_not_re_listed(self):
        """An output that lands next to its input (dest == source) must not be
        picked up as a new input on the next run."""
        _touch(self.src, "a.jpg")
        _touch(self.src, f"a{OUTPUT_SUFFIX}")

        files = self._manager()._list_files(self.src)

        self.assertEqual({p.name for p in files}, {"a.jpg"})

    def test_walk_is_non_recursive_so_a_dest_subfolder_is_ignored(self):
        _touch(self.src, "a.jpg")
        sub = self.src / "_film"
        sub.mkdir()
        _touch(sub, "a.jpg")

        files = self._manager()._list_files(self.src)

        self.assertEqual([p.name for p in files], ["a.jpg"])


class TestJobLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "in"
        self.dst = self.root / "out"
        self.src.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _wait(self, manager, job_id, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = manager.get(job_id)
            if state and state["status"] != "running":
                return state
            time.sleep(0.01)
        self.fail("job did not finish in time")

    def test_processes_every_file_and_writes_outputs(self):
        for n in ("a.jpg", "b.jpg", "c.jpg"):
            _touch(self.src, n)
        seen = []
        mgr = BatchManager({".jpg"}, lambda s, o, p: (seen.append(s.name), o.write_bytes(b"out")))

        job = mgr.start(self.src, self.dst, {})
        state = self._wait(mgr, job.id)

        self.assertEqual(state["status"], "done")
        self.assertEqual(state["done"], 3)
        self.assertEqual(sorted(seen), ["a.jpg", "b.jpg", "c.jpg"])
        self.assertTrue((self.dst / f"a{OUTPUT_SUFFIX}").exists())

    def test_dest_is_created_if_absent(self):
        _touch(self.src, "a.jpg")
        mgr = BatchManager({".jpg"}, lambda s, o, p: o.write_bytes(b"out"))

        job = mgr.start(self.src, self.dst, {})
        self._wait(mgr, job.id)

        self.assertTrue(self.dst.is_dir())

    def test_already_done_files_are_skipped_so_a_run_resumes(self):
        for n in ("a.jpg", "b.jpg", "c.jpg"):
            _touch(self.src, n)
        self.dst.mkdir()
        (self.dst / f"a{OUTPUT_SUFFIX}").write_bytes(b"done earlier")
        processed = []
        mgr = BatchManager({".jpg"}, lambda s, o, p: (processed.append(s.name), o.write_bytes(b"out")))

        job = mgr.start(self.src, self.dst, {})
        state = self._wait(mgr, job.id)

        self.assertEqual(state["skipped"], 1)
        self.assertEqual(state["done"], 2)
        self.assertNotIn("a.jpg", processed)  # the finished one was not redone
        self.assertEqual((self.dst / f"a{OUTPUT_SUFFIX}").read_bytes(), b"done earlier")

    def test_one_bad_file_does_not_kill_the_run(self):
        for n in ("a.jpg", "bad.jpg", "c.jpg"):
            _touch(self.src, n)

        def process(s, o, p):
            if s.name == "bad.jpg":
                raise RuntimeError("boom")
            o.write_bytes(b"out")

        mgr = BatchManager({".jpg"}, process)
        job = mgr.start(self.src, self.dst, {})
        state = self._wait(mgr, job.id)

        self.assertEqual(state["status"], "done")
        self.assertEqual(state["done"], 2)
        self.assertEqual(state["failed"], 1)

    def test_cancel_stops_further_processing(self):
        for i in range(50):
            _touch(self.src, f"{i:03d}.jpg")
        gate = threading.Event()
        started = threading.Event()

        def process(s, o, p):
            started.set()
            gate.wait(2.0)  # hold the first file so we can cancel mid-run
            o.write_bytes(b"out")

        mgr = BatchManager({".jpg"}, process)
        job = mgr.start(self.src, self.dst, {})
        started.wait(2.0)
        self.assertTrue(mgr.cancel(job.id))
        gate.set()
        state = self._wait(mgr, job.id)

        self.assertEqual(state["status"], "cancelled")
        self.assertLess(state["done"], 50)  # it stopped early

    def test_rejects_a_source_that_is_not_a_directory(self):
        mgr = BatchManager({".jpg"}, lambda s, o, p: None)

        with self.assertRaises(ValueError):
            mgr.start(self.root / "does_not_exist", self.dst, {})

    def test_only_one_job_runs_at_a_time(self):
        for i in range(20):
            _touch(self.src, f"{i:03d}.jpg")
        gate = threading.Event()
        mgr = BatchManager({".jpg"}, lambda s, o, p: gate.wait(2.0) or o.write_bytes(b"out"))

        job = mgr.start(self.src, self.dst, {})
        with self.assertRaises(RuntimeError):
            mgr.start(self.src, self.dst, {})
        gate.set()
        self._wait(mgr, job.id)

    def test_a_new_job_can_start_once_the_previous_finishes(self):
        _touch(self.src, "a.jpg")
        mgr = BatchManager({".jpg"}, lambda s, o, p: o.write_bytes(b"out"))

        first = mgr.start(self.src, self.dst, {})
        self._wait(mgr, first.id)
        second = mgr.start(self.src, self.dst / "again", {})
        state = self._wait(mgr, second.id)

        self.assertNotEqual(first.id, second.id)
        self.assertEqual(state["status"], "done")

    def test_get_unknown_job_is_none(self):
        mgr = BatchManager({".jpg"}, lambda s, o, p: None)
        self.assertIsNone(mgr.get("nope"))

    def test_a_failed_render_leaves_no_skippable_partial(self):
        """The whole point of 'resumable'. A render that dies after writing some
        bytes must NOT leave a file at the final name — otherwise the next run
        skips it as done and a truncated output ships forever."""
        _touch(self.src, "a.jpg")

        def write_then_die(s, out, p):
            out.write_bytes(b"HALF A JPEG")  # partial output to the path we were given
            raise RuntimeError("disk full")

        mgr = BatchManager({".jpg"}, write_then_die)
        job = mgr.start(self.src, self.dst, {})
        state = self._wait(mgr, job.id)

        self.assertEqual(state["failed"], 1)
        self.assertFalse((self.dst / f"a{OUTPUT_SUFFIX}").exists(),
                         "a failed render left a partial output at the final name")

    def test_a_failed_file_is_retried_on_the_next_run(self):
        _touch(self.src, "a.jpg")
        attempts = []

        def flaky(s, out, p):
            attempts.append(1)
            if len(attempts) == 1:
                out.write_bytes(b"partial")
                raise RuntimeError("transient")
            out.write_bytes(b"good")

        mgr = BatchManager({".jpg"}, flaky)
        self._wait(mgr, mgr.start(self.src, self.dst, {}).id)
        state = self._wait(mgr, mgr.start(self.src, self.dst, {}).id)  # second run

        self.assertEqual(state["done"], 1)  # retried, not skipped
        self.assertEqual((self.dst / f"a{OUTPUT_SUFFIX}").read_bytes(), b"good")

    def test_a_successful_render_leaves_no_part_file(self):
        _touch(self.src, "a.jpg")
        mgr = BatchManager({".jpg"}, lambda s, out, p: out.write_bytes(b"out"))
        self._wait(mgr, mgr.start(self.src, self.dst, {}).id)

        leftovers = [p.name for p in self.dst.iterdir() if p.name.endswith(".part")]
        self.assertEqual(leftovers, [])

    def test_unreadable_source_is_a_value_error_not_a_crash(self):
        from unittest.mock import patch
        mgr = BatchManager({".jpg"}, lambda s, o, p: None)

        with patch.object(Path, "iterdir", side_effect=PermissionError("locked")):
            with self.assertRaises(ValueError):
                mgr.start(self.src, self.dst, {})

    def test_dest_that_is_an_existing_file_is_rejected(self):
        _touch(self.src, "a.jpg")
        dest_file = self.root / "out.jpg"
        dest_file.write_bytes(b"i am a file")
        mgr = BatchManager({".jpg"}, lambda s, o, p: o.write_bytes(b"x"))

        with self.assertRaises(ValueError):
            mgr.start(self.src, dest_file, {})


if __name__ == "__main__":
    unittest.main()
