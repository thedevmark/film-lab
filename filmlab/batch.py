"""Folder-in, folder-out batch processing.

One job at a time, one worker thread. Sequential by design: each photo already
holds several full-resolution float32 buffers, so a thread pool would multiply
peak memory for a wall-clock win that does not matter on an overnight run.

Resumable: an output that already exists is skipped, so a cancelled or crashed
run continues where it stopped rather than redoing finished work. That also makes
dest == source safe — a re-run sees the previous outputs, recognises them as
already done, and moves on.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger("filmlab.batch")

# Outputs are named <stem>_film.jpg. The suffix is load-bearing: it is how a
# re-run tells a previous output from a fresh input when the two share a folder.
OUTPUT_SUFFIX = "_film.jpg"

# Work-in-progress lands here first, then gets renamed onto the real name. A
# crash mid-render leaves a .part, which _list_files ignores and the next run
# retries — the final name only ever appears complete.
PART_SUFFIX = ".part"

# A processor takes (source_file, output_file, params) and writes the output.
Processor = Callable[[Path, Path, dict], None]


@dataclass
class JobState:
    id: str
    source: str
    dest: str
    total: int = 0
    done: int = 0        # newly processed this run
    skipped: int = 0     # output already existed
    failed: int = 0      # raised, and were logged and passed over
    current: str = ""    # the file in flight, "" when idle
    status: str = "running"   # running | done | cancelled | error
    error: str = ""


class BatchManager:
    """Runs one batch job at a time on a background worker thread."""

    def __init__(self, extensions, process_fn: Processor):
        self._extensions = {e.lower() for e in extensions}
        self._process_fn = process_fn
        self._lock = threading.Lock()
        self._job: JobState | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._counter = 0

    # ── Listing ──────────────────────────────────────────────────────────────

    def _list_files(self, source: Path):
        """Top-level supported files, excluding our own previous outputs.

        Non-recursive on purpose: a shoot folder is flat, and a dest subfolder is
        then naturally ignored rather than fed back in as input.
        """
        return sorted(
            p for p in source.iterdir()
            if p.is_file()
            and p.suffix.lower() in self._extensions
            and not p.name.endswith(OUTPUT_SUFFIX)
        )

    # ── Control ──────────────────────────────────────────────────────────────

    def start(self, source: Path, dest: Path, params: dict) -> JobState:
        source, dest = Path(source), Path(dest)
        if not source.is_dir():
            raise ValueError("Source folder does not exist.")

        try:
            files = self._list_files(source)
        except OSError as exc:
            # e.g. a permission error or a disconnected network path. A clean 400,
            # not an unhandled 500.
            raise ValueError("Source folder could not be read.") from exc

        with self._lock:
            if self._job is not None and self._job.status == "running":
                raise RuntimeError("A batch is already running.")
            self._counter += 1
            job = JobState(
                id=f"job{self._counter}",
                source=str(source),
                dest=str(dest),
                total=len(files),
            )
            self._job = job
            self._cancel.clear()

        try:
            dest.mkdir(parents=True, exist_ok=True)
        except (OSError, FileExistsError) as exc:
            # e.g. dest is an existing file. Fail the request, not mid-run.
            with self._lock:
                self._job = None
            raise ValueError("Output folder could not be created.") from exc

        self._thread = threading.Thread(
            target=self._run, args=(files, dest, params, job), daemon=True
        )
        self._thread.start()
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if self._job and self._job.id == job_id and self._job.status == "running":
                self._cancel.set()
                return True
        return False

    def get(self, job_id: str):
        with self._lock:
            if self._job and self._job.id == job_id:
                return asdict(self._job)
        return None

    # ── Worker ───────────────────────────────────────────────────────────────

    def _run(self, files, dest: Path, params: dict, job: JobState):
        try:
            for src in files:
                if self._cancel.is_set():
                    break
                job.current = src.name
                out = dest / (src.stem + OUTPUT_SUFFIX)
                if out.exists():
                    job.skipped += 1
                    continue
                # Render to a .part and rename onto the real name. os.replace is
                # atomic on the same volume, so a crash or a full disk mid-write
                # can never leave a truncated output that a later run would skip
                # as "already done".
                part = out.with_name(out.name + PART_SUFFIX)
                try:
                    self._process_fn(src, part, params)
                    os.replace(part, out)
                    job.done += 1
                except Exception:
                    # One unreadable or corrupt file must not sink the whole run.
                    self._discard(part)
                    job.failed += 1
                    logger.exception("Batch: failed on %s", src.name)
            job.status = "cancelled" if self._cancel.is_set() else "done"
        except Exception:
            logger.exception("Batch: run aborted")
            job.status = "error"
            job.error = "The batch stopped unexpectedly."
        finally:
            job.current = ""

    @staticmethod
    def _discard(part: Path):
        try:
            part.unlink(missing_ok=True)
        except OSError:
            pass
