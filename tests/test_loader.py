import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from filmlab import loader


class TestLoadImage(unittest.TestCase):
    def _jpeg(self, directory, size=(40, 20), colour=(128, 64, 32), orientation=None):
        pil = Image.new("RGB", size, colour)
        path = Path(directory) / "photo.jpg"
        if orientation is not None:
            exif = pil.getexif()
            exif[274] = orientation
            pil.save(path, exif=exif)
        else:
            pil.save(path)
        return path

    def test_rendered_input_is_tagged_display(self):
        with tempfile.TemporaryDirectory() as d:
            arr, state = loader.load_image(self._jpeg(d))

        self.assertEqual(state, loader.DISPLAY)

    def test_rendered_input_is_linearised(self):
        """Mid-grey sRGB 0.5 must come back as linear ~0.214, not 0.5."""
        with tempfile.TemporaryDirectory() as d:
            path = self._jpeg(d, colour=(128, 128, 128))
            arr, _ = loader.load_image(path)

        self.assertAlmostEqual(float(arr.mean()), 0.2158, delta=0.01)

    def test_exif_orientation_is_applied(self):
        """A vertically-held shot is stored landscape plus Orientation=6."""
        with tempfile.TemporaryDirectory() as d:
            path = self._jpeg(d, size=(40, 20), orientation=6)
            arr, _ = loader.load_image(path)

        self.assertEqual(arr.shape[:2], (40, 20))

    def test_downscales_past_the_long_edge_cap(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._jpeg(d, size=(800, 400))
            arr, _ = loader.load_image(path, max_long_edge=100)

        self.assertEqual(max(arr.shape[:2]), 100)

    def test_extreme_aspect_ratio_does_not_collapse_to_zero(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "wide.png"
            Image.new("RGB", (2000, 2)).save(path)

            arr, _ = loader.load_image(path, max_long_edge=100)

        self.assertGreaterEqual(arr.shape[0], 1)
        self.assertGreaterEqual(arr.shape[1], 1)

    def test_output_is_float32_in_zero_one(self):
        with tempfile.TemporaryDirectory() as d:
            arr, _ = loader.load_image(self._jpeg(d))

        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.all((arr >= 0.0) & (arr <= 1.0)))


class _FakeRawContext:
    """Stands in for the object `rawpy.imread(...)` hands back as a context
    manager. Records the kwargs passed to postprocess() and returns a
    caller-supplied array, so a test can assert on exactly what the loader
    asked rawpy to do without needing a real RAW file fixture."""

    def __init__(self, array, capture):
        self._array = array
        self._capture = capture

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def postprocess(self, **kwargs):
        self._capture.append(kwargs)
        return self._array


class TestLoadRawImage(unittest.TestCase):
    """The RAW/SCENE path is the one the whole task exists to fix, and PIL
    cannot synthesize a real RAW fixture, so it is pinned by monkeypatching
    `rawpy` itself.

    If load_image ever regressed to rawpy.postprocess's defaults (BT.709
    gamma, auto-brightness, 8-bit output) -- the exact bug this task fixes --
    test_postprocess_is_called_with_scene_linear_arguments would catch it by
    inspecting the literal kwargs passed, not by rendering anything.
    """

    def _load_with_fake_rawpy(self, array, max_long_edge=6000):
        capture = []
        fake_module = types.ModuleType("rawpy")
        fake_module.imread = lambda path: _FakeRawContext(array, capture)

        with tempfile.TemporaryDirectory() as d:
            raw_path = Path(d) / "photo.arw"
            raw_path.write_bytes(b"not a real raw file -- rawpy.imread is mocked")
            with mock.patch.dict(sys.modules, {"rawpy": fake_module}):
                arr, state = loader.load_image(raw_path, max_long_edge=max_long_edge)

        return arr, state, capture

    def test_postprocess_is_called_with_scene_linear_arguments(self):
        array = np.full((8, 16, 3), 32768, dtype=np.uint16)
        _, _, capture = self._load_with_fake_rawpy(array)

        self.assertEqual(len(capture), 1, "postprocess must be called exactly once")
        kwargs = capture[0]
        self.assertEqual(
            kwargs.get("gamma"), (1, 1),
            "gamma must be (1,1) -- the default applies a BT.709 gamma",
        )
        self.assertTrue(
            kwargs.get("no_auto_bright"),
            "no_auto_bright must be True -- the default stretches brightness automatically",
        )
        self.assertEqual(
            kwargs.get("output_bps"), 16,
            "output_bps must be 16 -- 8-bit output clips highlight headroom",
        )

    def test_raw_result_is_tagged_scene(self):
        array = np.full((8, 16, 3), 32768, dtype=np.uint16)
        _, state, _ = self._load_with_fake_rawpy(array)

        self.assertEqual(state, loader.SCENE)

    def test_raw_output_is_normalised_by_65535(self):
        """A wrong divisor (e.g. 255, the 8-bit constant) would leave every
        raw-derived value far outside [0, 1]."""
        array = np.full((4, 4, 3), 32768, dtype=np.uint16)
        arr, _, _ = self._load_with_fake_rawpy(array)

        self.assertEqual(arr.dtype, np.float32)
        self.assertAlmostEqual(float(arr.mean()), 32768 / 65535, places=4)

    def test_raw_downscales_past_the_long_edge_cap(self):
        """Also pins that a (H, W, 3) uint16 array survives the resize step:
        PIL has no native 3-channel 16-bit mode, only single-channel, so this
        path cannot reuse the 8-bit _fit() helper unmodified."""
        array = np.full((200, 400, 3), 40000, dtype=np.uint16)
        arr, _, _ = self._load_with_fake_rawpy(array, max_long_edge=100)

        self.assertEqual(max(arr.shape[:2]), 100)

    def test_missing_rawpy_raises_runtime_error(self):
        with tempfile.TemporaryDirectory() as d:
            raw_path = Path(d) / "photo.arw"
            raw_path.write_bytes(b"placeholder")
            with mock.patch.dict(sys.modules, {"rawpy": None}):
                with self.assertRaises(RuntimeError):
                    loader.load_image(raw_path)


if __name__ == "__main__":
    unittest.main()
