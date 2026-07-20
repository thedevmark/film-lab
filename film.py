"""Film Lab pipeline — photo style processing with a measured film LUT, grain, and halation."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import sys
import tempfile
import threading
from dataclasses import asdict as _dataclass_dict
from pathlib import Path

import numpy as np
from PIL import Image
from flask import request, jsonify, send_file

from filmlab.batch import BatchManager
from filmlab.effects import add_grain, add_halation
from filmlab.loader import IMAGE_EXTENSIONS, RAW_EXTENSIONS, SCENE, load_image
from filmlab.lut import apply_lut, identity_cube, load_hald
from filmlab.tone import highlight_rolloff, hue_preserving_clip, srgb_encode

logger = logging.getLogger(__name__)

# ── Built-in presets ─────────────────────────────────────────────────────────
# Built-ins live in code only — cannot be overwritten by the user.
#
# These are aesthetic starting points, not measurements.
#
# An earlier version of this file claimed they were "derived from a scan of 45
# shoot folders / 2,602 raw-to-edited pairs". That claim did not survive review:
# mean luminance, per-channel means, saturation and shadow percentile carry no
# information at all about grain amplitude, grain size, halation radius, or
# grade strength — five of the seven parameters were never derivable from the
# statistics they were attributed to. The two that were have an unexcluded
# alternative explanation in crop bias (photographers crop toward the subject
# and away from blown edges, which moves mean luminance on its own).
#
# exposure_bias and contrast_strength are deliberately 0 in every preset. The
# LUT carries the colour; exposure is per-photo and always was.

BUILTIN_PRESETS = {
    "Ambient Film": {
        "lut":                "kodak_gold_200",
        "grade_strength":     0.85,
        "exposure_bias":      0.0,
        "contrast_strength":  0.0,
        "grain_intensity":    0.022,
        "grain_size":         0.0018,
        "halation_intensity": 0.50,
        "halation_radius":    0.010,
    },
    "Flash Film": {
        "lut":                "kodak_gold_200",
        "grade_strength":     0.50,
        "exposure_bias":      0.0,
        "contrast_strength":  0.0,
        "grain_intensity":    0.014,
        "grain_size":         0.0012,
        "halation_intensity": 0.30,
        "halation_radius":    0.012,
    },
}

# ── Parameter validation ──────────────────────────────────────────────────────
# Params arrive as untrusted JSON from the browser and from the presets file.
# They are coerced and clamped here, at the boundary, before any of them reach
# numpy or the filesystem. grain_size and halation_radius both scale a Gaussian
# kernel, so an unbounded value asks for arbitrary work from an arbitrarily
# small photo; `lut` reaches the filesystem as a path component.

PARAM_SPEC = {
    "grade_strength":     (float, 0.0,   1.0),
    "exposure_bias":      (float, -5.0,  5.0),    # EV stops, not an offset
    "contrast_strength":  (float, -1.0,  1.0),
    "grain_intensity":    (float, 0.0,   0.10),  # 0.02 ~= 5 levels of 8-bit std
    "grain_size":         (float, 0.0,   0.05),   # fraction of the short edge
    "halation_intensity": (float, 0.0,   1.0),
    "halation_radius":    (float, 0.0,   0.10),   # fraction of the long edge
    "seed":               (int,   0,     2 ** 31 - 1),
}

DEFAULT_PARAMS = {
    "lut":                "kodak_gold_200",
    "grade_strength":     0.85,
    "exposure_bias":      0.0,
    "contrast_strength":  0.0,
    "grain_intensity":    0.018,
    "grain_size":         0.0015,
    "halation_intensity": 0.45,
    "halation_radius":    0.010,
    "seed":               0,   # 0 == auto: derive it from the file. See _seed_from_file.
}


def coerce_params(params) -> dict:
    """Validate and clamp untrusted params. Raises ValueError on garbage."""
    if not isinstance(params, dict):
        raise ValueError("params must be an object")

    clean = dict(DEFAULT_PARAMS)
    for key, value in params.items():
        if key not in PARAM_SPEC:
            continue  # ignore unknown keys rather than failing the whole request
        cast, low, high = PARAM_SPEC[key]
        if isinstance(value, bool) or value is None:
            raise ValueError(f"{key}: expected a number")
        try:
            number = cast(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{key}: expected a number") from None
        if not math.isfinite(number):
            raise ValueError(f"{key}: must be finite")
        clean[key] = max(low, min(high, number))

    # `lut` is a string, so it cannot go through the numeric spec above. It is
    # also the one param that becomes a path component, so it is restricted to a
    # simple name — no separators, no dots, nothing that can climb out of luts/.
    lut_name = params.get("lut", DEFAULT_PARAMS["lut"])
    if not isinstance(lut_name, str) or not lut_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("lut: expected a simple name")
    clean["lut"] = lut_name
    return clean


# ── LUTs ──────────────────────────────────────────────────────────────────────

LUT_DIR = Path(__file__).parent / "luts"


def _lut_dirs() -> list[Path]:
    """LUT search roots, highest priority first.

    In a frozen build LUT_DIR points inside the unpacked bundle, which ships the
    open LUTs but is a temp dir the user cannot add files to — so a luts/ folder
    beside the exe is searched first, keeping the documented "drop a file into
    luts/private/ and re-render" workflow alive.
    """
    if getattr(sys, "frozen", False):
        return [Path(sys.executable).parent / "luts", LUT_DIR]
    return [LUT_DIR]

GREY_SCENE = 0.18    # linear scene middle grey
GREY_DISPLAY = 0.18  # where we want it to land before the LUT

# Keyed on the FILE's identity, not the LUT's name: a name is not stable enough
# to cache under. Dropping a freshly extracted LUT into luts/private/ replaces
# the file behind a name that has already been rendered with, and a name-keyed
# entry would go on serving the old table for the life of the process.
_LUT_CACHE: dict[tuple, np.ndarray] = {}


def get_lut(name: str) -> np.ndarray:
    """Load a LUT by name, from luts/private/ first, then luts/open/."""
    candidates = (base / folder / f"{name}.png"
                  for folder in ("private", "open") for base in _lut_dirs())
    for path in candidates:
        if path.exists():
            stat = path.stat()
            # A replaced file gets a new mtime (or a new size), and so a new key.
            key = (str(path), stat.st_mtime_ns, stat.st_size)
            if key not in _LUT_CACHE:
                _LUT_CACHE[key] = load_hald(path)
            return _LUT_CACHE[key]

    # No LUT installed: fall through to a no-op rather than failing the render.
    #
    # NEVER cache this. The documented workflow (docs/extracting-a-lut.md) is
    # "drop a file into luts/private/ and re-render", so the miss is expected to
    # stop being a miss without a restart. A cached identity would silently
    # swallow the new LUT — and a LUT doing nothing is indistinguishable from a
    # LUT that is wrong, which is what sends someone off to re-extract a LUT
    # that was fine. The cube is 2x2x2 and the exists() checks are two stats;
    # both are free next to decoding the photo.
    return identity_cube(2)


# ── Exposure and contrast ─────────────────────────────────────────────────────

def apply_exposure(linear_rgb, ev: float):
    """Exposure is a multiply in linear light.

    Doubling the photons doubles the linear value. That preserves the ratios
    between channels, and so preserves hue and saturation exactly. Adding a
    constant to gamma-encoded values — which is what this used to do — is a
    black-level lift whose effective gain varies per channel, which is why it
    shifted hue.

    Nothing is clipped here: the highlight rolloff downstream needs the
    headroom this creates.
    """
    if ev == 0.0:
        return linear_rgb
    return linear_rgb * np.float32(2.0 ** ev)


def apply_contrast(rgb, strength: float):
    """Post-LUT user finish. Never baked into a film preset — the CLUT is the look."""
    if strength == 0.0:
        return rgb
    return np.clip(0.5 + (rgb - 0.5) * np.float32(1.0 + strength), 0.0, 1.0)


# ── Grain seed ────────────────────────────────────────────────────────────────

def _seed_from_file(input_path: Path) -> int:
    """Derive a grain seed from the input file's bytes.

    seed=0 is the "auto" sentinel, and it used to mean literally zero: nothing in
    DEFAULT_PARAMS, the presets or the UI ever set it, so every photo of a given
    size drew the SAME noise field. That is fixed-pattern noise across a whole
    shoot — invisible in one frame and obvious the moment the set is viewed
    together.

    Hashing the file keeps both properties that matter: a photo renders the same
    way every time (so a re-export matches the preview), and no two photos share
    a field. An explicit non-zero seed still overrides this.
    """
    digest = hashlib.blake2b(input_path.read_bytes(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_photo(input_path: Path, params: dict) -> bytes:
    """load -> exposure -> [if scene] grey-point scale -> halation -> highlights
    -> encode -> LUT -> contrast -> grain -> JPEG.

    The highlight stage is where the state tag earns its keep, and the two states
    need opposite things:

      SCENE (a RAW) is scene-linear, with real headroom above 1.0 that nothing
      has rendered yet. It needs a soft shoulder — that shoulder IS the
      scene-to-display render.

      DISPLAY (a JPEG) has already been rendered by the camera. At EV=0 nothing
      in it exceeds 1.0, so the correct transform is the IDENTITY: the LUT must
      receive exactly the sRGB space it was authored against, and a shoulder here
      is a second tone curve in front of it. Only an exposure push can send
      values over 1.0, and those come back down with a hue-preserving clip.
    """
    params = coerce_params(params)

    linear, state = load_image(input_path)

    linear = apply_exposure(linear, params["exposure_bias"])

    if state == SCENE:
        # Land scene middle grey where the CLUTs expect it. This is the whole of
        # the scene-to-display exposure mapping: no look, no contrast curve. The
        # CLUTs were authored against a neutral render, and anything opinionated
        # here would be counted twice.
        #
        # BEFORE halation, not after: HALATION_THRESHOLD (0.70 linear) and the
        # rolloff knee (0.8 linear) are both ABSOLUTE values, so they have to see
        # the same exposure. With the scale between them, one RAW highlight would
        # be tested for halation at one exposure and rolled off at another.
        linear = linear * np.float32(GREY_DISPLAY / GREY_SCENE)

    linear = add_halation(
        linear,
        intensity=params["halation_intensity"],
        radius=params["halation_radius"],
    )

    if state == SCENE:
        linear = highlight_rolloff(linear)
    else:
        linear = hue_preserving_clip(linear)

    # Both branches already land in [0,1], so this clip is not shaping anything —
    # it is float32 hygiene. srgb_encode's power function can return 1.0 plus an
    # ULP at the top of the range, and uint8 conversion downstream is unforgiving
    # about it. The tone decision was made above, deliberately, per state.
    rgb = np.clip(srgb_encode(linear), 0.0, 1.0)

    rgb = apply_lut(rgb, get_lut(params["lut"]), strength=params["grade_strength"])
    # Contrast is AFTER the LUT: the CLUTs were authored against a neutral,
    # standard-contrast render, so look-contrast in front of them is counted
    # twice. Grain is after contrast: it is the texture the emulsion leaves on a
    # finished frame, not a signal that gets graded.
    rgb = apply_contrast(rgb, params["contrast_strength"])
    seed = params["seed"] or _seed_from_file(input_path)
    rgb = add_grain(
        rgb,
        intensity=params["grain_intensity"],
        size=params["grain_size"],
        seed=seed,
    )

    # ROUND, don't truncate. astype("uint8") floors, which biases every pixel
    # down by half a level and — the reason it matters here — throws away the top
    # code value outright: a float32 sRGB decode/encode round trip of pure white
    # lands a hair under 1.0, and 254.99998 floors to 254. White has to stay 255.
    pil_out = Image.fromarray(np.rint(rgb * 255.0).clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    pil_out.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf.read()


# ── Preset persistence ────────────────────────────────────────────────────────

_PRESETS_LOCK = threading.Lock()

# grain_size and halation_radius changed units. They used to be PIXEL counts
# (grain_size: 3, halation_radius: 38); they are now FRACTIONS of an edge, with
# maxima of 0.05 and 0.10. A preset written before that change and fed straight
# through coerce_params clamps BOTH to their maximum, which on a 4000x3000 photo
# is a 150px grain sigma and a 400px halation sigma — the render is destroyed.
#
# The migration cannot live in coerce_params: that is a boundary clamp on
# untrusted input and has no way to tell a v1 file from a v2 request. It lives
# here, where "this value came out of a stored file" is known.
#
# Detection is by range, which is unambiguous: the new maxima are 0.05 and 0.10,
# and the old pixel values were >= 1 and >= 5. Anything above the new maximum in
# a stored preset is therefore a v1 pixel count.
#
# There is no conversion. A pixel count only becomes a fraction if you know the
# dimensions of the image it was tuned on, and those are not in the file. The
# default is the honest answer; an invented conversion is not.
_LEGACY_PIXEL_PARAMS = ("grain_size", "halation_radius")


def _migrate_legacy_units(data: dict) -> dict:
    for preset_name, params in data.items():
        if not isinstance(params, dict):
            continue
        for key in _LEGACY_PIXEL_PARAMS:
            value = params.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            _cast, _low, high = PARAM_SPEC[key]
            if value > high:
                logger.warning(
                    "Preset %r: %s=%s is a pixel count from an older version "
                    "(%s is now a fraction of an edge, max %s). Falling back to "
                    "the default of %s — the original value cannot be converted "
                    "without the dimensions of the photo it was tuned on.",
                    preset_name, key, value, key, high, DEFAULT_PARAMS[key],
                )
                params[key] = DEFAULT_PARAMS[key]
    return data


def _load_user_presets(presets_file: Path, tolerate_corrupt: bool = True) -> dict:
    """Read the presets file.

    tolerate_corrupt is the difference between a read and a read-modify-write.
    On a read path an unparseable file may degrade to "no user presets". On a
    write path it must NOT: returning {} there would write the empty dict back
    over the file and destroy every preset the user had, while reporting 200.
    """
    if not presets_file.exists():
        return {}
    try:
        data = json.loads(presets_file.read_text(encoding="utf-8"))
    except Exception:
        if tolerate_corrupt:
            return {}
        raise
    if not isinstance(data, dict):
        if tolerate_corrupt:
            return {}
        raise ValueError("presets file is not a JSON object")
    return _migrate_legacy_units(data)


def _save_user_presets(presets_file: Path, user_presets: dict):
    """Write atomically — a torn write leaves invalid JSON, which the reader
    used to swallow, turning a crash mid-save into silent total data loss."""
    tmp = presets_file.with_suffix(presets_file.suffix + ".tmp")
    tmp.write_text(
        json.dumps(user_presets, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp, presets_file)


# ── Flask routes ──────────────────────────────────────────────────────────────

MAX_PRESET_NAME = 80


def register_film_routes(app, presets_file: Path):
    """Register all Film Lab routes on the Flask app."""

    @app.route("/api/film/presets")
    def film_get_presets():
        user_presets = _load_user_presets(presets_file)
        result = []
        for name, params in BUILTIN_PRESETS.items():
            result.append({"name": name, "params": params, "builtin": True})
        for name, params in user_presets.items():
            result.append({"name": name, "params": params, "builtin": False})
        return jsonify(result)

    @app.route("/api/film/presets", methods=["POST"])
    def film_save_preset():
        # Not force=True: that parses any Content-Type, which makes a
        # text/plain cross-origin POST a CORS simple request with no preflight —
        # letting any site the user visits write presets into a Film Lab running
        # on localhost. Requiring application/json forces a preflight.
        if not request.is_json:
            return jsonify({"error": "Expected application/json."}), 415

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request body."}), 400

        name = str(data.get("name", "")).strip()
        if not name:
            return jsonify({"error": "Preset name is required."}), 400
        if len(name) > MAX_PRESET_NAME:
            return jsonify({"error": "Preset name is too long."}), 400
        if "/" in name or "\\" in name:
            # The default <name> URL converter does not match slashes, so such a
            # preset could be created but never deleted.
            return jsonify({"error": "Preset name cannot contain slashes."}), 400
        if name in BUILTIN_PRESETS:
            return jsonify({"error": "Cannot overwrite a built-in preset."}), 400

        try:
            params = coerce_params(data.get("params"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        with _PRESETS_LOCK:
            try:
                user_presets = _load_user_presets(presets_file, tolerate_corrupt=False)
            except Exception:
                return jsonify({
                    "error": "Presets file is unreadable; refusing to overwrite it."
                }), 500
            user_presets[name] = params
            _save_user_presets(presets_file, user_presets)
        return jsonify({"ok": True})

    @app.route("/api/film/presets/<name>", methods=["DELETE"])
    def film_delete_preset(name):
        if name in BUILTIN_PRESETS:
            return jsonify({"error": "Cannot delete a built-in preset."}), 400

        with _PRESETS_LOCK:
            try:
                user_presets = _load_user_presets(presets_file, tolerate_corrupt=False)
            except Exception:
                return jsonify({
                    "error": "Presets file is unreadable; refusing to overwrite it."
                }), 500
            if name not in user_presets:
                return jsonify({"error": "Preset not found."}), 404
            del user_presets[name]
            _save_user_presets(presets_file, user_presets)
        return jsonify({"ok": True})

    @app.route("/api/film/process", methods=["POST"])
    def film_process():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        f = request.files["file"]
        suffix = Path(f.filename).suffix.lower() if f.filename else ""
        if suffix not in RAW_EXTENSIONS | IMAGE_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type: {suffix}"}), 400

        try:
            params = coerce_params(json.loads(request.form.get("params", "{}")))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            return jsonify({"error": "Invalid params."}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            f.save(str(tmp_path))
            result_bytes = process_photo(tmp_path, params)
        except Exception:
            # Never echo the exception: it carries the temp path, which leaks
            # the OS username and filesystem layout to the caller.
            app.logger.exception("Processing failed for %s", f.filename)
            return jsonify({"error": "Could not process this photo."}), 500
        finally:
            tmp_path.unlink(missing_ok=True)

        buf = io.BytesIO(result_bytes)
        buf.seek(0)
        return send_file(
            buf,
            mimetype="image/jpeg",
            as_attachment=True,
            download_name="film_processed.jpg",
        )

    # ── Batch ─────────────────────────────────────────────────────────────────

    def _process_to_disk(src: Path, out: Path, params: dict):
        out.write_bytes(process_photo(src, params))

    batch = BatchManager(RAW_EXTENSIONS | IMAGE_EXTENSIONS, _process_to_disk)

    @app.route("/api/film/batch", methods=["POST"])
    def film_batch_start():
        if not request.is_json:
            return jsonify({"error": "Expected application/json."}), 415

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request body."}), 400

        source = str(data.get("source", "")).strip()
        dest = str(data.get("dest", "")).strip()
        if not source or not dest:
            return jsonify({"error": "Source and output folders are required."}), 400

        try:
            params = coerce_params(data.get("params") or {})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            job = batch.start(Path(source), Path(dest), params)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

        return jsonify(_dataclass_dict(job))

    @app.route("/api/film/batch/<job_id>")
    def film_batch_status(job_id):
        state = batch.get(job_id)
        if state is None:
            return jsonify({"error": "No such job."}), 404
        return jsonify(state)

    @app.route("/api/film/batch/<job_id>/cancel", methods=["POST"])
    def film_batch_cancel(job_id):
        return jsonify({"cancelled": batch.cancel(job_id)})
