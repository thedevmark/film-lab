"""Film Lab pipeline — photo style processing with a measured film LUT, grain, and halation."""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
from PIL import Image
from flask import request, jsonify, send_file

from filmlab.effects import add_grain, add_halation
from filmlab.loader import IMAGE_EXTENSIONS, RAW_EXTENSIONS, SCENE, load_image
from filmlab.lut import apply_lut, identity_cube, load_hald
from filmlab.tone import highlight_rolloff, srgb_encode

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
        "grain_intensity":    0.065,
        "grain_size":         0.0018,
        "halation_intensity": 0.50,
        "halation_radius":    0.010,
    },
    "Flash Film": {
        "lut":                "kodak_gold_200",
        "grade_strength":     0.50,
        "exposure_bias":      0.0,
        "contrast_strength":  0.0,
        "grain_intensity":    0.040,
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
    "grain_intensity":    (float, 0.0,   1.0),
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
    "grain_intensity":    0.055,
    "grain_size":         0.0015,
    "halation_intensity": 0.45,
    "halation_radius":    0.010,
    "seed":               0,
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

GREY_SCENE = 0.18    # linear scene middle grey
GREY_DISPLAY = 0.18  # where we want it to land before the LUT

_LUT_CACHE: dict[str, np.ndarray] = {}


def get_lut(name: str) -> np.ndarray:
    """Load a LUT by name, from luts/private/ first, then luts/open/."""
    if name in _LUT_CACHE:
        return _LUT_CACHE[name]

    for folder in ("private", "open"):
        path = LUT_DIR / folder / f"{name}.png"
        if path.exists():
            cube = load_hald(path)
            _LUT_CACHE[name] = cube
            return cube

    # No LUT installed: fall through to a no-op rather than failing the render.
    cube = identity_cube(2)
    _LUT_CACHE[name] = cube
    return cube


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


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_photo(input_path: Path, params: dict) -> bytes:
    """load -> exposure -> halation -> render -> LUT -> contrast -> grain -> JPEG."""
    params = coerce_params(params)

    linear, state = load_image(input_path)

    linear = apply_exposure(linear, params["exposure_bias"])
    linear = add_halation(
        linear,
        intensity=params["halation_intensity"],
        radius=params["halation_radius"],
    )

    if state == SCENE:
        # Land scene middle grey where the CLUTs expect it. This is the whole of
        # the scene-to-display render: no look, no contrast curve. The CLUTs were
        # authored against a neutral render, and anything opinionated here would
        # be counted twice.
        linear = linear * np.float32(GREY_DISPLAY / GREY_SCENE)

    # Both branches: catch whatever the exposure push sent past 1.0, without
    # rotating hue the way a per-channel clip would.
    linear = highlight_rolloff(linear)

    rgb = np.clip(srgb_encode(linear), 0.0, 1.0)

    rgb = apply_lut(rgb, get_lut(params["lut"]), strength=params["grade_strength"])
    # Contrast is AFTER the LUT: the CLUTs were authored against a neutral,
    # standard-contrast render, so look-contrast in front of them is counted
    # twice. Grain is after contrast: it is the texture the emulsion leaves on a
    # finished frame, not a signal that gets graded.
    rgb = apply_contrast(rgb, params["contrast_strength"])
    rgb = add_grain(
        rgb,
        intensity=params["grain_intensity"],
        size=params["grain_size"],
        seed=params["seed"],
    )

    pil_out = Image.fromarray((rgb * 255).clip(0, 255).astype("uint8"))
    buf = io.BytesIO()
    pil_out.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf.read()


# ── Preset persistence ────────────────────────────────────────────────────────

_PRESETS_LOCK = threading.Lock()


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
    return data


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
