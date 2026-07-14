"""Film Lab pipeline — photo style processing with mathematical film rendering, grain, and halation."""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
import threading
from pathlib import Path

from PIL import Image, ImageOps
from flask import request, jsonify, send_file

# ── Built-in presets ─────────────────────────────────────────────────────────
# Built-ins live in code only — cannot be overwritten by the user.
#
# Derived from a scan of 45 shoot folders / 2,602 raw-to-edited pairs from a
# personal library (2023-2026). Two measured scene families:
#
#   Ambient Film — daylight/ambient sets:
#     avg luminance −6.9 units, blue reduced most (B > G > R), sat +4.1pp,
#     shadow crush +3.1pp, contrast essentially flat.
#     → darker, denser, warm-biased, moodier blacks.
#
#   Flash Film — event/flash sets:
#     avg luminance +9.9 units, all channels lift equally (no forced warmth),
#     contrast +8.9, sat −1.3pp, shadow barely touched.
#     → brighter, punchier, colour-neutral, highlights preserved.

BUILTIN_PRESETS = {
    # ── Your signature looks ──────────────────────────────────────────────────
    "Ambient Film": {
        "grade_strength":     0.85,
        "exposure_bias":     -0.025,
        "contrast_strength":  0.00,
        "grain_intensity":    0.065,
        "grain_size":         3,
        "halation_intensity": 0.50,
        "halation_radius":    38,
    },
    "Flash Film": {
        "grade_strength":     0.50,
        "exposure_bias":      0.040,
        "contrast_strength":  0.20,
        "grain_intensity":    0.040,
        "grain_size":         2,
        "halation_intensity": 0.30,
        "halation_radius":    45,
    },
    # ── Fuji Gold 400 reference ───────────────────────────────────────────────
    "Fuji Gold 400 — Standard": {
        "grade_strength":     0.85,
        "exposure_bias":      0.00,
        "contrast_strength":  0.00,
        "grain_intensity":    0.055,
        "grain_size":         3,
        "halation_intensity": 0.45,
        "halation_radius":    38,
    },
    "Fuji Gold 400 — Subtle": {
        "grade_strength":     0.50,
        "exposure_bias":      0.00,
        "contrast_strength":  0.00,
        "grain_intensity":    0.030,
        "grain_size":         2,
        "halation_intensity": 0.20,
        "halation_radius":    25,
    },
}

RAW_EXTENSIONS   = {".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".dng", ".rw2", ".pef"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
MAX_LONG_EDGE    = 6000

# ── Parameter validation ──────────────────────────────────────────────────────
# Params arrive as untrusted JSON from the browser and from the presets file.
# They are coerced and clamped here, at the boundary, before any of them reach
# numpy. grain_size in particular is load-bearing: add_grain() allocates a
# size x size array independent of the image, so an unbounded value asks for
# arbitrary memory from an arbitrarily small photo.

PARAM_SPEC = {
    "grade_strength":     (float, 0.0,   1.0),
    "exposure_bias":      (float, -1.0,  1.0),
    "contrast_strength":  (float, -1.0,  1.0),
    "grain_intensity":    (float, 0.0,   1.0),
    "grain_size":         (int,   1,     64),
    "halation_intensity": (float, 0.0,   1.0),
    "halation_radius":    (float, 0.0,   200.0),
}

DEFAULT_PARAMS = {
    "grade_strength":     0.85,
    "exposure_bias":      0.0,
    "contrast_strength":  0.0,
    "grain_intensity":    0.055,
    "grain_size":         3,
    "halation_intensity": 0.45,
    "halation_radius":    38.0,
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
    return clean


_NUMPY = None


def _require_numpy():
    global _NUMPY
    if _NUMPY is None:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is not installed — Film Lab is unavailable until its dependencies are installed.") from exc
        _NUMPY = np
    return _NUMPY


# ── Exposure and contrast ─────────────────────────────────────────────────────

def apply_exposure(img, bias):
    np = _require_numpy()
    if bias == 0.0:
        return img
    return np.clip(img + bias, 0.0, 1.0)


def apply_contrast(img, strength):
    np = _require_numpy()
    if strength == 0.0:
        return img
    return np.clip(0.5 + (img - 0.5) * (1.0 + strength), 0.0, 1.0)


# ── Fuji Gold 400 colour rendering ────────────────────────────────────────────

def apply_film_color(img, strength):
    np = _require_numpy()
    if strength <= 0:
        return img

    luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    result = img.copy()

    # Lift blacks — film base fog raises the shadow floor
    shadow_lift = 0.022
    result = result * (1.0 - shadow_lift * 2.0) + shadow_lift

    # Per-channel tone response
    result[:, :, 0] = np.power(result[:, :, 0].clip(1e-6, 1.0), 0.88)
    result[:, :, 1] = np.power(result[:, :, 1].clip(1e-6, 1.0), 0.96)
    result[:, :, 2] = np.power(result[:, :, 2].clip(1e-6, 1.0), 1.18)

    # Shadow crossover — green-cyan bleed in deepest shadows
    shadow_mask = np.clip(1.0 - luma / 0.25, 0.0, 1.0) ** 2
    result[:, :, 0] -= shadow_mask * 0.012
    result[:, :, 1] += shadow_mask * 0.048
    result[:, :, 2] += shadow_mask * 0.022

    # Midtone warmth
    mid_mask = np.exp(-((luma - 0.40) ** 2) / (2 * 0.18 ** 2))
    result[:, :, 0] += mid_mask * 0.032
    result[:, :, 1] += mid_mask * 0.018
    result[:, :, 2] -= mid_mask * 0.022

    # Orange-red saturation boost
    orange = np.clip(result[:, :, 0] - np.maximum(result[:, :, 1], result[:, :, 2]), 0.0, 1.0)
    result[:, :, 0] += orange * 0.055
    result[:, :, 2] -= orange * 0.030

    # Highlight warmth
    hi_mask = np.clip((luma - 0.70) * 3.5, 0.0, 1.0)
    result[:, :, 0] += hi_mask * 0.014
    result[:, :, 1] += hi_mask * 0.006
    result[:, :, 2] -= hi_mask * 0.010

    return np.clip(strength * result + (1.0 - strength) * img, 0.0, 1.0)


# ── Grain ─────────────────────────────────────────────────────────────────────

def add_grain(img, intensity, size):
    np = _require_numpy()
    if intensity <= 0:
        return img

    h, w = img.shape[:2]
    # Clamp against the image: floor division below pins nh/nw at 1 once size
    # exceeds the image, but np.repeat then re-expands by size regardless, so
    # the intermediate would be size x size x 3 no matter how small the photo.
    size = max(1, min(int(size), h, w))
    nh, nw = max(1, h // size), max(1, w // size)

    small = np.random.normal(0.0, intensity, (nh, nw, 3)).astype(np.float32)
    coarse = np.repeat(np.repeat(small, size, axis=0), size, axis=1)
    ph = max(0, h - coarse.shape[0])
    pw = max(0, w - coarse.shape[1])
    if ph or pw:
        coarse = np.pad(coarse, ((0, ph), (0, pw), (0, 0)), mode="wrap")
    coarse = coarse[:h, :w]

    luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    weight = (1.0 - luma * 0.85)[..., np.newaxis]
    coarse *= weight

    return np.clip(img + coarse, 0.0, 1.0)


# ── Halation ──────────────────────────────────────────────────────────────────

def add_halation(img, intensity, radius):
    """Highlights scatter through the emulsion, reflect off the base, re-expose the film.

    Additive, and red-dominant: the red-sensitive layer sits deepest, so it
    catches most of the light that bounced. Blue is left alone.
    """
    np = _require_numpy()
    from filmlab.blur import gaussian_blur
    if intensity <= 0:
        return img

    highlights = np.clip(img - 0.75, 0.0, 1.0) * 4.0
    red = (highlights[:, :, 0] * 0.78
           + highlights[:, :, 1] * 0.17
           + highlights[:, :, 2] * 0.05)

    # Blur is linear, so the green bloom is just a scaled copy of the red one —
    # blurring it separately would double the work and buy nothing.
    blurred_red = gaussian_blur(red, float(radius))

    bloom = np.zeros_like(img)
    bloom[:, :, 0] = blurred_red
    bloom[:, :, 1] = blurred_red * 0.20

    return np.clip(img + bloom * intensity, 0.0, 1.0)


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_image(path: Path):
    np = _require_numpy()
    suffix = path.suffix.lower()

    if suffix in RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError:
            raise RuntimeError("rawpy is not installed — RAW file support unavailable.")
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        pil = Image.fromarray(rgb, mode="RGB")  # rawpy already applies orientation
    else:
        # A vertically-held shot is stored in landscape layout plus an
        # Orientation tag. Without this the photo comes back rotated 90°, since
        # the output JPEG carries no EXIF to compensate.
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")

    # Downscale on the PIL object: converting to float32 first would cost 4x the
    # uint8 size at full resolution, before the cap ever applied.
    w, h = pil.size
    long_edge = max(h, w)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    return np.asarray(pil, dtype=np.float32) / 255.0


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_photo(input_path: Path, params: dict) -> bytes:
    """Order: load → exposure → colour grade → contrast → halation → grain → export."""
    params = coerce_params(params)

    img = _load_image(input_path)
    img = apply_exposure(img, params["exposure_bias"])
    img = apply_film_color(img, params["grade_strength"])
    img = apply_contrast(img, params["contrast_strength"])
    img = add_halation(
        img,
        intensity=params["halation_intensity"],
        radius=params["halation_radius"],
    )
    img = add_grain(
        img,
        intensity=params["grain_intensity"],
        size=params["grain_size"],
    )

    pil_out = Image.fromarray((img * 255).clip(0, 255).astype("uint8"))
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
