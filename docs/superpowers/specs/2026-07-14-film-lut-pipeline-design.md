# Film Lab — LUT-based colour pipeline and batch processing

**Date:** 2026-07-14
**Status:** Design, awaiting approval

---

## Problem

`film.py` applies a "film look" by way of `apply_film_color()`: per-channel gamma
powers, a Gaussian midtone-warmth mask, a shadow crossover term, an orange
saturation hack. It is hand-tuned and it is a guess. It produces a pleasant warm
rendering; it does not reproduce any particular film stock.

Authentic film emulation — what DxO FilmPack sells and what darktable's `lut3d`
module implements — uses a **3D LUT**: a measured mapping from every input colour
to an output colour. That is the gap this design closes.

A second gap: photos are processed one at a time through an HTTP round-trip. A
shoot is 200 files.

## What the design reviews changed

Three independent adversarial reviews were run against an earlier draft. They
overturned four decisions, and the corrections are load-bearing enough to state
before the design rather than inside it.

**The tone curves stack at the decoder, not at the LUT.** The earlier draft
worried that applying a tone map (filmic/sigmoid) before a film LUT would stack
two tone curves. It does not: the RawTherapee/G'MIC HaldCLUTs were authored by
hand in GIMP against *already-rendered* images, contain no scene-to-display
transform, and both darktable and RawTherapee apply them after tone mapping.
RawPedia is explicit that "you cannot make a HaldCLUT image give a tone-mapped
look."

The real stacking is in `_load_image`. `rawpy.postprocess(use_camera_wb=True,
output_bps=8)` applies BT.709 gamma, **auto-brightness**, highlight clipping, and
returns 8 bits — a complete camera rendering. Linearising that recovers a
linearised camera render, not scene light, and a tone map on top of it is a
*second* rendering. **The tone map must therefore branch on input state**, and
there is no single ordering correct for both a RAW and a JPEG.

**Halation's formula was veiling glare.** A normalised mix,
`out = (1-s)·I + s·blur(I)`, conserves the local mean, so it cannot add density
around a highlight — it just softens the frame and lays colour fringes on every
edge, including dark ones. Halation is light that punched through the emulsion,
reflected off the base, and *re-exposed* it. It is strictly additive. The
per-channel asymmetry is in **amount**, not radius: the red-sensitive layer sits
deepest, so `s_R ≫ s_G > s_B ≈ 0`.

**Trilinear interpolation tints neutrals.** In the standard six-tetrahedra
decomposition the P000–P111 diagonal is a shared edge of every tetrahedron, so an
input with `R=G=B` interpolates only from lattice nodes that are themselves
neutral — greys stay grey *by construction*. Trilinear draws from all eight
corners, most off-axis, and introduces a faint hue shift into what should be a
pure neutral. It is also only C⁰ across cell boundaries, which reads as stepping
in skies and skin falloff. Tetrahedral is darktable's default, is ~25 lines of
vectorised numpy, and is *cheaper* (4 lattice fetches, not 8).

**The plan to re-derive presets as post-LUT residuals is a fixed point.** Let the
DxO edit be `A_i ∘ F ∘ R` — per-photo adjustments, film preset, raw development.
The residual `DxO_final − LUT(raw)` either reduces to exactly `A_i` (per-photo by
construction, so not generalisable) or is dominated by the mismatch between our
raw converter and DxO's. **The photos where the residual is clean are precisely
the photos where it is zero.** There is no third kind of photo.

Worse, the existing provenance claim does not survive inspection. Mean luminance,
per-channel means, saturation, and shadow percentile carry **zero information**
about grain amplitude, grain size, halation radius, or grade strength — five of
the seven preset parameters were never derivable from the stated statistics. And
the two that were have a mechanical alternative explanation nobody ruled out:
photographers crop toward the subject and away from blown edges, so flash work
(tight crop onto a lit subject) raises mean luminance and daylight work (crop the
sky out) lowers it *and* drops blue most — reproducing both headline numbers with
zero editing.

---

## Architecture

Two entry states, converging on one back half.

```
                    ┌─ RAW ──────────────────────────────────┐
                    │  postprocess(gamma=(1,1),              │
                    │              no_auto_bright=True,      │
                    │              output_bps=16)            │
                    │  → SCENE-LINEAR, real highlight headroom│
                    └────────────────────────────────────────┘
                    ┌─ JPEG / PNG / TIFF ────────────────────┐
                    │  exif_transpose, sRGB EOTF⁻¹           │
                    │  → LINEARISED DISPLAY RENDER           │
                    │    (camera already applied its S-curve)│
                    └────────────────────────────────────────┘
                                     │
                                     ▼
                        exposure   × 2^EV          (linear: a multiply
                                                    preserves channel ratios,
                                                    hence hue and saturation)
                                     │
                                     ▼
                        halation   E + s_c·blur(E, σ)        (linear, ADDITIVE,
                                                    s_R ≫ s_G > s_B ≈ 0)
                                     │
                       ┌─────────────┴─────────────┐
              scene-linear                   display-rendered
                    │                               │
              filmic / sigmoid            hue-preserving highlight
              (scene → display)           rolloff on the RGB norm
                    │                     (catch what the EV push
                    │                      pushed past 1.0)
                    └─────────────┬─────────────┘
                                  ▼
                        encode sRGB gamma, clip [0,1]
                                  ▼
                        LUT        tetrahedral, strength blend
                                  ▼
                        contrast   (optional user finish, default 0)
                                  ▼
                        grain      monochrome field, midtone-peaked
                                  ▼
                                 JPEG
```

Grain is last. Applying contrast after it would stretch the grain along with the
image, which is backwards: grain is the texture the emulsion leaves on a finished
frame, not a signal that gets graded.

Two constraints are doing real work here and must not be quietly violated later:

The pre-LUT image must be a **neutral, standard-contrast sRGB render**. That is
the state the CLUTs were authored against — darktable's manual says the module
"should be applied to a neutral image (without first applying a specific look)."
So **no look-contrast before the LUT**. `contrast_strength` moves *after* the LUT
and becomes a user finishing control, defaulting to zero, never baked into a film
preset. The CLUT is the look.

And the LUT is display-referred. Letting it *be* the tone map — the tempting
simplification — is not viable: a 64³ table sampled uniformly in display code
values has roughly four lattice nodes of headroom above diffuse white in which to
perform a ten-stop compression. It would posterise catastrophically.

---

## Components

### `filmlab/io.py` — loading

Returns `(linear_rgb: float32, state: Literal["scene", "display"])`. The state
tag is the whole point: it is what the tone-mapping branch dispatches on, and it
is the thing the current code gets wrong by pretending everything is the same.

RAW: `postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16,
output_color=sRGB, use_camera_wb=True)`. Rendered: `exif_transpose`, then the
sRGB EOTF inverse.

Downscale happens on the PIL object, before the float32 conversion. (Already
fixed on `main`.)

### `filmlab/lut.py` — the film profile

A HaldCLUT of level *N* encodes a cube of `S = N²` per axis as an `N³ × N³`
image; level 8 → 512×512 → 64³. Read in raster order into a flat `S³ × 3` table
indexed `r + S·g + S²·b`. Read 16-bit PNGs at 16 bits — Pat David's originals
were authored at 16-bit, and reading them as 8 quantises the table before we
interpolate it.

Tetrahedral interpolation: integer index plus fractional offsets, six orderings
selected by comparing `(dr, dg, db)`, four-node weighted sum. Vectorised over the
whole image, no Python loop.

`grade_strength` is a linear blend between the input and the LUT output.

### `filmlab/effects.py` — halation and grain

**Halation.** One blur of the highlight signal, three per-channel scales — since
blur is linear, a separate blur per channel is redundant work. `σ` is a fraction
of the long edge, not a pixel count, so the look does not change between preview
and export. Additive.

**Grain.** One **monochrome** noise field, not `np.random.normal(size=(h,w,3))` —
independent per-channel noise is chroma speckle; it reads as a noisy sensor, not
film. Generated fine and Gaussian-filtered to size, not `np.repeat`-ed into
square axis-aligned blocks. Size normalised by `min(h, w)`. Amplitude weighted by
a midtone-peaked response (darktable's `paper_resp`: the noise perturbs
*exposure*, and the perturbation is pushed through a paper S-curve, so the same
noise produces large ΔL in the steep midtones and little in the shoulders). The
current `weight = 1.0 - luma*0.85` puts maximum grain in the blacks, which is
backwards — real RMS granularity peaks in the midtones.

Seeded from a hash of the input file and the params, so one photo through one
preset is reproducible, and a batch does not boil.

### `filmlab/batch.py` — folder in, folder out

`POST /api/film/batch {src, dst, preset}` → job id. A single worker thread walks
the source folder and processes sequentially. `GET /api/film/batch/<id>` reports
progress. Outputs that already exist are skipped, which is what makes it
resumable. Cancellable.

Sequential, not parallel: each photo already holds several full-resolution
float32 buffers, and a thread pool would multiply peak memory by the pool size
for a wall-clock win that does not matter on an overnight run.

### Presets

```python
{"lut": "kodak_gold_200", "grade_strength": 0.85,
 "grain_intensity": 0.055, "grain_size": 3,
 "halation_intensity": 0.45, "halation_radius": 0.006}   # fraction of long edge
```

**`exposure_bias` and `contrast_strength` are gone from every preset.** Once a
LUT carries the colour there is no defensible reason a preset also carries a
global exposure offset — exposure is per-photo, and averaging a library's
per-photo exposure decisions estimates the *camera's meter bias*, not the
photographer's taste. Both remain as user sliders, defaulting to zero.

The provenance comment at `film.py:16-27` is deleted. `Ambient` and `Flash` stay
as named starting points; they are honest aesthetic choices and dishonest
measurements.

---

## Where the LUT comes from

The repo is public and MIT. It ships the **loader**, an **openly-licensed
default**, and **instructions**. It does not redistribute profiles extracted from
commercial software.

- `luts/open/` — shipped. Baked from **spektrafilm** (GPLv3 code, CC BY-SA 4.0
  profiles; spectral simulation from measured Kodak/Fuji datasheet density
  curves). Its *output* is data, so a baked LUT is CC BY-SA 4.0 and attributed —
  the MIT code licence is unaffected.
- `luts/private/` — **gitignored**. Where a user's own extracted LUT lives.
- `docs/extracting-a-lut.md` — how to render an identity Hald through an editor
  you have licensed, for your own use.

**Extraction is booby-trapped and the doc must say so.** If the identity Hald is
rendered with grain enabled, every lattice node receives an independent noise
sample — you get a LUT that is noise-corrupted at every node, and it will look
"filmic" enough that you will not notice. Grain, halation, ClearView, Smart
Lighting, lens softness, and every local tool must be off. A 3D LUT is a
*pointwise* map: it structurally cannot represent grain or halation anyway, which
is exactly why those stay as our own modules.

No free HaldCLUT collection contains Kodak Gold. Nearest open analogues are Agfa
Vista 200, Kodak Elite Color 200, and Fuji Superia 200.

---

## Error handling

Params are coerced and clamped at the boundary before reaching numpy; malformed
input is a 400, not a 500. Internal exceptions are logged, never echoed. Preset
writes are atomic and locked. (All landed on `main` already.)

New surface: batch takes **filesystem paths from the browser**. A job that writes
to an arbitrary directory is a real capability. Destination must be created if
absent, must not be inside the source, and the job refuses to overwrite an
existing file unless explicitly told to. Recorded as a risk below.

---

## Testing

Unit: tetrahedral interpolation against an identity LUT (must be a no-op to
float tolerance) and against a known-answer cube; the neutral-axis property
(`R=G=B` in ⇒ `R=G=B` out) which is the entire reason for choosing it; halation
energy against a float reference blur, in absolute terms, not the relative
channel ordering the current test asserts (that passes even when 100% of the
bloom has been quantised away); grain determinism under a fixed seed.

Fixtures: at least one **real camera JPEG with `Orientation=6`**. Every existing
image test feeds synthetic numpy arrays, which is why the rotation bug shipped.

Integration: a RAW and a JPEG through the full pipeline; a batch run over a
temp folder, killed halfway and resumed, asserting it skips what it finished.

Visual: A/B the output against 8–10 DxO finals you already like. This is the
real acceptance test. A 2,602-sample mean of a confounded quantity is not more
scientific than ten careful visual comparisons — it is merely harder to falsify.

---

## Phases

**1 — LUT core.** `lut.py`: Hald loader, tetrahedral interpolation, strength
blend. No pipeline changes yet.
*Done when:* identity LUT is a no-op to float tolerance; neutral axis holds; a
real film HaldCLUT visibly changes a photo.

**2 — Loader and linear pipeline.** `io.py` with the state tag; multiplicative
exposure; additive halation with per-channel strength; the tone-map branch.
*Done when:* a RAW and a JPEG both render sanely; a +1 EV push is a true stop and
does not shift hue; halation adds density around highlights rather than softening
the frame.

**3 — Swap the colour.** Delete `apply_film_color`. Move contrast after the LUT.
Rewrite grain (monochrome, midtone-peaked, resolution-independent, seeded).
Re-point presets at LUTs; strip `exposure_bias`/`contrast_strength`; delete the
provenance comment.
*Done when:* the A/B against your DxO finals is closer than the current pipeline.
**This is the phase that can fail** — if it is not closer, the LUT is wrong and
we stop and fix the LUT rather than proceeding.

**4 — Batch.** Worker, progress, skip-existing, cancel.
*Done when:* a 200-file folder completes, and killing it halfway and restarting
resumes rather than redoing.

**5 — Ship.** `luts/open/` baked from spektrafilm, extraction doc, README.

---

## Open risks

**~~The default LUT may not exist.~~ RESOLVED — it does.** spektrafilm ships
`kodak_gold_200.json`: a measured profile with 81 wavelengths of spectral
sensitivity, per-layer base density, 256-point density curves, Status M
densitometry, D55 reference. It also has a first-class HaldCLUT exporter whose
**supported path is sRGB in, sRGB out** — its scene-linear input spaces are
deliberately disabled, for exactly the reason given above (a uniform [0,1] cube
cannot represent scene-linear highlights past diffuse white). The earlier worry
that spektrafilm's linear-in/linear-out core made it unsuitable for a
display-referred HaldCLUT was backwards. Kodak Gold *100* does not exist in the
set — long discontinued, no datasheet — but 200 is the one that matters.

**The baked LUT is CC BY-SA 4.0, not MIT.** spektrafilm's licence states that
LUTs "are interpreted as direct encodings of the information in the original
profiles" and carry the profile licence. So the shipped LUT is a per-directory
licence carve-out with attribution, and **the bake script cannot live in this
repo** — it imports GPL code. The command is documented instead. Photographs made
with the LUT carry no copyleft.

**The LUT will render brighter than the source.** spektrafilm's
`stops_above_midgray` defaults to 4.0 against a film whose native headroom is
~2.47 stops, chosen so that encoded 1.0 lands on the shoulder and the rolloff
engages on already-rendered SDR input. Its own docs call this "an aesthetic
interpretation, not a measurement." If the phase 3 A/B shows a systematic
brightness offset, this is the knob — not the grey-point scale, and not the
sliders.

**8-bit rendered input, pushed.** A JPEG decoded to linear, EV-pushed, halated,
rolled off, re-encoded, and LUT'd is a lot of transform on 256 levels per
channel. Posterisation in shadows is plausible. Mitigation is to carry float
throughout (we do) and accept that a JPEG is a JPEG. Watch for it in phase 2.

**Batch writes to a browser-supplied path.** A local tool, but the capability is
real. Phase 4 must decide: confine to a configured root, or accept arbitrary
paths with an explicit confirmation step.

**The presets will look different.** Every parameter was fitted against the old
display-space pipeline where `exposure_bias` was an additive offset in gamma
space, not an EV. They will need re-tuning by eye in phase 3. This is expected,
not a defect — but it means "the presets changed" is not evidence something broke.
