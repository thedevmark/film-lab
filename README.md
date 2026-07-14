<p align="center">
  <img src="static/img/logo.png" width="120" alt="Film Lab">
</p>

<h1 align="center">Film Lab</h1>

<p align="center">
  A small, self-hosted tool for putting a film look on digital photographs.<br>
  Python + Flask + numpy. Runs in your browser, on your machine, on your files.
</p>

---

Drop in a JPEG or a RAW, pick a preset, get a rendered photo back. The pipeline
models the parts of film that actually make it look like film: a color response,
grain, and halation — the red glow that blooms around highlights when light
punches through the emulsion and reflects off the film base.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Open <http://localhost:3100>.

RAW support (`.arw`, `.cr2`, `.cr3`, `.nef`, `.raf`, `.dng`, …) comes from
`rawpy` and is optional — JPEG/PNG/TIFF work without it.

## What's in the box

| Control | What it does |
| --- | --- |
| **Grade strength** | How far to blend in the film LUT |
| **Exposure** | EV stops — a multiply in linear light, so hue is preserved |
| **Contrast** | S-curve around mid-gray, applied *after* the LUT |
| **Grain** | Intensity, and size as a fraction of the short edge |
| **Halation** | Intensity, and radius as a fraction of the long edge |

Sizes are fractions rather than pixel counts so the look is the same on a
downscaled preview and on a full-resolution export.

Presets ship as built-ins and can't be overwritten. Anything you dial in
yourself can be saved alongside them, and lives at
`%LOCALAPPDATA%\film-lab\film_presets.json` on Windows or `./.appstate/` elsewhere.

## Where the color comes from

From a **3D LUT** — a measured mapping from every input color to an output
color, applied with tetrahedral interpolation. It is the same thing DxO FilmPack
and darktable's `lut3d` module do. There is no hand-tuned color math in the
pipeline, by design: a guess at what film does is a guess, however pleasant.

The pipeline, in order:

```
load -> (linear, scene|display)
exposure      linear x 2**EV
halation      linear, additive, red-dominant
[if scene]    grey-point scale — neutral, no look
rolloff       hue-preserving highlight compression
srgb encode   + clip to [0,1]
LUT           tetrahedral, strength blend
contrast      post-LUT user finish, 0 in every preset
grain         last — texture on a finished frame
-> JPEG
```

Contrast sits after the LUT because the CLUTs were authored against a neutral,
standard-contrast render; look-contrast in front of them would be counted twice.

**Kodak Gold 200 ships with the repo** (`luts/open/`), baked from
[spektrafilm](https://github.com/andreavolpato/spektrafilm) — a spectral
photochemical simulation built from measured datasheet density curves, not a
hand-tuned approximation. So it works out of the box.

You can add your own to `luts/private/`, which is gitignored — see
[docs/extracting-a-lut.md](docs/extracting-a-lut.md). This repo will not
redistribute profiles extracted from commercial software. If you own a licensed
copy of an editor, extracting a LUT from it for your own use is your business;
publishing that LUT is not something this project will do for you.

Still to come: batch — folder in, folder out, resumable.

## Files

- `film.py` — pipeline order, params, presets, routes
- `filmlab/` — `loader` (linear + scene/display state), `tone` (transfer functions,
  rolloff), `lut` (HaldCLUT + tetrahedral interpolation), `effects` (halation, grain), `blur`
- `app.py` — Flask host, port discovery, state paths
- `static/` — single-page UI
- `luts/` — HaldCLUT PNGs; `private/` is gitignored
- `tools/make_hald.py` — writes an identity HaldCLUT to render through an editor
- `tools/make_icon.py` — regenerates the icon from `static/img/app-icon.svg`
- `tests/` — `python -m unittest discover -s tests`

## Prior art worth reading

- [darktable](https://github.com/darktable-org/darktable) — `lut3d` and `grain` are the reference implementations
- [spektrafilm](https://github.com/andreavolpato/spektrafilm) (formerly agx-emulsion) — spectral photochemical film simulation in numpy, built from measured datasheet density curves
- [RawTherapee film simulation collection](https://rawpedia.rawtherapee.com/Film_Simulation) — the large open HaldCLUT set

## License

Code is [MIT](LICENSE).

**One exception:** `luts/open/kodak_gold_200.png` is baked from
[spektrafilm](https://github.com/andreavolpato/spektrafilm)'s measured film
profile and is **[CC BY-SA 4.0](luts/open/LICENSE)**, not MIT — spektrafilm's
license explicitly treats a LUT as a direct encoding of its profile data, so
share-alike follows it. See [luts/open/ATTRIBUTION.md](luts/open/ATTRIBUTION.md),
and [docs/baking-the-default-lut.md](docs/baking-the-default-lut.md) to reproduce it.

Photographs you make with it are yours. No copyleft reaches your images.

Any LUTs you add yourself carry their own licenses — check them.

Film stock names are trademarks of their respective owners. Nothing here is
affiliated with, endorsed by, or derived from Kodak, Fujifilm, or DxO.
