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
| **Grade strength** | How far to push the film color response |
| **Exposure** | Brightness offset |
| **Contrast** | S-curve around mid-gray |
| **Grain** | Intensity and size, weighted by luminance |
| **Halation** | Intensity and radius of the red highlight bloom |

Presets ship as built-ins and can't be overwritten. Anything you dial in
yourself can be saved alongside them, and lives at
`%LOCALAPPDATA%\film-lab\film_presets.json` on Windows or `./.appstate/` elsewhere.

## Where the color comes from — and where it's going

Be aware of what this currently is. `apply_film_color()` in `film.py` is
**hand-tuned math, not a measured film profile**: per-channel gamma powers, a
Gaussian midtone-warmth mask, a shadow crossover term. It produces a pleasant
warm negative-film rendering, and it is emphatically an approximation.

Real film emulation — what DxO FilmPack and darktable's `lut3d` module do — uses
a **3D LUT**: a measured mapping from every input color to an output color. That
is the direction this is headed:

- [ ] Replace the hand-tuned grade with a HaldCLUT loader (trilinear interpolation)
- [ ] Move exposure and halation into **linear light**, where they're physical —
      exposure is a multiply (`× 2^EV`), not the additive offset it is today,
      and halation is per-channel scatter with σ<sub>R</sub> > σ<sub>G</sub> > σ<sub>B</sub>
- [ ] Apply the LUT *after* tone mapping, in gamma-encoded sRGB, clipped to
      [0,1] — which is where film LUTs are defined and where darktable applies them
- [ ] Weight grain toward the **midtones** via a paper-response curve, rather
      than toward the shadows as it is now
- [ ] Batch: folder in, folder out, resumable

LUTs are **bring-your-own**. This repo will ship a loader and an openly-licensed
default, and will not redistribute profiles extracted from commercial software.
If you own a licensed copy of an editor, extracting a LUT from it for your own
use is your business; publishing that LUT is not something this project will do
for you.

## Files

- `film.py` — the pipeline (exposure, color, contrast, halation, grain), presets, routes
- `app.py` — Flask host, port discovery, state paths
- `static/` — single-page UI
- `tools/make_icon.py` — regenerates the icon from `static/img/app-icon.svg`
- `tests/` — `python -m unittest discover -s tests`

## Prior art worth reading

- [darktable](https://github.com/darktable-org/darktable) — `lut3d` and `grain` are the reference implementations
- [spektrafilm](https://github.com/andreavolpato/spektrafilm) (formerly agx-emulsion) — spectral photochemical film simulation in numpy, built from measured datasheet density curves
- [RawTherapee film simulation collection](https://rawpedia.rawtherapee.com/Film_Simulation) — the large open HaldCLUT set

## License

Code is [MIT](LICENSE). Any LUTs or profiles you add carry their own licenses —
check them.

Film stock names are trademarks of their respective owners. Nothing here is
affiliated with, endorsed by, or derived from Kodak, Fujifilm, or DxO.
