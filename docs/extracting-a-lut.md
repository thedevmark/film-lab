# Extracting a LUT from an editor you have licensed

A 3D LUT is just a mapping from every input colour to an output colour. So if
you render an image *containing every colour* through a colour tool, the result
**is** that tool's colour transform. That image is called an identity HaldCLUT.

This is for your own use, on software you have licensed. Do not redistribute the
result — the colour science in it is not yours to give away.

## 1. Generate the identity Hald

```bash
python tools/make_hald.py --level 8 --out identity_hald_8.png
```

That is a 512x512 16-bit PNG containing all 262,144 colours of a 64³ cube.

## 2. Render it through your editor

Open it, apply **only** the film emulation preset, and export it — same size, no
resize, no crop, PNG.

**Turn everything else off.** This matters more than it sounds:

- **Grain — off.** This is the trap. Grain is *noise*, and the Hald's pixels are
  the LUT's lattice nodes. Leaving grain on gives every node an independent noise
  sample: you get a LUT that is noise-corrupted at every entry, and it will look
  "filmic" enough that you will not notice for weeks.
- **Halation, vignetting, blur — off.** A 3D LUT is a *pointwise* map. It cannot
  represent any effect that depends on neighbouring pixels, so these cannot be
  captured; they would only smear the table. Film Lab renders them itself.
- **Auto-corrections — off.** Smart Lighting, ClearView, auto-exposure, lens
  corrections, denoising. Each one contaminates the table with a transform that
  has nothing to do with the film.
- **Sharpening — off.** Same reason as halation.

## 3. Install it

Save the exported PNG as `luts/private/<name>.png` and select it with
`{"lut": "<name>"}`. `luts/private/` is gitignored.

## 4. Check it

Apply it to a photo at `grade_strength: 1.0` and look at a neutral grey card or
a white wall. If it has taken on a colour cast, something was left on in step 2.
