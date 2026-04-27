# Film Studio

Private personal photo style tool. Mathematical Fuji Gold 400 rendering with grain and halation. Browser-based UI on top of a small Flask host.

Extracted from `alert-alert` so this stays private and develops independently of the streaming tools.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:3100.

## Files

- `film.py` — pipeline (exposure, film color, contrast, halation, grain), preset persistence, Flask routes.
- `app.py` — minimal Flask host, port discovery, presets file location.
- `static/index.html` — single-page Film Studio UI.
- `static/js/film.js` — frontend module (presets, sliders, drop zone, process button).
- `static/css/style.css` — design tokens + shared shell + film section. Copied whole from alert-alert for now; can be slimmed later to film-only rules.

## State location

User presets are saved to:

- Windows: `%LOCALAPPDATA%\film-studio\film_presets.json`
- Other:   `./.appstate/film_presets.json`

## Built-ins

Built-in presets are defined in `film.py` and cannot be overwritten or deleted from the UI:

- `Ambient Film` — daylight/ambient sets, darker, denser, warm-biased.
- `Flash Film` — event/flash sets, brighter, punchier, color-neutral.
- `Fuji Gold 400 — Standard`
- `Fuji Gold 400 — Subtle`

## Notes

- RAW support requires `rawpy`.
- Long edge is capped at 6000 px to keep memory in check.
- Output is JPEG quality 95.
