# LUTs

Film colour renderings live here as HaldCLUT PNGs.

- `open/` — openly-licensed LUTs, shipped with the repo. Check each one's
  attribution before redistributing.
- `private/` — **gitignored.** Your own LUTs, including any you extract from an
  editor you have licensed. They stay on your disk.

A LUT is selected by name: `{"lut": "kodak_gold_200"}` loads
`luts/private/kodak_gold_200.png` if it exists, otherwise
`luts/open/kodak_gold_200.png`. If neither exists the colour stage is a no-op,
so the app still runs with no LUTs installed.

To make your own, see [../docs/extracting-a-lut.md](../docs/extracting-a-lut.md).
