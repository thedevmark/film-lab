# LUT colour pipeline — progress ledger

Plan: docs/superpowers/plans/2026-07-14-lut-colour-pipeline.md
Branch: lut-colour-pipeline
Base: b52cc3e

Task 1: complete (commits f96f6db..3151b94, review clean after fix; Minor deferred: srgb_encode does not clamp upper bound — intentional, highlight_rolloff handles >1)
Task 2: complete (commits a4d63a2..c97a0bf, review clean after fix — CAUGHT: np.maximum.accumulate made rolloff non-pointwise; added pointwise + raster-order tests)
Task 3: complete (commits 332ee8c..b178b40, review clean after test fixes — Pillow cannot write 16-bit RGB PNG, so lut.py hand-rolls a bounded stdlib codec that fails loudly; tests now pin byte order + filter wraparound)
  Minor deferred (T3): _unfilter_scanlines Average/Paeth is a pure-Python O(H*W) loop — fine at level 8, slow at level 12; _png_bit_depth indexes header[24] without a length guard
Task 4: complete (commit b317c1d, review via sabotage — CAUGHT MAJOR: brief's 6 tetrahedral masks paired 3 formulas with wrong regions; 9929/20000 pts wrong. Masks partitioned correctly so the partition check passed; identity/grey/lattice tests all blind to it. Fixed + added non-affine reference-equivalence test.)
  Verified independently: neutral drift 0.00 (tetra) vs 6.6e-2 (trilinear).
Task 5: complete (commit 83023b4, film.py correctly untouched — hit the same Pillow 16-bit RGB limit as T3, fixed by per-channel resize; RAW path pinned via rawpy monkeypatch asserting gamma=(1,1)/no_auto_bright/output_bps=16)
Task 6: complete (commits f91f2a6..3548ee4 — additive halation verified; CAUGHT: resolution-independence test was vacuous (0.05px sigma clamped to blur floor at both sizes, delta tolerance accepted noise), rewritten to fail under pixel-count sabotage)
Task 7: complete (commit d763f4e — grain verified: peaks at midtones 0.197 vs shadows 0.035 / highlights 0.021, monochrome, seeded; agent proactively caught the same vacuous-tolerance trap as T6)
Task 8: complete (commit 02c3927 - THE ATOMIC SWAP. apply_film_color/_load_image/add_grain/add_halation/additive apply_exposure + provenance comment deleted; filmlab wired in. Suite 84 -> 90, green. CAUGHT: the brief's pipeline-order test only recorded halation/lut/grain, so it passed under BOTH named sabotages (contrast-before-LUT, grain-before-contrast) - extended to record exposure+contrast, both sabotages now fail it. _require_numpy() deleted: film.py already imported PIL/Flask eagerly, so the lazy-numpy contract was already dead. DEVIATIONS: static/js/film.js slider ranges (px -> fractions, else every browser render clamps to max grain+halation) and README.md (documented the deleted function).)
  Step 8 (visual A/B gate vs DxO finals) NOT RUN - needs a real Kodak Gold LUT + the user photo library. STILL OWED.
  Open: grey-point scale is a no-op with GREY_SCENE == GREY_DISPLAY == 0.18, so RAW gets no scene-to-display correction; awaiting the calibration constant Step 8 would reveal.
Task 8: complete (commits 02c3927..08378e3 — ATOMIC SWAP: apply_film_color deleted, LUT pipeline live. 19-mutation review CAUGHT: get_lut negative-cached identity under the missing LUT's name (broke the extraction workflow + could fake a Step-8 failure); legacy pixel-unit presets clamped to MAXIMA on load; rolloff/grade_strength/seed wirings all untested. All fixed + pinned. 96 tests.)
