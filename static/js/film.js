/**
 * Film Lab — frontend module for photo style processing.
 * Film effect: 3D LUT, grain, halation. Preset system + sliders.
 */
const FilmLab = (() => {
    // Default params — mirrors film.DEFAULT_PARAMS.
    //
    // exposure_bias is EV stops. grain_size is a fraction of the short edge and
    // halation_radius a fraction of the long edge, so the look is the same on a
    // preview and on a full-resolution export. Neither is a pixel count.
    const DEFAULT_PARAMS = {
        lut:                 "kodak_gold_200",
        grade_strength:      0.85,
        exposure_bias:       0.00,
        contrast_strength:   0.00,
        grain_intensity:     0.018,
        grain_size:          0.0015,
        halation_intensity:  0.45,
        halation_radius:     0.010,
        seed:                0,  // 0 == auto: the server derives it from the file
    };

    let currentParams = { ...DEFAULT_PARAMS };
    let presets = [];
    let selectedFile = null;
    let processing = false;
    let initialized = false;
    let sourcePreviewUrl = null;
    let outputPreviewUrl = null;
    let outputFilename = "film_processed.jpg";

    const PREVIEWABLE_SOURCE_EXTENSIONS = new Set([
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"
    ]);

    // ── Slider config ─────────────────────────────────────────────────────────
    const SLIDERS = [
        {
            id:      "film-grade-strength",
            param:   "grade_strength",
            valueId: "film-grade-strength-val",
            format:  v => `${Math.round(v * 100)}%`,
            min: 0, max: 1, step: 0.01,
        },
        {
            id:      "film-exposure-bias",
            param:   "exposure_bias",
            valueId: "film-exposure-bias-val",
            format:  v => `${v > 0 ? "+" : ""}${v.toFixed(2)} EV`,
            min: -3, max: 3, step: 0.05,
        },
        {
            id:      "film-contrast-strength",
            param:   "contrast_strength",
            valueId: "film-contrast-strength-val",
            format:  v => v === 0 ? "0" : (v > 0 ? `+${Math.round(v * 100)}` : `${Math.round(v * 100)}`),
            min: -0.30, max: 0.30, step: 0.01,
        },
        {
            id:      "film-grain-intensity",
            param:   "grain_intensity",
            valueId: "film-grain-intensity-val",
            format:  v => `${Math.round(v * 100)}%`,
            min: 0, max: 0.06, step: 0.002,
        },
        {
            id:      "film-grain-size",
            param:   "grain_size",
            valueId: "film-grain-size-val",
            format:  v => `${(v * 100).toFixed(2)}%`,  // of the short edge
            min: 0, max: 0.005, step: 0.0001,
        },
        {
            id:      "film-halation-intensity",
            param:   "halation_intensity",
            valueId: "film-halation-intensity-val",
            format:  v => `${Math.round(v * 100)}%`,
            min: 0, max: 1, step: 0.01,
        },
        {
            id:      "film-halation-radius",
            param:   "halation_radius",
            valueId: "film-halation-radius-val",
            format:  v => `${(v * 100).toFixed(1)}%`,  // of the long edge
            min: 0, max: 0.05, step: 0.001,
        },
    ];

    // ── Init ──────────────────────────────────────────────────────────────────
    function init() {
        if (initialized) {
            return;
        }
        initialized = true;
        _bindSliders();
        _bindFileInput();
        _bindPresetControls();
        _bindOutputControls();
        _clearSourcePreview();
        _clearOutputPreview();
        _bindBatchControls();
        loadPresets();
    }

    // ── Sliders ───────────────────────────────────────────────────────────────
    function _bindSliders() {
        for (const s of SLIDERS) {
            const el = document.getElementById(s.id);
            const valEl = document.getElementById(s.valueId);
            if (!el) continue;
            el.min  = s.min;
            el.max  = s.max;
            el.step = s.step;
            _setSlider(s, currentParams[s.param]);
            el.addEventListener("input", () => {
                const v = parseFloat(el.value);
                currentParams[s.param] = v;
                if (valEl) valEl.textContent = s.format(v);
            });
        }
    }

    function _setSlider(s, value) {
        const el = document.getElementById(s.id);
        const valEl = document.getElementById(s.valueId);
        if (el) el.value = value;
        if (valEl) valEl.textContent = s.format(value);
    }

    function _applyParams(params) {
        currentParams = { ...DEFAULT_PARAMS, ...params };
        for (const s of SLIDERS) {
            _setSlider(s, currentParams[s.param] ?? DEFAULT_PARAMS[s.param]);
        }
    }

    // ── File input ────────────────────────────────────────────────────────────
    function _bindFileInput() {
        const input = document.getElementById("film-file-input");
        const zone  = document.getElementById("film-drop-zone");
        const label = document.getElementById("film-file-label");
        const processBtn = document.getElementById("film-process-btn");

        if (input) {
            input.addEventListener("change", () => {
                _setSelectedFile(input.files[0] || null, label, processBtn);
            });
        }

        if (zone) {
            zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
            zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
            zone.addEventListener("drop", e => {
                e.preventDefault();
                zone.classList.remove("drag-over");
                const file = e.dataTransfer.files[0];
                if (file) {
                    _setSelectedFile(file, label, processBtn);
                }
            });
        }
    }

    function _setSelectedFile(file, labelEl, processBtn) {
        selectedFile = file;
        _updateFileLabel(labelEl, file);
        _renderSourcePreview(file);
        _clearOutputPreview();
        if (processBtn) processBtn.disabled = !selectedFile;
    }

    function _updateFileLabel(labelEl, file) {
        if (!labelEl) return;
        labelEl.textContent = file ? file.name : "Drop a photo here or click to browse";
    }

    function _bindOutputControls() {
        const downloadBtn = document.getElementById("film-download-btn");
        if (!downloadBtn) return;
        downloadBtn.addEventListener("click", downloadProcessedPhoto);
    }

    function _clearSourcePreview(message = "Load a photo to preview it here before processing.") {
        _revokeObjectUrl("source");
        const img = document.getElementById("film-source-preview");
        const placeholder = document.getElementById("film-source-preview-placeholder");
        if (img) {
            img.classList.add("hidden");
            img.removeAttribute("src");
        }
        if (placeholder) {
            placeholder.textContent = message;
            placeholder.classList.remove("hidden");
        }
    }

    function _clearOutputPreview(message = "Process the loaded photo to see the rendered JPEG here.") {
        _revokeObjectUrl("output");
        outputFilename = "film_processed.jpg";
        const img = document.getElementById("film-output-preview");
        const placeholder = document.getElementById("film-output-preview-placeholder");
        const downloadBtn = document.getElementById("film-download-btn");
        if (img) {
            img.classList.add("hidden");
            img.removeAttribute("src");
        }
        if (placeholder) {
            placeholder.textContent = message;
            placeholder.classList.remove("hidden");
        }
        if (downloadBtn) downloadBtn.disabled = true;
    }

    function _revokeObjectUrl(kind) {
        if (kind === "source" && sourcePreviewUrl) {
            URL.revokeObjectURL(sourcePreviewUrl);
            sourcePreviewUrl = null;
        }
        if (kind === "output" && outputPreviewUrl) {
            URL.revokeObjectURL(outputPreviewUrl);
            outputPreviewUrl = null;
        }
    }

    function _renderSourcePreview(file) {
        const img = document.getElementById("film-source-preview");
        const placeholder = document.getElementById("film-source-preview-placeholder");
        if (!img || !placeholder) return;

        if (!file) {
            _clearSourcePreview();
            return;
        }

        const dot = file.name.lastIndexOf(".");
        const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
        if (!PREVIEWABLE_SOURCE_EXTENSIONS.has(ext)) {
            _clearSourcePreview("This format can be processed, but browser preview is unavailable. Process it to see the rendered JPEG output.");
            return;
        }

        _revokeObjectUrl("source");
        sourcePreviewUrl = URL.createObjectURL(file);

        img.onload = () => {
            img.classList.remove("hidden");
            placeholder.classList.add("hidden");
        };
        img.onerror = () => {
            _clearSourcePreview("Preview failed for this file, but you can still process it.");
        };
        img.src = sourcePreviewUrl;
    }

    function _renderOutputPreview(blob, filename) {
        const img = document.getElementById("film-output-preview");
        const placeholder = document.getElementById("film-output-preview-placeholder");
        const downloadBtn = document.getElementById("film-download-btn");
        if (!img || !placeholder) return;

        _revokeObjectUrl("output");
        outputPreviewUrl = URL.createObjectURL(blob);
        outputFilename = filename;

        img.onload = () => {
            img.classList.remove("hidden");
            placeholder.classList.add("hidden");
            if (downloadBtn) downloadBtn.disabled = false;
        };
        img.onerror = () => {
            _clearOutputPreview("Processing finished, but preview could not be displayed.");
        };
        img.src = outputPreviewUrl;
    }

    function downloadProcessedPhoto() {
        if (!outputPreviewUrl) return;
        const a = document.createElement("a");
        a.href = outputPreviewUrl;
        a.download = outputFilename;
        document.body.appendChild(a);
        a.click();
        a.remove();
    }

    // ── Presets ───────────────────────────────────────────────────────────────
    function loadPresets() {
        fetch("/api/film/presets")
            .then(r => r.json())
            .then(data => {
                presets = data;
                _renderPresetSelect();
            })
            .catch(() => {});
    }

    function _renderPresetSelect() {
        const sel = document.getElementById("film-preset-select");
        if (!sel) return;
        sel.innerHTML = '<option value="">— choose preset —</option>';
        const builtins = presets.filter(p => p.builtin);
        const user = presets.filter(p => !p.builtin);

        if (builtins.length) {
            const g = document.createElement("optgroup");
            g.label = "Built-in";
            for (const p of builtins) {
                const opt = document.createElement("option");
                opt.value = p.name;
                opt.textContent = p.name;
                g.appendChild(opt);
            }
            sel.appendChild(g);
        }

        if (user.length) {
            const g = document.createElement("optgroup");
            g.label = "My Presets";
            for (const p of user) {
                const opt = document.createElement("option");
                opt.value = p.name;
                opt.textContent = p.name;
                g.appendChild(opt);
            }
            sel.appendChild(g);
        }
    }

    function _bindPresetControls() {
        const sel    = document.getElementById("film-preset-select");
        const saveBtn = document.getElementById("film-preset-save-btn");
        const delBtn  = document.getElementById("film-preset-delete-btn");

        if (sel) {
            sel.addEventListener("change", () => {
                const preset = presets.find(p => p.name === sel.value);
                if (preset) _applyParams(preset.params);
                if (delBtn) delBtn.disabled = !preset || preset.builtin;
            });
        }

        if (saveBtn) saveBtn.addEventListener("click", savePreset);
        if (delBtn)  delBtn.addEventListener("click", deleteSelectedPreset);
    }

    function savePreset() {
        const name = prompt("Preset name:");
        if (!name || !name.trim()) return;
        fetch("/api/film/presets", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: name.trim(), params: currentParams }),
        })
            .then(r => r.json())
            .then(data => {
                if (data.error) { alert(data.error); return; }
                loadPresets();
            })
            .catch(() => alert("Failed to save preset."));
    }

    function deleteSelectedPreset() {
        const sel = document.getElementById("film-preset-select");
        if (!sel || !sel.value) return;
        const name = sel.value;
        if (!confirm(`Delete preset "${name}"?`)) return;
        fetch(`/api/film/presets/${encodeURIComponent(name)}`, { method: "DELETE" })
            .then(r => r.json())
            .then(data => {
                if (data.error) { alert(data.error); return; }
                loadPresets();
            })
            .catch(() => alert("Failed to delete preset."));
    }

    // ── Process ───────────────────────────────────────────────────────────────
    function processPhoto() {
        if (!selectedFile) { alert("Please select a photo first."); return; }
        if (processing) return;

        processing = true;
        const btn = document.getElementById("film-process-btn");
        const statusEl = document.getElementById("film-status");
        if (btn) { btn.disabled = true; btn.textContent = "Processing…"; }
        if (statusEl) {
            statusEl.textContent = "Processing photo…";
            statusEl.classList.remove("error-msg");
        }

        const form = new FormData();
        form.append("file", selectedFile);
        form.append("params", JSON.stringify(currentParams));

        fetch("/api/film/process", { method: "POST", body: form })
            .then(r => {
                if (!r.ok) {
                    const contentType = r.headers.get("content-type") || "";
                    if (contentType.includes("application/json")) {
                        return r.json().then(d => Promise.reject(d.error || "Processing failed."));
                    }
                    return r.text().then(text => Promise.reject(text || `Processing failed (${r.status}).`));
                }
                return r.blob();
            })
            .then(blob => {
                const base = selectedFile.name.replace(/\.[^.]+$/, "");
                const filename = `${base}_film.jpg`;
                _renderOutputPreview(blob, filename);
                if (statusEl) statusEl.textContent = "Done — preview ready. Click Download JPG to save it.";
            })
            .catch(err => {
                if (statusEl) {
                    statusEl.textContent = typeof err === "string" ? err : "An error occurred.";
                    statusEl.classList.add("error-msg");
                }
            })
            .finally(() => {
                processing = false;
                if (btn) { btn.disabled = !selectedFile; btn.textContent = "Process Photo"; }
            });
    }

    // ── Batch ─────────────────────────────────────────────────────────────────
    // Applies the look currently dialled in above to a whole folder, one file at
    // a time, resumable. It sends `currentParams`, so what you see on the single
    // photo is what the batch renders.
    let batchJobId = null;
    let batchPollTimer = null;

    function _bindBatchControls() {
        const runBtn = document.getElementById("film-batch-run-btn");
        const cancelBtn = document.getElementById("film-batch-cancel-btn");
        if (runBtn) runBtn.addEventListener("click", startBatch);
        if (cancelBtn) cancelBtn.addEventListener("click", cancelBatch);
        _bindFolderPickers();
    }

    function _bindFolderPickers() {
        // Native folder dialogs exist only inside the desktop shell, where
        // pywebview exposes window.pywebview.api. In a plain browser the
        // Browse buttons stay hidden: a web page cannot obtain an absolute
        // folder path, and the typed-path inputs still work.
        const wire = () => {
            document.querySelectorAll(".film-browse-btn").forEach(btn => {
                btn.classList.remove("hidden");
                btn.addEventListener("click", async () => {
                    const input = document.getElementById(btn.dataset.target);
                    try {
                        const dir = await window.pywebview.api.pick_folder(input?.value.trim() || "");
                        if (dir && input) input.value = dir;
                    } catch (e) { /* dialog dismissed or bridge gone — keep the typed value */ }
                });
            });
        };
        if (window.pywebview && window.pywebview.api) wire();
        else window.addEventListener("pywebviewready", wire);
    }

    function startBatch() {
        if (batchJobId) return;
        const source = document.getElementById("film-batch-source")?.value.trim();
        const dest = document.getElementById("film-batch-dest")?.value.trim();
        const statusEl = document.getElementById("film-batch-status");

        if (!source || !dest) {
            _setBatchStatus("Enter a source folder and an output folder.", true);
            return;
        }

        _setBatchRunning(true);
        _setBatchStatus("Starting…", false);
        _renderBatchProgress(null);
        batchPollFailures = 0;

        fetch("/api/film/batch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source, dest, params: currentParams }),
        })
            .then(r => r.json().then(d => ({ ok: r.ok, d })))
            .then(({ ok, d }) => {
                if (!ok) return Promise.reject(d.error || "Could not start the batch.");
                batchJobId = d.id;
                _renderBatchProgress(d);
                _pollBatch();
            })
            .catch(err => {
                _setBatchRunning(false);
                _setBatchStatus(typeof err === "string" ? err : "Could not start the batch.", true);
            });
    }

    let batchPollFailures = 0;
    const BATCH_POLL_MAX_FAILURES = 10;  // ~10s of transient errors before giving up

    function _pollBatch() {
        if (!batchJobId) return;
        fetch(`/api/film/batch/${batchJobId}`)
            .then(r => {
                if (r.status === 404) {
                    // The job is gone — the server was probably restarted mid-run.
                    return Promise.reject({ terminal: true,
                        message: "Lost track of the batch — the server may have restarted." });
                }
                if (!r.ok) return Promise.reject({ message: `Status check failed (${r.status}).` });
                return r.json();
            })
            .then(state => {
                batchPollFailures = 0;
                _renderBatchProgress(state);
                if (state.status === "running") {
                    batchPollTimer = setTimeout(_pollBatch, 500);
                    return;
                }
                _finishBatch(state);
            })
            .catch(err => {
                if (err && err.terminal) {
                    _setBatchRunning(false);
                    batchJobId = null;
                    _setBatchStatus(err.message, true);
                    return;
                }
                batchPollFailures += 1;
                if (batchPollFailures >= BATCH_POLL_MAX_FAILURES) {
                    _setBatchRunning(false);
                    batchJobId = null;
                    _setBatchStatus("Lost contact with the batch. It may still be running on the server.", true);
                    return;
                }
                batchPollTimer = setTimeout(_pollBatch, 1000);
            });
    }

    function _finishBatch(state) {
        _setBatchRunning(false);
        batchJobId = null;
        const parts = [`${state.done} rendered`];
        if (state.skipped) parts.push(`${state.skipped} skipped`);
        if (state.failed) parts.push(`${state.failed} failed`);
        const summary = parts.join(", ");
        const label = {
            done: `Done — ${summary}.`,
            cancelled: `Cancelled — ${summary}.`,
            error: state.error || "The batch stopped unexpectedly.",
        }[state.status] || summary;
        _setBatchStatus(label, state.status === "error");
    }

    function cancelBatch() {
        if (!batchJobId) return;
        _setBatchStatus("Cancelling…", false);
        fetch(`/api/film/batch/${batchJobId}/cancel`, { method: "POST" }).catch(() => {});
    }

    function _renderBatchProgress(state) {
        const bar = document.getElementById("film-batch-bar");
        const count = document.getElementById("film-batch-count");
        if (!bar || !count) return;
        if (!state || !state.total) {
            bar.style.width = "0%";
            count.textContent = state ? "0 / 0" : "";
            return;
        }
        const finished = state.done + state.skipped + state.failed;
        bar.style.width = `${Math.round((finished / state.total) * 100)}%`;
        const tail = state.current ? ` · ${state.current}` : "";
        count.textContent = `${finished} / ${state.total}${tail}`;
    }

    function _setBatchRunning(running) {
        const runBtn = document.getElementById("film-batch-run-btn");
        const cancelBtn = document.getElementById("film-batch-cancel-btn");
        if (runBtn) { runBtn.disabled = running; runBtn.textContent = running ? "Running…" : "Run Batch"; }
        if (cancelBtn) cancelBtn.disabled = !running;
        if (!running && batchPollTimer) { clearTimeout(batchPollTimer); batchPollTimer = null; }
    }

    function _setBatchStatus(message, isError) {
        const el = document.getElementById("film-batch-status");
        if (!el) return;
        el.textContent = message;
        el.classList.toggle("error-msg", !!isError);
    }

    return { init, loadPresets, processPhoto, savePreset, startBatch, cancelBatch };
})();
