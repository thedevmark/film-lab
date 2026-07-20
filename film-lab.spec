# PyInstaller spec — builds the single-file Windows exe.
#
#     pip install -r requirements.txt pyinstaller
#     pyinstaller film-lab.spec
#
# Output lands in dist/film-lab.exe. The same spec is what the release
# workflow (.github/workflows/release.yml) runs on every v* tag.
#
# What ships inside the bundle:
#   static/     — the single-page UI
#   luts/open/  — Kodak Gold 200 plus its CC BY-SA license and attribution
#                 (the license files travel with the LUT deliberately; do not
#                 trim them for size)
# luts/private/ is NEVER bundled — user LUTs may be derived from commercial
# software and must not be redistributed. A frozen build looks for a luts/
# folder beside the exe for those (see film._lut_dirs).

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("static", "static"),
        ("luts/open", "luts/open"),
    ],
    # pywebview picks its Windows backend (WinForms + WebView2) at runtime via
    # dynamic import, which static analysis misses.
    hiddenimports=[
        "webview.platforms.winforms",
        "webview.platforms.edgechromium",
        "clr",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="film-lab",
    icon="static/favicon.ico",
    console=False,  # standalone desktop app: native window, no terminal
    debug=False,
    strip=False,
    upx=False,
    bootloader_ignore_signals=False,
    disable_windowed_traceback=False,
)
