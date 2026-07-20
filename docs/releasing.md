# Cutting a release

The whole process is pushing a tag:

```bash
git tag v0.3.0
git push origin v0.3.0
```

That triggers `.github/workflows/release.yml`, which on a Windows runner:

1. runs the test suite,
2. builds `dist/film-lab.exe` from `film-lab.spec` (PyInstaller, one file),
3. smoke-tests the exe — starts it and waits for the UI to answer HTTP 200,
4. creates the GitHub release with generated notes and the exe attached.

Re-running the workflow for an existing tag re-uploads the exe over the old
asset instead of failing.

## Building the exe locally

```bash
pip install -r requirements.txt pyinstaller
python -m PyInstaller film-lab.spec
```

Output is `dist/film-lab.exe`. Everything about what gets bundled — the UI,
the open LUTs and their licenses, and why `luts/private/` never ships — is
documented in `film-lab.spec` itself. Keep the spec as the single source of
truth for the build; don't add flags on the command line.
