# PyInstaller spec for CutForge — bundles the FastAPI app, templates, static assets and
# channel configs into a single windowed executable (dist/CutForge.exe).
#
# Build:  pyinstaller build.spec
#
# Notes:
# - opentimelineio + otio-fcp-adapter register adapters via entry points / plugin
#   manifests, so we collect their data files and hidden imports explicitly.
# - uvicorn/anyio pull in dynamically-imported modules; collect_submodules covers them.
# - librosa (reference rhythm analysis) pulls numba + llvmlite + scipy + soundfile/soxr.
#   These use lazy loading + native DLLs that defeat static analysis, so we collect their
#   binaries, data files and submodules explicitly. NOTE: this grows the exe by ~150-300 MB.
#   If the frozen exe crashes on the first rhythm analysis, UPX may have corrupted the
#   llvmlite/numba DLLs — see upx_exclude below.

from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules,
)

datas = [
    ("src/cutforge/ui/templates", "cutforge/ui/templates"),
    ("src/cutforge/ui/static", "cutforge/ui/static"),
    ("channels", "channels"),
]
# OTIO ships plugin manifests + the FCP adapter as data/entry points.
datas += collect_data_files("opentimelineio")
datas += collect_data_files("otio_fcp_adapter")
# librosa ships data files; soxr ships resampler tables.
datas += collect_data_files("librosa")
datas += collect_data_files("soxr", include_py_files=False)

binaries = []
binaries += collect_dynamic_libs("llvmlite")
binaries += collect_dynamic_libs("numba")
binaries += collect_dynamic_libs("soundfile")

hiddenimports = []
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("anyio")
hiddenimports += collect_submodules("opentimelineio")
hiddenimports += ["otio_fcp_adapter", "otio_fcp_adapter.fcp_xml"]
# librosa + its heavy, lazily-imported dependency tree.
hiddenimports += collect_submodules("librosa")
hiddenimports += collect_submodules("numba")
hiddenimports += collect_submodules("scipy")
hiddenimports += [
    "llvmlite", "llvmlite.binding", "soundfile", "audioread", "soxr",
    "lazy_loader", "sklearn.utils._typedefs",
]

block_cipher = None

a = Analysis(
    ["src/cutforge/ui/desktop.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pytest"],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="CutForge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # UPX can corrupt numba/llvmlite native DLLs — exclude them from compression.
    upx_exclude=["llvmlite*.dll", "*numba*.dll", "libopenblas*.dll", "libsndfile*.dll"],
    console=False,          # windowed app (no console)
    disable_windowed_traceback=False,
)
