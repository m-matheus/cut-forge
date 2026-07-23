# PyInstaller spec for CutForge — bundles the FastAPI app, templates, static assets and
# channel configs into a single windowed executable (dist/CutForge.exe).
#
# Build:  pyinstaller build.spec
#
# Notes:
# - opentimelineio + otio-fcp-adapter register adapters via entry points / plugin
#   manifests, so we collect their data files and hidden imports explicitly.
# - uvicorn/anyio pull in dynamically-imported modules; collect_submodules covers them.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = [
    ("src/cutforge/ui/templates", "cutforge/ui/templates"),
    ("src/cutforge/ui/static", "cutforge/ui/static"),
    ("channels", "channels"),
]
# OTIO ships plugin manifests + the FCP adapter as data/entry points.
datas += collect_data_files("opentimelineio")
datas += collect_data_files("otio_fcp_adapter")

hiddenimports = []
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("anyio")
hiddenimports += collect_submodules("opentimelineio")
hiddenimports += ["otio_fcp_adapter", "otio_fcp_adapter.fcp_xml"]

block_cipher = None

a = Analysis(
    ["src/cutforge/ui/desktop.py"],
    pathex=["src"],
    binaries=[],
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
    console=False,          # windowed app (no console)
    disable_windowed_traceback=False,
)
