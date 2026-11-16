# PyInstaller spec: builds a double-clickable app with no Python install needed.
#   pyinstaller packaging/eegvid.spec --noconfirm
# Output: dist/EEGVideoRecorder/ (Windows/Linux folder) or dist/EEG Video Recorder.app (macOS)
#
# macOS: BioSemi's library is Intel-only, so on Apple Silicon build the Intel app
# (runs under Rosetta, like ActiView-Mac) from an x86_64 environment, e.g.
#   arch -x86_64 .venv-x86_64/bin/pyinstaller packaging/eegvid.spec --noconfirm
# The target follows the running Python; override with EEGVID_TARGET_ARCH.
import os
import platform
import re
import sys
from pathlib import Path

root = Path(SPECPATH).parent
version = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.M).group(1)
drivers_dir = root / "packaging" / "drivers"
lib_suffix = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
drivers = [(str(p), "drivers") for p in drivers_dir.glob("*")
           if p.is_file() and (p.suffix.lower() == lib_suffix or ".so." in p.name and lib_suffix == ".so")]
# Framework bundles (macOS): keep the layout so @loader_path/Resources/libusb resolves.
# Collected as data so PyInstaller doesn't rewrite the library's install names.
framework_files = []
if sys.platform == "darwin":
    for fw in drivers_dir.glob("*.framework"):
        for f in fw.rglob("*"):
            rel = f.relative_to(fw)
            if f.is_file() and rel.parts[0] not in {"Versions", "Headers", "Modules"}:
                framework_files.append((str(f), str(Path("drivers") / fw.name / rel.parent)))
target_arch = os.environ.get("EEGVID_TARGET_ARCH") or (
    platform.machine() if sys.platform == "darwin" else None)
if not drivers and not framework_files:
    print("\n*** WARNING: no BioSemi library in packaging/drivers/ — the app will only "
          "work in simulator mode until a driver is supplied at runtime. ***\n")

a = Analysis(
    [str(root / "src" / "eegvid" / "__main__.py")],
    pathex=[str(root / "src")],
    binaries=drivers,
    datas=framework_files,
    hiddenimports=[],
    # Analysis-only deps are not needed in the recorder; keeps the download small.
    excludes=["mne", "matplotlib", "pandas", "scipy", "tkinter", "PyQt5", "PyQt6", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="EEGVideoRecorder",
          console=False, upx=False, target_arch=target_arch)
coll = COLLECT(exe, a.binaries, a.datas, name="EEGVideoRecorder", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="EEG Video Recorder.app",
        bundle_identifier="org.eegvid.recorder",
        info_plist={
            "NSCameraUsageDescription": "Records video of the participant during EEG sessions.",
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": version,
            "CFBundleVersion": version,
            "CFBundleDisplayName": "EEG Video Recorder",
        },
    )
