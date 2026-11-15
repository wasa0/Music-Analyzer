# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller の設定. 直接ではなく tools/build_exe.py から使う.

環境変数 MUSICA_VARIANT:
  lite  : Demucs なし (簡易分離のみ, 小さい)
  cpu / cu126 / cu130 : Demucs + PyTorch を同梱 (ビルド環境に入っている PyTorch をそのまま同梱)
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve().parent
VARIANT = os.environ.get("MUSICA_VARIANT", "lite")
WITH_TORCH = VARIANT != "lite"
NAME = "MUSICA"

datas = []
binaries = []
hiddenimports = collect_submodules("musica")

# deno (YouTube の JavaScript チャレンジ用ランタイム) を bin/ に同梱
import deno  # noqa: E402

binaries.append((deno.find_deno_bin(), "bin"))

# importlib.metadata で版を調べるもの (動作チェックの表示など)
for dist in ("numpy", "scipy", "librosa", "numba", "pyqtgraph", "soundfile", "yt-dlp", "yt-dlp-ejs", "imageio-ffmpeg"):
    datas += copy_metadata(dist)

excludes = ["tkinter", "IPython", "matplotlib", "pytest", "PyInstaller"]
if WITH_TORCH:
    datas += collect_data_files("demucs")          # remote/*.yaml, files.txt (モデル一覧)
    hiddenimports += collect_submodules("demucs")
    for dist in ("torch", "torchaudio", "demucs"):
        datas += copy_metadata(dist)
else:
    excludes += ["torch", "torchaudio", "torchvision", "demucs", "triton", "julius", "openunmix", "dora",
                 "lameenc", "diffq"]

a = Analysis(
    [str(ROOT / "tools" / "pyi_entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

icon = ROOT / "assets" / ("musica.ico" if sys.platform == "win32" else "musica.png")
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=False,
    strip=False,
    upx=False,
    console=False,               # GUI アプリ (Windows でコンソール窓を出さない)
    icon=str(icon) if icon.exists() else None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=NAME)
