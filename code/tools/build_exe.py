"""実行ファイル版 (Python 不要で動くフォルダ) を PyInstaller で作る.

    build_exe_windows.bat                     (Windows)
    ./build_exe.sh                            (Linux / macOS)
    ... --variant lite   Demucs なし (簡易分離のみ, 小さい)      ← 既定
    ... --variant cpu    Demucs + CPU 版 PyTorch
    ... --variant cu130  Demucs + CUDA 13.0 版 PyTorch (NVIDIA ドライバ 580 以降, 非常に大きい)
    ... --variant cu126  Demucs + CUDA 12.6 版 PyTorch
    ... --zip            配布用の zip も作る

PyInstaller はクロスコンパイルできないので、Windows 用の exe は Windows 上で作る
(GitHub Actions の .github/workflows/build-windows.yml でも作れる)。

出力: dist/<variant>/MUSICA/ に MUSICA(.exe) (統合版) と単体版の MUSICA Analyzer(.exe) / MUSICA Visualizer(.exe)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import install as inst  # noqa: E402  (tools/install.py の関数を使い回す)

ROOT = inst.ROOT
IS_WIN = inst.IS_WIN
EXE_SUFFIX = ".exe" if IS_WIN else ""


def say(msg: str = "") -> None:
    print(msg, flush=True)


def build_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def prepare_venv(uv: str, variant: str, fresh: bool) -> Path:
    venv = ROOT / "build" / f"venv-{variant}"
    py = build_python(venv)
    if fresh and venv.exists():
        shutil.rmtree(venv)
    if not py.is_file():
        inst.run([uv, "venv", venv, "--python", "3.12", "--managed-python"])
    # アプリ本体 (通常インストール) + PyInstaller
    inst.run([uv, "pip", "install", "--python", py, f"{ROOT}[build]"])
    if variant != "lite":
        inst.run([uv, "pip", "install", "--python", py, *inst.torch_args(variant)])
    else:
        # lite に PyTorch が混ざらないように (以前の full ビルドの残りなど)
        inst.run([uv, "pip", "uninstall", "--python", py, "torch", "torchaudio", "demucs"], check=False)
    return py


def ensure_icon(py: Path) -> None:
    ico = ROOT / "assets" / "musica.ico"
    if ico.exists() and (ROOT / "assets" / "musica.png").exists():
        return
    inst.run([py, "-m", "musica.gui.icon", ico], env={"QT_QPA_PLATFORM": "offscreen"})


def run_pyinstaller(py: Path, variant: str) -> Path:
    dist = ROOT / "dist" / variant
    work = ROOT / "build" / f"work-{variant}"
    inst.run([py, "-m", "PyInstaller", "--noconfirm", "--clean", "--log-level", "WARN",
              "--distpath", dist, "--workpath", work, ROOT / "tools" / "musica.spec"],
             env={"MUSICA_VARIANT": variant})
    app_dir = dist / "MUSICA"
    exe = app_dir / f"MUSICA{EXE_SUFFIX}"
    if not exe.is_file():
        raise RuntimeError(f"実行ファイルができていません: {exe}")
    # 単体版用: 同じ exe を別名で置く (起動したファイル名でモードを切り替える。musica/app.py)
    for name in ("MUSICA Analyzer", "MUSICA Visualizer"):
        shutil.copy2(exe, app_dir / f"{name}{EXE_SUFFIX}")
    return app_dir


def write_readme(app_dir: Path, variant: str) -> None:
    sep = {"lite": "簡易分離のみ (Demucs なし)", "cpu": "Demucs (CPU)", "cu126": "Demucs (NVIDIA GPU, CUDA 12.6)",
           "cu130": "Demucs (NVIDIA GPU, CUDA 13.0 / ドライバ 580 以降)"}.get(variant, variant)
    text = f"""MUSICA_EX {_version()}  ({sep})

  MUSICA{EXE_SUFFIX}             歌声アナライザー + ビジュアライザー (タブで切り替え。解析した区間をパート別に可視化)
  MUSICA Analyzer{EXE_SUFFIX}    歌声アナライザー単体 (YouTube / ファイルの歌声・楽器分離・コード)
  MUSICA Visualizer{EXE_SUFFIX}  ビジュアライザー単体 (解析せずにファイル / YouTube を可視化)

インストールは不要です。このフォルダごと好きな場所に置いて、上のファイルを起動してください。
(Windows で「PC が保護されました」と出たら「詳細情報」→「実行」。署名の無いアプリのため表示されます)

ダウンロードした音源・解析結果・ログ (cache) と Demucs のモデル (torch) は次の場所に保存されます:
  Windows: %LOCALAPPDATA%\\MUSICA_EX     macOS: ~/Library/Application Support/MUSICA_EX
  Linux:   ~/.local/share/MUSICA_EX
アンインストールはこのフォルダと上の保存先を削除するだけです。
動作チェック: アナライザーのメニュー「ヘルプ → 動作チェック」
"""
    (app_dir / "README.txt").write_text(text.replace("\n", "\r\n") if IS_WIN else text, encoding="utf-8")


def _version() -> str:
    for line in (ROOT / "musica" / "__init__.py").read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=")[1].strip().strip("\"'")
    return "0"


def smoke_test(app_dir: Path, with_demucs: bool) -> bool:
    """出来上がった実行ファイルで動作チェック (--selftest) を走らせる."""
    exe = app_dir / f"MUSICA{EXE_SUFFIX}"
    with tempfile.TemporaryDirectory(prefix="musica_selftest_") as tmp:
        env = dict(os.environ, MUSICA_CACHE=str(Path(tmp) / "selftest_cache"))
        if sys.platform.startswith("linux") and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
            env["QT_QPA_PLATFORM"] = "offscreen"
        args = [str(exe), "--selftest"]
        if with_demucs:
            args.append("--demucs")          # 同梱した PyTorch / Demucs で実際に分離できるか
        say(f"  {exe.name} {' '.join(args[1:])} を実行中...")
        proc = subprocess.run(args, env=env, capture_output=True, timeout=1800)
        report = Path(tmp) / "selftest_cache" / "selftest.txt"
        if report.exists():
            for line in report.read_text(encoding="utf-8").splitlines():
                if not line.startswith("       "):           # トレースバックの行は省略
                    say("    " + line[:200])
        else:
            say(proc.stdout.decode("utf-8", errors="replace")[-3000:])
            say(proc.stderr.decode("utf-8", errors="replace")[-3000:])
        return proc.returncode == 0


def make_zip(app_dir: Path, variant: str) -> Path:
    plat = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    out = app_dir.parent / f"MUSICA_EX-{_version()}-{plat}-{inst.platform.machine().lower()}-{variant}.zip"
    say(f"  zip を作成中: {out.name}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for p in sorted(app_dir.rglob("*")):
            arc = Path("MUSICA") / p.relative_to(app_dir)
            if p.is_symlink() or p.is_file():
                info = zipfile.ZipInfo.from_file(p, arc)
                info.compress_type = zipfile.ZIP_DEFLATED
                with open(p, "rb") as fh:
                    zf.writestr(info, fh.read(), compresslevel=6)
    return out


def dir_size(p: Path) -> float:
    """MB (PyInstaller が Linux/macOS で作るシンボリックリンクは二重に数えない)."""
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description="MUSICA_EX の実行ファイル版を作る")
    ap.add_argument("--variant", choices=["lite", "cpu", "cu126", "cu130"], default="lite")
    ap.add_argument("--zip", action="store_true", help="配布用 zip も作る")
    ap.add_argument("--fresh", action="store_true", help="ビルド用の仮想環境を作り直す")
    ap.add_argument("--no-test", action="store_true", help="ビルド後の動作チェックをしない")
    args = ap.parse_args()

    t0 = time.time()
    uv = inst.find_uv()
    say(f"[1/4] ビルド用の環境を準備 ({args.variant})")
    py = prepare_venv(uv, args.variant, args.fresh)
    ensure_icon(py)
    say("[2/4] PyInstaller で実行ファイルを作成 (数分かかります)")
    app_dir = run_pyinstaller(py, args.variant)
    write_readme(app_dir, args.variant)
    say(f"  → {app_dir}  ({dir_size(app_dir):.0f} MB)")
    ok = True
    if not args.no_test:
        say("[3/4] 動作チェック")
        ok = smoke_test(app_dir, with_demucs=args.variant != "lite")
        say("  動作チェック: " + ("OK" if ok else "失敗あり (上の [FAIL] を確認)"))
    if args.zip:
        say("[4/4] 配布用 zip")
        z = make_zip(app_dir, args.variant)
        say(f"  → {z}  ({z.stat().st_size / 1e6:.0f} MB)")
    say(f"完了 ({(time.time() - t0) / 60:.1f} 分)")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"エラー: {exc}", flush=True)
        sys.exit(1)
