"""MUSICA_EX セットアップ (Windows / Linux / macOS 共通).

普段は setup_windows.bat (Windows) / setup.sh (Linux, macOS) から呼ばれる。標準ライブラリだけで動く。

  1. プロジェクトの .venv に Python 3.12 の仮想環境を作る (uv が Python ごと自動で用意)
  2. 依存ライブラリと、このアプリ本体 (musica-ex / musica-analyzer / musica-visualizer の起動用 exe) を入れる
  3. GPU を調べて、合った PyTorch + Demucs (高品質な音源分離) を入れる
  4. Demucs のモデルを先にダウンロードし、動作チェック (selftest) をする
  5. デスクトップ / スタートメニュー (Windows)、アプリ一覧 (Linux) にショートカットを作る

オプション:
  --gpu auto|cuda|cpu|none  Demucs 用 PyTorch の種類 (auto=自動判定, none=Demucs を入れない)
  --no-shortcuts            ショートカットを作らない
  --shortcut-dir DIR        ショートカットの作成先を変える (テスト用)
  --no-model                Demucs モデルを先にダウンロードしない
  --no-selftest             最後の動作チェックをしない
  --uninstall               作成したショートカットを削除する
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INFO_FILE = ROOT / "tools" / "install_info.json"
IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

TORCH_VERSION = "2.14.1"
TORCHAUDIO_VERSION = "2.11.0"
DEMUCS_VERSION = "4.0.1"
TORCH_INDEX = "https://download.pytorch.org/whl/{}"

APPS = [
    # (ファイル名に使う ID, 表示名, 引数, 説明)
    ("musica-ex", "MUSICA", ["-m", "musica.app"],
     "歌声アナライザーとビジュアライザーをタブで切り替え (解析した区間をパート別に可視化)"),
    ("musica-analyzer", "MUSICA 歌声アナライザー", ["-m", "musica.app", "--analyzer"],
     "歌声アナライザー単体: YouTube の歌声から調教の特徴 (しゃくり・ビブラート等) とコード進行を解析"),
    ("musica-visualizer", "MUSICA ビジュアライザー", ["-m", "musica.app", "--visualizer"],
     "ビジュアライザー単体: 解析せずにファイル / YouTube の倍音・スペクトラム・コードを可視化"),
]


# ------------------------------------------------------------------------------------ 表示
def say(msg: str = "") -> None:
    print(msg, flush=True)


def head(n: int, total: int, msg: str) -> None:
    say("")
    say(f"[{n}/{total}] {msg}")


def warn(msg: str) -> None:
    say(f"  ⚠ {msg}")


def run(cmd: list[str], env: dict | None = None, check: bool = True, cwd: Path | None = None) -> int:
    """コマンドを実行 (出力はそのまま画面に流す = ダウンロードの進捗が見える)."""
    e = dict(os.environ)
    if env:
        e.update(env)
    proc = subprocess.run([str(c) for c in cmd], env=e, cwd=str(cwd or ROOT))
    if check and proc.returncode != 0:
        raise RuntimeError(f"コマンドが失敗しました (終了コード {proc.returncode}): {' '.join(map(str, cmd))[:300]}")
    return proc.returncode


def capture(cmd: list[str], timeout: float = 30) -> str | None:
    try:
        kw: dict = {"capture_output": True, "timeout": timeout}
        if IS_WIN:
            kw["creationflags"] = 0x08000000      # CREATE_NO_WINDOW
        p = subprocess.run([str(c) for c in cmd], **kw)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return p.stdout.decode("utf-8", errors="replace")


# ------------------------------------------------------------------------------------ uv / venv
def find_uv() -> str:
    for cand in (os.environ.get("MUSICA_UV"), os.environ.get("UV"),
                 ROOT / "tools" / "uv" / ("uv.exe" if IS_WIN else "uv"), shutil.which("uv")):
        if cand and Path(cand).is_file():
            return str(cand)
    raise RuntimeError("uv が見つかりません。setup_windows.bat / setup.sh から実行してください。")


def venv_dir() -> Path:
    return ROOT / ".venv"


def venv_python(gui: bool = False) -> Path:
    if IS_WIN:
        return venv_dir() / "Scripts" / ("pythonw.exe" if gui else "python.exe")
    return venv_dir() / "bin" / "python"


def venv_works() -> bool:
    py = venv_python()
    if not py.is_file():
        return False
    out = capture([py, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
    return bool(out) and out.strip() == "3.12"


def create_venv(uv: str) -> None:
    if venv_works():
        say(f"  既存の仮想環境を使います: {venv_dir()}")
        return
    cmd = [uv, "venv", venv_dir(), "--python", "3.12", "--managed-python"]
    if venv_dir().exists():
        cmd.append("--clear")          # 壊れている / 別の Python で作られた場合は作り直す
    run(cmd)


# ------------------------------------------------------------------------------------ GPU 判定
def find_nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if IS_WIN:
        for p in (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
                  Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"):
            if p.is_file():
                return str(p)
    return None


def detect_nvidia() -> dict | None:
    """{'name', 'driver' (メジャー版), 'driver_str', 'cc' (例 8.6)} または None."""
    smi = find_nvidia_smi()
    if not smi:
        return None
    out = capture([smi, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"])
    if out is None:     # 古いドライバは compute_cap を問い合わせられない
        out = capture([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if not out or not out.strip():
        return None
    parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    try:
        drv = int(parts[1].split(".")[0])
    except (IndexError, ValueError):
        return None
    cc = None
    if len(parts) > 2:
        try:
            cc = float(parts[2])
        except ValueError:
            cc = None
    return {"name": parts[0], "driver": drv, "driver_str": parts[1], "cc": cc}


def choose_torch(mode: str) -> tuple[str | None, str]:
    """(variant, 説明). variant: 'cu130' / 'cu126' / 'cpu' / 'pypi' / None (Demucs を入れない)."""
    machine = platform.machine().lower()
    if mode == "none":
        return None, "Demucs は入れません (簡易分離のみ)"
    if IS_MAC:
        if machine in ("arm64", "aarch64"):
            return "pypi", "macOS (Apple Silicon): PyTorch 標準版 (GPU = MPS)"
        return None, "Intel Mac 向けの PyTorch が提供されていないため Demucs は入れません"
    if mode == "cpu":
        return "cpu", "CPU 版 PyTorch (指定)"
    gpu = detect_nvidia()
    if gpu is None:
        if mode == "cuda":
            warn("NVIDIA GPU (nvidia-smi) が見つかりません")
        return "cpu", "NVIDIA GPU が見つからないため CPU 版 PyTorch"
    desc = f"{gpu['name']} (ドライバ {gpu['driver_str']}" + (f", 世代 sm_{int(round(gpu['cc'] * 10))})" if gpu["cc"] else ")")
    drv, cc = gpu["driver"], gpu["cc"]
    min_cu126 = 528 if IS_WIN else 525
    if drv >= 580 and (cc is None or cc >= 7.5):
        return "cu130", f"{desc} → CUDA 13.0 版 PyTorch"
    if cc is not None and cc >= 10.0:
        warn(f"{gpu['name']} を GPU で使うには NVIDIA ドライバ 580 以降が必要です (現在 {gpu['driver_str']})。"
             "ドライバを更新してからもう一度セットアップすると GPU が使えます")
        return "cpu", f"{desc} → ドライバが古いため CPU 版 PyTorch"
    if drv >= min_cu126:
        return "cu126", f"{desc} → CUDA 12.6 版 PyTorch"
    warn(f"NVIDIA ドライバ {gpu['driver_str']} は古いため GPU を使えません (CUDA 12.6 には {min_cu126} 以降が必要)")
    return "cpu", f"{desc} → CPU 版 PyTorch"


def torch_args(variant: str) -> list[str]:
    if variant == "pypi":
        return [f"torch=={TORCH_VERSION}", f"torchaudio=={TORCHAUDIO_VERSION}", f"demucs=={DEMUCS_VERSION}"]
    return [f"torch=={TORCH_VERSION}+{variant}", f"torchaudio=={TORCHAUDIO_VERSION}+{variant}",
            f"demucs=={DEMUCS_VERSION}", "--extra-index-url", TORCH_INDEX.format(variant),
            "--index-strategy", "unsafe-best-match"]


def install_demucs(uv: str, variant: str) -> str | None:
    """GPU に合った PyTorch + Demucs を入れる. 失敗したら CPU 版 → PyPI 版の順に試す."""
    need_gb = 7 if variant.startswith("cu") else 2
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < need_gb:
        warn(f"空き容量が少なめです ({free_gb:.1f} GB)。PyTorch には約 {need_gb} GB 必要です")
    order = [variant] + [v for v in ("cpu", "pypi") if v != variant]
    if IS_MAC:
        order = ["pypi"]
    for v in order:
        say(f"  PyTorch ({v}) と Demucs を入れています... (初回は数分〜十数分かかります)")
        if run([uv, "pip", "install", "--python", venv_python(), *torch_args(v)], check=False) == 0:
            return v
        warn(f"PyTorch ({v}) を入れられませんでした。別の版を試します")
    return None


# ------------------------------------------------------------------------------------ ショートカット
def _ps_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _windows_cmdline(args: list[str]) -> str:
    return subprocess.list2cmdline([str(a) for a in args])


def shortcuts_windows(target_dir: Path | None) -> list[str]:
    icon = ROOT / "assets" / "musica.ico"
    pyw = venv_python(gui=True)
    py = venv_python()
    out_file = Path(tempfile.gettempdir()) / f"musica_shortcuts_{os.getpid()}.txt"
    if target_dir is None:
        desk = "[Environment]::GetFolderPath('Desktop')"
        progs = "(Join-Path ([Environment]::GetFolderPath('Programs')) 'MUSICA_EX')"
    else:
        desk = progs = _ps_quote(target_dir)
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "$ws = New-Object -ComObject WScript.Shell",
        f"$desk = {desk}",
        f"$progs = {progs}",
        "New-Item -ItemType Directory -Force -Path $progs | Out-Null",
        "$made = @()",
        "function Make($dir, $name, $target, $arguments, $desc) {",
        "  $p = Join-Path $dir ($name + '.lnk')",
        "  $s = $ws.CreateShortcut($p)",
        "  $s.TargetPath = $target",
        "  $s.Arguments = $arguments",
        f"  $s.WorkingDirectory = {_ps_quote(ROOT)}",
        f"  $s.IconLocation = {_ps_quote(str(icon) + ',0')}",
        "  $s.Description = $desc",
        "  $s.Save()",
        "  $script:made += $p",
        "}",
    ]
    for _id, name, args, desc in APPS:
        a = _ps_quote(_windows_cmdline(args))
        lines.append(f"Make $desk {_ps_quote(name)} {_ps_quote(pyw)} {a} {_ps_quote(desc)}")
        lines.append(f"Make $progs {_ps_quote(name)} {_ps_quote(pyw)} {a} {_ps_quote(desc)}")
    # 動作チェック (コンソールに結果を表示して止まる)
    chk = _windows_cmdline(["/k", _windows_cmdline([str(py), "-m", "musica.app", "--selftest", "--network"])])
    cmd_exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "cmd.exe"
    lines.append(f"Make $progs 'MUSICA 動作チェック' {_ps_quote(cmd_exe)} {_ps_quote(chk)} 'MUSICA の動作チェック'")
    lines.append(f"[IO.File]::WriteAllText({_ps_quote(out_file)}, ($made -join \"`n\"), "
                 "(New-Object System.Text.UTF8Encoding($false)))")
    script = "\r\n".join(lines)
    enc = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    ps = shutil.which("powershell") or str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" /
                                            "WindowsPowerShell" / "v1.0" / "powershell.exe")
    run([ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", enc])
    made = out_file.read_text(encoding="utf-8").splitlines() if out_file.exists() else []
    out_file.unlink(missing_ok=True)
    return [m for m in made if m.strip()]


def _desktop_exec_quote(arg: str) -> str:
    """Desktop Entry 仕様の Exec 引数のクォート."""
    arg = str(arg)
    reserved = set(" \t\n\"'\\><~|&;$*?#()`")
    if not any(ch in reserved for ch in arg):
        return arg.replace("%", "%%")
    inner = arg.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    # ファイル全体のエスケープ規則で \ をもう一段エスケープする
    return '"' + inner.replace("\\", "\\\\").replace("%", "%%") + '"'


def shortcuts_linux(target_dir: Path | None) -> list[str]:
    apps_dir = target_dir or Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "applications"
    apps_dir.mkdir(parents=True, exist_ok=True)
    icon = ROOT / "assets" / "musica.png"
    made = []
    for app_id, name, args, desc in APPS:
        exec_line = " ".join(_desktop_exec_quote(a) for a in [str(venv_python()), *args])
        text = "\n".join([
            "[Desktop Entry]", "Type=Application", f"Name={name}", f"Comment={desc}",
            f"Exec={exec_line}", f"Path={ROOT}", f"Icon={icon}", "Terminal=false",
            "Categories=AudioVideo;Audio;Music;", f"StartupWMClass={app_id}", "",
        ])
        p = apps_dir / f"{app_id}.desktop"
        p.write_text(text, encoding="utf-8")
        p.chmod(0o755)
        made.append(str(p))
    return made


def shortcuts_mac(target_dir: Path | None) -> list[str]:
    d = target_dir or ROOT
    made = []
    for _app_id, name, args, _desc in APPS:
        p = d / f"{name}.command"
        p.write_text("#!/bin/bash\n"
                     f"cd {json.dumps(str(ROOT))}\n"
                     f"exec {json.dumps(str(venv_python()))} " + " ".join(args) + "\n", encoding="utf-8")
        p.chmod(0o755)
        made.append(str(p))
    return made


def make_shortcuts(target_dir: Path | None) -> list[str]:
    if IS_WIN:
        return shortcuts_windows(target_dir)
    if IS_MAC:
        return shortcuts_mac(target_dir)
    return shortcuts_linux(target_dir)


# ------------------------------------------------------------------------------------ 情報ファイル
def load_info() -> dict:
    try:
        return json.loads(INFO_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_info(info: dict) -> None:
    INFO_FILE.parent.mkdir(parents=True, exist_ok=True)
    INFO_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


def uninstall() -> int:
    info = load_info()
    removed = 0
    for p in info.get("shortcuts", []):
        try:
            Path(p).unlink()
            removed += 1
            say(f"  削除: {p}")
        except FileNotFoundError:
            pass
        except OSError as exc:
            warn(f"削除できませんでした: {p} ({exc})")
    if IS_WIN and not info.get("shortcut_dir"):
        progs = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "MUSICA_EX"
        if progs.is_dir() and not any(progs.iterdir()):
            progs.rmdir()
    info["shortcuts"] = []
    save_info(info)
    say(f"ショートカットを {removed} 個削除しました。")
    say(f"アプリ本体を消すには、このフォルダ ({ROOT}) ごと削除してください。")
    say("ダウンロードした音源・解析結果は cache フォルダ (または ユーザーデータの MUSICA_EX) にあります。")
    return 0


# ------------------------------------------------------------------------------------ メイン
def main() -> int:
    ap = argparse.ArgumentParser(description="MUSICA_EX のセットアップ")
    ap.add_argument("--gpu", choices=["auto", "cuda", "cpu", "none"], default="auto")
    ap.add_argument("--no-shortcuts", action="store_true")
    ap.add_argument("--shortcut-dir", type=Path)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-selftest", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    args = ap.parse_args()

    if args.uninstall:
        return uninstall()

    t0 = time.time()
    say("=" * 64)
    say(" MUSICA_EX セットアップ")
    say(f" フォルダ: {ROOT}")
    say(f" OS: {platform.system()} {platform.release()} ({platform.machine()})")
    say("=" * 64)
    uv = find_uv()
    total = 5
    info = load_info()

    head(1, total, "Python 3.12 の仮想環境 (.venv) を準備しています")
    create_venv(uv)

    head(2, total, "ライブラリとアプリ本体を入れています")
    run([uv, "pip", "install", "--python", venv_python(), "-e", ROOT])

    head(3, total, "音源分離 (Demucs) 用の PyTorch を選んでいます")
    variant, why = choose_torch(args.gpu)
    say(f"  {why}")
    installed = install_demucs(uv, variant) if variant else None
    if variant and not installed:
        warn("Demucs を入れられませんでした。簡易分離だけで動作します (後でもう一度セットアップできます)")

    head(4, total, "モデルの準備と動作チェック")
    env = {}
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        env["QT_QPA_PLATFORM"] = "offscreen"
    if installed and not args.no_model:
        say("  Demucs モデル (htdemucs, 約 80MB) をダウンロードしています...")
        if run([venv_python(), "-c", "from demucs.pretrained import get_model; get_model('htdemucs'); print('  OK')"],
               env=env, check=False) != 0:
            warn("モデルを先にダウンロードできませんでした (初回の解析時に自動でダウンロードされます)")
    selftest_ok = None
    if not args.no_selftest:
        say("  動作チェック中 (初回は 30 秒ほどかかります)...")
        selftest_ok = run([venv_python(), "-m", "musica.app", "--selftest"], env=env, check=False) == 0
        if not selftest_ok:
            warn("動作チェックで失敗した項目があります。上の [FAIL] の行を確認してください")

    head(5, total, "ショートカットを作成しています")
    made: list[str] = []
    if args.no_shortcuts:
        say("  (作成しません)")
    else:
        try:
            made = make_shortcuts(args.shortcut_dir)
            for m in made:
                say(f"  作成: {m}")
        except Exception as exc:
            warn(f"ショートカットを作れませんでした: {exc}")

    info.update({"installed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": installed or "none",
                 "torch_reason": why, "python": str(venv_python()), "selftest_ok": selftest_ok,
                 "shortcuts": sorted(set(info.get("shortcuts", [])) | set(made)) if made else info.get("shortcuts", []),
                 "shortcut_dir": str(args.shortcut_dir) if args.shortcut_dir else None})
    save_info(info)

    say("")
    say("=" * 64)
    say(f" セットアップ完了 ({(time.time() - t0) / 60:.1f} 分)")
    say(f"  音源分離: {'Demucs (' + installed + ')' if installed else '簡易分離のみ'}")
    if made and IS_WIN:
        say("  デスクトップの「MUSICA」(歌声アナライザー + ビジュアライザー) から起動できます。")
        say("  単体版: 「MUSICA 歌声アナライザー」「MUSICA ビジュアライザー」(解析せずに可視化)")
    elif made and IS_MAC:
        say("  このフォルダの「MUSICA.command」をダブルクリックで起動できます (単体版の .command も同じ場所)。")
    elif made:
        say("  アプリ一覧の「MUSICA」から起動できます (単体版: 「MUSICA 歌声アナライザー」「MUSICA ビジュアライザー」)。")
    say(f"  コマンドで起動: \"{venv_python()}\" -m musica.app   (単体版は --analyzer / --visualizer)")
    say("=" * 64)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\n中断しました。")
        sys.exit(130)
    except Exception as exc:  # 利用者に分かる形で表示
        say("")
        say(f"エラー: {exc}")
        say("ネットワーク接続とディスクの空き容量を確認して、もう一度実行してください。")
        sys.exit(1)
