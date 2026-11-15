"""共通の定数とパス設定 (Windows / macOS / Linux 共通)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "MUSICA_EX"
FROZEN = bool(getattr(sys, "frozen", False))      # PyInstaller でビルドした実行ファイルか

PROJECT_DIR = Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """OS ごとのユーザーデータの置き場所."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / APP_NAME


def _writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _default_cache_dir() -> Path:
    """ソースから動かすときはプロジェクト内の cache/、実行ファイル版や
    書き込めない場所 (Program Files 等) に置いたときはユーザーデータフォルダを使う."""
    if os.environ.get("MUSICA_CACHE"):
        return Path(os.environ["MUSICA_CACHE"])
    local = PROJECT_DIR / "cache"
    if not FROZEN and _writable(local):
        return local
    return user_data_dir() / "cache"


# キャッシュ (ダウンロード音源 / 分離結果 / 解析結果)
CACHE_DIR = _default_cache_dir()
DOWNLOAD_DIR = CACHE_DIR / "downloads"
RESULT_DIR = CACHE_DIR / "results"
LOG_FILE = CACHE_DIR / "musica.log"          # 動作チェックのレポート (selftest.txt) も同じ場所

# 音声処理パラメータ
IO_SR = 44100          # ダウンロード音源・分離処理のサンプリング周波数 (Demucs は 44.1kHz)
ANALYSIS_SR = 22050    # ピッチ・倍音解析用
HOP_SEC = 0.01         # 解析フレーム間隔 10ms (= 100 fps)
HOP = int(round(ANALYSIS_SR * HOP_SEC))  # 220 samples

# f0 探索範囲のプリセット (Hz)
VOICE_PRESETS = {
    "auto": ("自動 (C2〜C6)", 65.0, 1100.0),
    "male": ("男声 (C2〜A5)", 65.0, 900.0),
    "female": ("女声 (G2〜C6)", 98.0, 1300.0),
}

# 日本語が表示できるフォントの候補 (見つかった最初のものを使う)
APP_FONT_FAMILIES = [
    "Yu Gothic UI", "Meiryo UI", "Meiryo", "MS UI Gothic",          # Windows
    "Hiragino Sans", "Hiragino Kaku Gothic ProN",                    # macOS
    "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "TakaoPGothic",  # Linux
]


def ensure_dirs() -> None:
    for d in (CACHE_DIR, DOWNLOAD_DIR, RESULT_DIR):
        d.mkdir(parents=True, exist_ok=True)
