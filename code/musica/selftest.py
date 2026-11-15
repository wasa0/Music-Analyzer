"""動作チェック (自己診断).

    python -m musica.app --selftest            # オフラインでできる確認
    python -m musica.app --selftest --network  # YouTube 検索・取得も確認
    MUSICA.exe --selftest                      # 実行ファイル版 (結果はログファイルにも保存)

依存ライブラリ・ffmpeg・deno・音声出力・GPU・解析処理 (ピッチ/コード/簡易分離) を順に確かめる。
"""
from __future__ import annotations

import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__, config

Check = tuple[str, Callable[[], str]]


def _versions() -> str:
    import importlib.metadata as md

    import PySide6

    names = ["numpy", "scipy", "librosa", "numba", "pyqtgraph", "soundfile", "yt-dlp", "yt-dlp-ejs",
             "torch", "demucs"]
    out = [f"PySide6 {PySide6.__version__}"]
    for n in names:
        try:
            out.append(f"{n} {md.version(n)}")
        except md.PackageNotFoundError:
            out.append(f"{n} -")
    return ", ".join(out)


def _ffmpeg() -> str:
    from .audio_io import find_ffmpeg, run_quiet

    exe = find_ffmpeg()
    proc = run_quiet([exe, "-version"])
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg を実行できません: {exe}")
    return proc.stdout.decode(errors="ignore").splitlines()[0][:60] + f"  ({exe})"


def _deno() -> str:
    from .audio_io import run_quiet
    from .youtube import _deno_path

    exe = _deno_path()
    if not exe:
        raise RuntimeError("deno が見つかりません (YouTube の取得に必要)")
    proc = run_quiet([exe, "--version"])
    if proc.returncode != 0:
        raise RuntimeError(f"deno を実行できません: {exe}")
    return proc.stdout.decode(errors="ignore").splitlines()[0] + f"  ({exe})"


def _synth(sr: int = 22050) -> tuple[np.ndarray, np.ndarray]:
    """C→G→Am→F の伴奏と、ビブラート付きのメロディ (A4) を合成."""
    t = np.arange(int(sr * 8.0)) / sr
    harm = np.zeros_like(t)
    for i, chord in enumerate(([60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60])):
        seg = (t >= i * 2.0) & (t < (i + 1) * 2.0)
        for m in chord + [chord[0] - 12]:
            f = 440 * 2 ** ((m - 69) / 12)
            harm[seg] += sum(np.sin(2 * np.pi * f * k * t[seg]) / k ** 1.5 for k in range(1, 6))
    harm *= 0.05
    f0 = 440 * 2 ** ((40 * np.sin(2 * np.pi * 5.5 * t)) / 1200)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    voice = 0.2 * sum(np.sin(k * ph) / k for k in range(1, 10)) * (t > 0.5) * (t < 7.5)
    return voice.astype(np.float32), harm.astype(np.float32)


def _pitch() -> str:
    from .features import analyze_vibrato
    from .pitch import harmonic_analysis, hz_to_midi, track_f0

    sr, hop = config.ANALYSIS_SR, config.HOP
    voice, _h = _synth(sr)
    f0, _vp = track_f0(voice, sr, 65.0, 1100.0, hop)
    cur = harmonic_analysis(voice, sr, f0, hop)
    midi = hz_to_midi(cur["f0"])
    m = midi[np.isfinite(midi)]
    if m.size < 100:
        raise RuntimeError("ピッチがほとんど検出できませんでした")
    vib = analyze_vibrato((midi[100:700] - 69) * 100)
    if abs(np.median(m) - 69) > 0.3 or not vib["vib"]:
        raise RuntimeError(f"ピッチ解析の結果が想定外です (中央値 {np.median(m):.2f}, vib {vib})")
    return f"A4 を検出 (中央値 MIDI {np.median(m):.2f}), ビブラート {vib['vib_rate_hz']:.1f} Hz ±{vib['vib_extent_cents']:.0f} cent"


def _chords() -> str:
    from .chords import analyze_harmony

    sr = config.ANALYSIS_SR
    _v, harm = _synth(sr)
    res = analyze_harmony(harm, sr)
    labels = []
    for c in res["chords"]:
        if c["root"] is not None and (not labels or labels[-1] != c["label"]):
            labels.append(c["label"])
    want = ["C", "G", "Am", "F"]
    if [x for x in labels if x in want][:4] != want:
        raise RuntimeError(f"コード認識の結果が想定外です: {labels}")
    return f"{' → '.join(labels)} (キー {res['key']['name']})"


def _separation() -> str:
    from .separation import separate

    sr = config.IO_SR
    voice, harm = _synth(sr)
    y = np.stack([voice + harm, voice + harm])
    stems = separate(y, sr, method="simple")
    return "簡易分離 OK: " + ", ".join(f"{k} {v.shape}" for k, v in stems.items())


def _pipeline() -> str:
    """解析の全工程 (分離 → ピッチ → ノート → 特徴量 → コード → 保存/読み込み) を合成音で通す."""
    import shutil
    import tempfile

    from .analysis import AnalysisResult, run_analysis
    from .audio_io import save_wav

    sr = config.IO_SR
    voice, harm = _synth(sr)
    with tempfile.TemporaryDirectory(prefix="musica_pipe_") as tmp:
        wav = save_wav(Path(tmp) / "selftest.wav", np.stack([voice + harm, voice + harm]), sr)
        res = run_analysis(wav, {"video_id": "_selftest", "title": "selftest"}, 0.0, 8.0, method="simple")
        try:
            again = AnalysisResult.load(res.folder)
            if not res.notes or len(again.notes) != len(res.notes):
                raise RuntimeError(f"ノートの保存/読み込みが一致しません ({len(res.notes)} / {len(again.notes)})")
            labels = [c["label"] for c in again.chords if c["root"] is not None]
            if not labels:
                raise RuntimeError("コードが 1 つも検出されませんでした")
            return (f"{len(res.notes)} ノート, コード {' '.join(dict.fromkeys(labels))}, "
                    f"パート {', '.join(again.stems)}")
        finally:
            shutil.rmtree(res.folder, ignore_errors=True)


def _visual_parts() -> str:
    """ビジュアライザーのパート別表示 (ボーカル / フロント楽器 / カラオケ / 全パート) の準備と解析."""
    from .visual.dsp import VIEW_LABELS, FrameAnalyzer, build_track, build_views

    sr = config.IO_SR
    voice, harm = _synth(sr)
    stems = {"vocals": voice, "drums": np.zeros_like(harm), "bass": 0.5 * harm, "other": 0.5 * harm}
    t = np.arange(int(len(voice) / sr / config.HOP_SEC)) * config.HOP_SEC
    f0 = np.where((t > 0.5) & (t < 7.5), 440 * 2 ** ((40 * np.sin(2 * np.pi * 5.5 * t)) / 1200), np.nan)
    tr = build_track(stems, sr, "selftest", f0_t=t, f0=f0, notes=[(0.5, 7.5, 69.0)])
    views = build_views(tr)
    if list(views) != ["vocals", "front", "karaoke", "all"]:
        raise RuntimeError(f"表示パートが想定外です: {list(views)}")
    for key in ("vocals", "all"):
        fr = FrameAnalyzer(views[key]).analyze(4.0)
        if not (fr.voiced and abs(12 * np.log2(fr.f0 / 440.0)) < 1.0):
            raise RuntimeError(f"{VIEW_LABELS[key]} のピッチが想定外です ({fr.f0})")
    front = FrameAnalyzer(views["front"]).analyze(1.0)
    if front.bands.max() <= 0:
        raise RuntimeError("フロント楽器のスペクトラムが空です")
    return f"{' / '.join(VIEW_LABELS[k] for k in views)} ({len(tr.spec)} フレーム)"


def _gui_build() -> str:
    """統合版のウィンドウ (歌声アナライザー + ビジュアライザーのタブ) を作れるか."""
    from PySide6.QtCore import QCoreApplication, QThread

    app = QCoreApplication.instance()
    if app is None or QThread.currentThread() is not app.thread():
        return "省略 (アプリの中から実行したため)"     # ウィジェットは GUI のスレッドでしか作れない
    from .gui.main_window import MainWindow

    win = MainWindow()
    try:
        tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())] if win.tabs is not None else []
        if win.visualizer is None or len(tabs) != 2:
            raise RuntimeError(f"タブが想定外です: {tabs}")
        return " / ".join(tabs)
    finally:
        win.close()
        win.player.shutdown()
        win.deleteLater()


def _gpu() -> str:
    from .separation import cuda_usable, demucs_available, resolve_device

    if not demucs_available():
        return "Demucs 未インストール (簡易分離のみ使用可)"
    ok, why = cuda_usable()
    return f"Demucs 使用可 / デバイス: {resolve_device('auto')} ({why})"


def _demucs_run() -> str:
    """Demucs で実際に分離してみる (モデルが無ければダウンロードする)."""
    from .separation import demucs_available, separate

    if not demucs_available():
        return "Demucs 未インストールのため省略"
    sr = config.IO_SR
    voice, harm = _synth(sr)
    y = np.stack([voice + harm, voice + harm])[:, : sr * 4]
    msgs: list[str] = []
    stems = separate(y, sr, method="demucs_htdemucs", device="auto", progress=lambda m, f: msgs.append(m))
    if set(stems) < {"vocals", "drums", "bass", "other"}:
        raise RuntimeError(f"パートが足りません: {sorted(stems)}")
    dev = next((m for m in msgs if "音源分離中" in m), "")
    return f"htdemucs で 4 秒を分離: {', '.join(stems)}  {dev}"


def _audio_out() -> str:
    from PySide6.QtMultimedia import QMediaDevices

    outs = [d.description() for d in QMediaDevices.audioOutputs()]
    if not outs:
        return "デバイスが見つかりません (再生は無音になります)"
    return ", ".join(outs[:4])


def _gl() -> str:
    from .visual import widgets

    return f"{'GPU (OpenGL)' if widgets.USE_GL else 'ソフトウェア'} {widgets.GL_RENDERER}".strip()


def _network() -> str:
    from . import youtube

    res = youtube.search("NoCopyrightSounds", 5)
    if not res:
        raise RuntimeError("YouTube 検索の結果が 0 件でした")
    return f"YouTube 検索 OK ({res[0].title[:40]} …)"


def _youtube_streams() -> str:
    """動画の音声ストリームの URL を取得できるか (deno による JavaScript チャレンジの解決を含む).

    実際のダウンロードはしない。"""
    import yt_dlp

    from . import youtube

    res = youtube.search("NCS release", 10)
    if not res:
        raise RuntimeError("確認用の動画が見つかりませんでした")
    # 1 曲分の長さの動画を優先 (無ければ一番短いもの, 長さ不明なら先頭)
    songs = [r for r in res if r.duration and 60 <= r.duration <= 600]
    timed = sorted((r for r in res if r.duration), key=lambda r: r.duration)
    pick = (songs or timed or res)[0]
    opts = youtube._base_opts() | {"skip_download": True, "format": "bestaudio/best"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(pick.url, download=False)
    audio = [f for f in info.get("formats") or [] if f.get("acodec") not in (None, "none") and f.get("url")]
    if not audio:
        raise RuntimeError("音声ストリームの URL を取得できませんでした (yt-dlp の更新が必要かもしれません)")
    best = max(audio, key=lambda f: f.get("abr") or 0)
    return f"音声ストリーム {len(audio)} 件 (最良 {best.get('ext')} {best.get('abr') or '?'} kbps)"


def run(network: bool = False, demucs: bool = False, out=None) -> int:
    """全チェックを実行. 失敗数を返す."""
    out = out or sys.stdout
    from PySide6.QtWidgets import QApplication

    from .gui.common import setup_environment

    setup_environment()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    _ = app
    lines = [f"MUSICA_EX {__version__} 自己診断  {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"Python {platform.python_version()} / {platform.platform()} / frozen={config.FROZEN}",
             f"キャッシュ: {config.CACHE_DIR}"]
    checks: list[Check] = [
        ("ライブラリ", _versions), ("ffmpeg", _ffmpeg), ("deno", _deno), ("音声出力", _audio_out),
        ("描画", _gl), ("画面 (統合版)", _gui_build), ("GPU / Demucs", _gpu), ("ピッチ解析", _pitch),
        ("コード認識", _chords), ("音源分離 (簡易)", _separation), ("解析の全工程", _pipeline),
        ("ビジュアライザー (パート別)", _visual_parts),
    ]
    if demucs:
        checks.append(("音源分離 (Demucs)", _demucs_run))
    if network:
        checks += [("YouTube 検索", _network), ("YouTube 取得準備", _youtube_streams)]
    fails = 0
    for name, fn in checks:
        t0 = time.time()
        try:
            msg = fn()
            lines.append(f"[OK]   {name}: {msg}  ({time.time() - t0:.1f}s)")
        except Exception as exc:
            fails += 1
            lines.append(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            lines.append("       " + traceback.format_exc().strip().replace("\n", "\n       "))
        print(lines[-1], file=out, flush=True)
    lines.append("結果: すべて OK" if fails == 0 else f"結果: {fails} 件の失敗")
    print(lines[-1], file=out, flush=True)
    try:
        report = config.LOG_FILE.with_name("selftest.txt")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"レポート: {report}", file=out, flush=True)
    except OSError:
        pass
    return fails


if __name__ == "__main__":
    sys.exit(1 if run(network="--network" in sys.argv, demucs="--demucs" in sys.argv) else 0)


__all__ = ["run", "Path"]
