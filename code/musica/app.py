"""アプリの起動処理 (統合版 / 歌声アナライザー単体 / ビジュアライザー単体 共通).

    python -m musica.app                       # 統合版: 歌声アナライザーとビジュアライザーをタブで切り替え
    python -m musica.app --tab visualizer      #   最初からビジュアライザーのタブを開く
    python -m musica.app --analyzer            # 歌声アナライザー単体
    python -m musica.app --visualizer          # ビジュアライザー単体 (解析せずにファイル / YouTube を可視化)
    python -m musica.app --selftest            # 動作チェック

Windows のショートカット・PyInstaller 版の実行ファイル・pip の gui-scripts は全てここを呼ぶ。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MODES = ("integrated", "analyzer", "visualizer")
# 起動モード → (Windows の AppUserModelID, アプリ名, Linux の .desktop ファイル名)
APP_INFO = {
    "integrated": ("MUSICA.EX", "MUSICA", "musica-ex"),
    "analyzer": ("MUSICA.EX.Analyzer", "MUSICA Vocal Analyzer", "musica-analyzer"),
    "visualizer": ("MUSICA.EX.Visualizer", "MUSICA Visualizer", "musica-visualizer"),
}


def _ensure_stdio() -> None:
    """コンソールの無い GUI 起動 (pythonw.exe / 実行ファイル版) では sys.stdout / stderr が None になり、
    print や進捗表示 (tqdm 等) で落ちるので、ログファイルに向ける."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    from . import config

    try:
        config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log = open(config.LOG_FILE, "a", encoding="utf-8", buffering=1, errors="replace")  # noqa: SIM115
    except OSError:
        log = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log


def _prepare_runtime() -> None:
    """ライブラリを読み込む前に必要な環境変数."""
    from . import config

    # librosa が使う numba の JIT キャッシュの置き場所 (実行ファイル版や書き込めない場所でも動くように)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(config.CACHE_DIR / "numba"))
    if config.FROZEN:
        # 実行ファイル版: Demucs のモデル (torch hub) をユーザーデータ側に保存
        os.environ.setdefault("TORCH_HOME", str(config.user_data_dir() / "torch"))


def _mode_from_executable() -> str:
    """実行ファイル名で起動モードを決める (PyInstaller 版は同じ exe を別名で置いているため).

    'visual' を含む → ビジュアライザー単体 / 'analyzer' を含む → アナライザー単体 / それ以外 → 統合版。
    """
    stem = Path(sys.executable if getattr(sys, "frozen", False) else sys.argv[0]).stem.lower()
    if "visual" in stem:
        return "visualizer"
    if "analyzer" in stem or "analyser" in stem:
        return "analyzer"
    return "integrated"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="musica", description="MUSICA 歌声アナライザー + オーディオビジュアライザー")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--analyzer", action="store_true", help="歌声アナライザー単体で起動 (ビジュアライザーのタブなし)")
    mode.add_argument("--visualizer", action="store_true",
                      help="ビジュアライザー単体で起動 (解析せずにファイル / YouTube を可視化)")
    ap.add_argument("--tab", choices=["analyzer", "visualizer"], help="統合版で最初に開くタブ")
    ap.add_argument("--url", help="起動時に取得する YouTube URL")
    ap.add_argument("--file", help="音声 / 動画ファイル (ビジュアライザー単体)")
    ap.add_argument("--start", type=float, help="開始位置 (秒)")
    ap.add_argument("--end", type=float, help="解析終了 (秒, アナライザー)")
    ap.add_argument("--result", help="解析結果フォルダを開く")
    ap.add_argument("--source", choices=["mix", "vocal"], default="mix",
                    help="解析結果のどちらを鳴らすか (ビジュアライザー単体)")
    ap.add_argument("--selftest", action="store_true", help="動作チェックをして終了")
    ap.add_argument("--network", action="store_true", help="--selftest で YouTube への接続も確認")
    ap.add_argument("--demucs", action="store_true", help="--selftest で Demucs の分離も実行 (モデルを取得)")
    return ap


def main(argv: list[str] | None = None, visualizer: bool | None = None, mode: str | None = None,
         default_mode: str | None = None) -> int:
    """起動. モードの優先順: mode 引数 > --analyzer / --visualizer > default_mode > 実行ファイル名.

    visualizer=True / False は旧版との互換 (ビジュアライザー単体 / アナライザー単体)。
    """
    _ensure_stdio()
    import multiprocessing

    multiprocessing.freeze_support()            # PyInstaller 版で子プロセスが起動されても安全に
    _prepare_runtime()

    args = build_parser().parse_args(argv)
    if mode is None and visualizer is not None:
        mode = "visualizer" if visualizer else "analyzer"
    if mode is None:
        if args.visualizer:
            mode = "visualizer"
        elif args.analyzer:
            mode = "analyzer"
        else:
            mode = default_mode or _mode_from_executable()
    if mode not in MODES:
        raise ValueError(f"未知の起動モード: {mode}")

    if args.selftest:
        from . import selftest

        return 1 if selftest.run(network=args.network, demucs=args.demucs) else 0

    from .gui.common import setup_environment

    app_id, app_name, desktop = APP_INFO[mode]
    setup_environment(app_id)
    from PySide6.QtWidgets import QApplication

    from .gui.common import app_icon, apply_theme, wait_all_workers

    app = QApplication(sys.argv[:1])
    app.setApplicationName(app_name)
    app.setOrganizationName("MUSICA_EX")
    # Linux: .desktop ファイル (セットアップで作成) とウィンドウを結び付けてアイコンを出す
    app.setDesktopFileName(desktop)
    app.setWindowIcon(app_icon())
    apply_theme(app)

    if mode == "visualizer":
        _start_visualizer(args)
    else:
        _start_main(args, with_visualizer=mode == "integrated")
    _maybe_screenshot(app)
    code = app.exec()
    wait_all_workers(2000)
    return code


def _start_main(args, with_visualizer: bool) -> None:
    """統合版 (with_visualizer=True) / 歌声アナライザー単体."""
    from .gui.main_window import MainWindow

    win = MainWindow(with_visualizer=with_visualizer)
    win.show()
    _keep.append(win)
    if args.result:
        win.open_result(args.result)
        if args.start is not None:
            win.analyzer.seek_rel(args.start)                     # 解析区間の先頭からの秒数
    elif args.url:
        win.source.url_edit.setText(args.url)
        if args.start is not None:
            def _apply_range(*_):
                win.source.set_selection(args.start, args.end if args.end is not None else args.start + 30)
                win.source.audioReady.disconnect(_apply_range)

            win.source.audioReady.connect(_apply_range)
        win.source._fetch_url()
    if args.tab:
        win.show_tab(args.tab)


def _start_visualizer(args) -> None:
    """ビジュアライザー単体."""
    from .visual.window import VisualizerWindow

    win = VisualizerWindow()
    win.show()
    _keep.append(win)
    if args.result:
        win.load_result(args.result, args.source, start=args.start)
    elif args.file:
        win.load_file(args.file, start=args.start)
    elif args.url:
        win.load_url(args.url, start=args.start)


_keep: list = []


def _maybe_screenshot(app) -> None:
    """自動テスト用: 環境変数 MUSICA_SCREENSHOT=<png> があれば、起動して数秒後に画面を保存して終了する
    (MUSICA_SCREENSHOT_DELAY ミリ秒, 既定 6000)."""
    out = os.environ.get("MUSICA_SCREENSHOT")
    if not out:
        return
    from PySide6.QtCore import QTimer

    def shot() -> None:
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QPainter

        wins = [w for w in app.topLevelWidgets() if w.isVisible() and w.isWindow()]
        if wins:
            win = wins[-1]
            img = win.grab().toImage()
            # QOpenGLWidget はウィンドウ全体の grab() では正しく写らないことがあるので、
            # 各パネルのフレームバッファを個別に取得して上から貼る (画面に出ている内容と同じ)
            try:
                from PySide6.QtOpenGLWidgets import QOpenGLWidget

                p = QPainter(img)
                for gl in win.findChildren(QOpenGLWidget):
                    if gl.isVisible():
                        fb = gl.grabFramebuffer()
                        fb.setDevicePixelRatio(img.devicePixelRatio())
                        p.drawImage(gl.mapTo(win, QPoint(0, 0)), fb)
                p.end()
            except ImportError:
                pass
            img.save(out)
        print(f"screenshot: {out} ({', '.join(w.windowTitle() for w in wins)})", flush=True)
        app.closeAllWindows()
        app.quit()

    QTimer.singleShot(int(os.environ.get("MUSICA_SCREENSHOT_DELAY", "6000")), shot)


def integrated_main() -> int:
    """pip / uv の gui-scripts 用 (musica-ex.exe: 統合版)."""
    return main(default_mode="integrated")


def analyzer_main() -> int:
    """pip / uv の gui-scripts 用 (musica-analyzer.exe: 歌声アナライザー単体)."""
    return main(default_mode="analyzer")


def visualizer_main() -> int:
    """pip / uv の gui-scripts 用 (musica-visualizer.exe: ビジュアライザー単体)."""
    return main(default_mode="visualizer")


def selftest_main() -> int:
    """pip / uv の scripts 用 (musica-selftest, コンソール)."""
    return main(["--selftest", *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
