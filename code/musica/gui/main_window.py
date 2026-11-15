"""MUSICA のメインウィンドウ.

- MainWindow()                       統合版: 「歌声アナライザー」と「ビジュアライザー」をタブで切り替える。
                                     プレーヤーとミキサーは共有なので、タブを切り替えても再生は続き、
                                     解析した結果はビジュアライザーのタブでパート別に見られる。
- MainWindow(with_visualizer=False)  歌声アナライザー単体 (ビジュアライザーは別ウィンドウで開く)。
- ビジュアライザー単体 (解析せずにファイル / YouTube を可視化) は visual.window.VisualizerWindow。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import QMainWindow, QMessageBox, QTabWidget

from .. import config
from .analyzer_window import AnalyzerPanel
from .common import app_icon, confirm_quit_if_busy, start_worker
from .player import AudioPlayer
from .stems_view import MixerModel


class MainWindow(QMainWindow):
    TITLE_INTEGRATED = "MUSICA 歌声アナライザー + ビジュアライザー"
    TITLE_ANALYZER = "MUSICA 歌声調教アナライザー"

    def __init__(self, with_visualizer: bool = True) -> None:
        super().__init__()
        config.ensure_dirs()
        self.with_visualizer = with_visualizer
        self.base_title = self.TITLE_INTEGRATED if with_visualizer else self.TITLE_ANALYZER
        self.setWindowTitle(self.base_title)
        self.setWindowIcon(app_icon())
        self.resize(1680, 980)
        self.player = AudioPlayer(self)
        self.mixer = MixerModel(self)
        self._vis_windows: list = []
        self._was_max = False

        self.analyzer = AnalyzerPanel(self.player, self.mixer,
                                      vis_text="🎆 ビジュアライザーで見る" if with_visualizer else "🎆 ビジュアライザーで開く")
        self.analyzer.resultShown.connect(self._result_shown)
        self.analyzer.titleChanged.connect(self._set_song_title)
        self.analyzer.visualizerRequested.connect(self._show_visualizer)
        self.visualizer = None
        self.tabs: QTabWidget | None = None
        if with_visualizer:
            from ..visual.window import VisualizerPanel

            self.visualizer = VisualizerPanel(self.player, self.mixer, standalone=False)
            self.visualizer.fullScreenRequested.connect(self._set_fullscreen)
            self.tabs = QTabWidget()
            self.tabs.setDocumentMode(True)
            self.tabs.addTab(self.analyzer, "🎤 歌声アナライザー")
            self.tabs.addTab(self.visualizer, "🎆 ビジュアライザー")
            self.tabs.setTabToolTip(0, "音源を選んで解析 (調教の特徴・コード進行)  Ctrl+1")
            self.tabs.setTabToolTip(1, "解析した区間を倍音・スペクトラム・コードで可視化 (パート別に表示できます)  Ctrl+2")
            self.tabs.currentChanged.connect(self._tab_changed)
            self.setCentralWidget(self.tabs)
        else:
            self.setCentralWidget(self.analyzer)
        self._build_menu()
        self._set_actions_enabled(False)

    # ------------------------------------------------------------------ app.py から使う
    @property
    def source(self):
        return self.analyzer.source

    def open_result(self, folder: str | Path) -> None:
        self.analyzer.open_result(folder)

    def show_tab(self, name: str) -> None:
        """'analyzer' / 'visualizer'."""
        if self.tabs is None:
            return
        self.tabs.setCurrentWidget(self.visualizer if name == "visualizer" and self.visualizer else self.analyzer)

    # ------------------------------------------------------------------ メニュー
    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("ファイル")
        act = QAction("解析結果を開く...", self)
        act.setShortcut(QKeySequence.Open)
        act.triggered.connect(self.analyzer.open_result_dialog)
        m.addAction(act)
        m.addSeparator()
        self.act_csv = QAction("ノート特徴量を CSV で書き出し...", self)
        self.act_csv.triggered.connect(lambda: self.analyzer.export("notes"))
        self.act_ccsv = QAction("コード進行を CSV で書き出し...", self)
        self.act_ccsv.triggered.connect(lambda: self.analyzer.export("chords"))
        self.act_pcsv = QAction("ピッチ・コントロールカーブを CSV で書き出し...", self)
        self.act_pcsv.triggered.connect(lambda: self.analyzer.export("pitch"))
        self.act_folder = QAction("結果フォルダを開く (分離した WAV / JSON)", self)
        self.act_folder.triggered.connect(
            lambda: self.analyzer.res and QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.analyzer.res.folder))))
        for a in (self.act_csv, self.act_ccsv, self.act_pcsv, self.act_folder):
            m.addAction(a)
        m.addSeparator()
        q = QAction("終了", self)
        q.setShortcut(QKeySequence.Quit)
        q.triggered.connect(self.close)
        m.addAction(q)

        if self.tabs is not None:
            v = self.menuBar().addMenu("表示")
            for i, (text, key) in enumerate((("歌声アナライザー", "Ctrl+1"), ("ビジュアライザー", "Ctrl+2"))):
                a = QAction(text, self)
                a.setShortcut(QKeySequence(key))
                a.triggered.connect(lambda _=False, k=i: self.tabs.setCurrentIndex(k))
                v.addAction(a)

        t = self.menuBar().addMenu("ツール")
        vis = QAction("ビジュアライザー単体を開く (解析せずにファイル / YouTube を可視化)", self)
        vis.triggered.connect(self._new_visualizer_window)
        t.addAction(vis)

        h = self.menuBar().addMenu("ヘルプ")
        chk = QAction("動作チェック...", self)
        chk.triggered.connect(self._run_selftest)
        data = QAction("データ / ログのフォルダを開く", self)
        data.triggered.connect(lambda: (config.ensure_dirs(),
                                        QDesktopServices.openUrl(QUrl.fromLocalFile(str(config.CACHE_DIR)))))
        about = QAction("バージョン情報", self)
        about.triggered.connect(self._about)
        for a in (chk, data, about):
            h.addAction(a)

    def _set_actions_enabled(self, on: bool) -> None:
        for a in (self.act_csv, self.act_ccsv, self.act_pcsv, self.act_folder):
            a.setEnabled(on)

    def _run_selftest(self) -> None:
        """ヘルプ → 動作チェック: 結果をダイアログに表示 (YouTube 接続も確認)."""
        import io

        from .. import selftest

        def job():
            buf = io.StringIO()
            fails = selftest.run(network=True, out=buf)
            return fails, buf.getvalue()

        def done(res):
            fails, text = res
            box = QMessageBox(self)
            box.setWindowTitle("動作チェック")
            box.setIcon(QMessageBox.Information if fails == 0 else QMessageBox.Warning)
            box.setText("すべて OK です。" if fails == 0 else f"{fails} 件の項目で問題が見つかりました。")
            box.setDetailedText(text)
            box.exec()

        status = self.analyzer.show_status
        status("動作チェック中 (30 秒ほどかかります)...", -1)
        start_worker(job, use_progress=False, on_done=lambda r: (status("動作チェック完了", 1.0), done(r)),
                     on_error=lambda msg: status(f"動作チェックでエラー: {msg}", 0.0))

    def _about(self) -> None:
        from .. import __version__
        from ..separation import demucs_available

        QMessageBox.about(self, "MUSICA_EX", f"MUSICA_EX {__version__}\n\n"
                          f"音源分離: {'Demucs + 簡易分離' if demucs_available() else '簡易分離のみ'}\n"
                          f"データの保存先: {config.CACHE_DIR}\nログ: {config.LOG_FILE}")

    # ------------------------------------------------------------------ 解析結果
    def _result_shown(self, res) -> None:
        self._set_actions_enabled(True)
        if self.visualizer is not None:
            self.visualizer.show_result(res)          # 準備はビジュアライザーのタブを開いたときに行う

    def _set_song_title(self, text: str) -> None:
        self.setWindowTitle(f"{self.base_title} — {text}" if text else self.base_title)

    # ------------------------------------------------------------------ ビジュアライザー
    def _show_visualizer(self) -> None:
        """「ビジュアライザーで見る」: 統合版はタブを切り替え、単体版は別ウィンドウで開く."""
        res = self.analyzer.res
        if res is None:
            return
        start = self.analyzer.current_rel()
        if self.visualizer is not None:
            self.visualizer.show_result(res, start=start)
            self.tabs.setCurrentWidget(self.visualizer)
        else:
            w = self._new_visualizer_window()
            w.load_result(res.folder, gains=self.mixer.gains(), start=start)

    def _new_visualizer_window(self):
        from ..visual.window import VisualizerWindow

        self.player.pause()                        # 別ウィンドウは別のプレーヤーなので二重に鳴らさない
        w = VisualizerWindow()
        w.setAttribute(Qt.WA_DeleteOnClose)
        self._vis_windows.append(w)
        w.destroyed.connect(lambda *_: self._vis_windows.remove(w) if w in self._vis_windows else None)
        w.show()
        return w

    def _tab_changed(self, _idx: int) -> None:
        if self.visualizer is not None and self.tabs.currentWidget() is not self.visualizer \
                and self.visualizer.is_fullscreen:
            self.visualizer.set_fullscreen(False)

    def _set_fullscreen(self, on: bool) -> None:
        """ビジュアライザーの全画面: メニューとタブも隠す (Esc / F11 で戻る)."""
        if on:
            self._was_max = self.isMaximized()
            self.menuBar().hide()
            if self.tabs is not None:
                self.tabs.tabBar().hide()
            self.showFullScreen()
        else:
            self.menuBar().show()
            if self.tabs is not None:
                self.tabs.tabBar().show()
            if self._was_max:
                self.showMaximized()
            else:
                self.showNormal()

    def changeEvent(self, ev) -> None:  # noqa: N802
        super().changeEvent(ev)
        # ウィンドウマネージャー側の操作で全画面が解除されたら、隠したものを戻す
        if (ev.type() == QEvent.WindowStateChange and self.visualizer is not None and not self.isFullScreen()
                and self.visualizer.is_fullscreen):
            self.visualizer.set_fullscreen(False, notify=False)
            self.menuBar().show()
            if self.tabs is not None:
                self.tabs.tabBar().show()

    def closeEvent(self, ev) -> None:  # noqa: N802
        if not confirm_quit_if_busy(self):
            ev.ignore()
            return
        for w in list(self._vis_windows):
            w.close()
        if self.visualizer is not None:
            self.visualizer.shutdown()
        self.player.shutdown()
        super().closeEvent(ev)
