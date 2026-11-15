"""歌声調教アナライザーの画面 (音源の選択・解析設定・解析結果の表示).

ウィンドウ (メニュー・タブ・閉じる処理) は main_window.MainWindow が持つ。
統合版ではビジュアライザーのタブとプレーヤー / ミキサーを共有する。
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSplitter,
                               QTabWidget, QVBoxLayout, QWidget)

from .. import config
from ..analysis import AnalysisResult, run_analysis
from ..separation import available_methods
from ..youtube import VideoInfo, cached_audio
from .common import fmt_time, start_worker
from .note_detail import NoteDetailPanel
from .pianoroll import PianoRollView
from .player import AudioPlayer
from .source_panel import SourcePanel
from .stems_view import MixerModel, StemsView
from .summary_view import NotesTable, SpectrogramView, SummaryView


class AnalyzerPanel(QWidget):
    """左: 音源と解析設定 / 右: 解析結果 (音程バー・パート・ノート一覧・傾向・スペクトログラム)."""

    resultShown = Signal(object)          # AnalysisResult (解析が終わった / 結果を開いた)
    titleChanged = Signal(str)            # ウィンドウのタイトルに出す曲の情報
    visualizerRequested = Signal()        # 「ビジュアライザーで見る / 開く」ボタン

    def __init__(self, player: AudioPlayer, mixer: MixerModel, vis_text: str = "🎆 ビジュアライザーで開く",
                 parent=None) -> None:
        super().__init__(parent)
        config.ensure_dirs()
        self.player = player
        self.mixer = mixer
        self.res: AnalysisResult | None = None
        self.sel_note = -1
        self._analyzing = False
        self._temp_gains = False

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)

        # ---------------------------------------------------------------- 左: 入力と設定
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 4, 8)
        src_box = QGroupBox("1. 音源を選ぶ / 区間を指定")
        sb = QVBoxLayout(src_box)
        self.source = SourcePanel(self.player)
        sb.addWidget(self.source)
        ll.addWidget(src_box)

        opt_box = QGroupBox("2. 解析設定")
        form = QFormLayout(opt_box)
        self.method = QComboBox()
        for k, v in available_methods().items():
            self.method.addItem(v, k)
        self.method.setToolTip("ボーカルと楽器 (ドラム・ベース・ギター・ピアノ等) に分離します。\n"
                               "コード認識は分離した伴奏パートから行います。")
        self.voice = QComboBox()
        for k, (label, lo, hi) in config.VOICE_PRESETS.items():
            self.voice.addItem(label, (lo, hi))
        self.device = QComboBox()
        devs = [("自動 (GPU があれば使用)", "auto"), ("GPU (NVIDIA CUDA)", "cuda")]
        if sys.platform == "darwin":
            devs.append(("GPU (Apple Silicon)", "mps"))
        devs.append(("CPU", "cpu"))
        for label, key in devs:
            self.device.addItem(label, key)
        form.addRow("音源分離", self.method)
        form.addRow("声域 (f0 探索範囲)", self.voice)
        form.addRow("デバイス", self.device)
        self.run_btn = QPushButton("解析開始")
        self.run_btn.setObjectName("primary")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._run)
        form.addRow(self.run_btn)
        ll.addWidget(opt_box)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.status_label = QLabel("YouTube の URL を入力するか、検索してください")
        self.status_label.setWordWrap(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setStyleSheet("font-size:9pt; color:#9aa4b8;")
        ll.addWidget(self.status_label)
        ll.addWidget(self.progress)
        ll.addWidget(self.log, 1)
        split.addWidget(left)

        # ---------------------------------------------------------------- 右: 結果
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 8, 8, 8)
        bar = QHBoxLayout()
        self.play_btn = QPushButton("▶ 再生")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setToolTip("再生 / 一時停止 (Space)")
        self.preset_btns = []
        for key, text in (("all", "全パート"), ("vocals", "ボーカルのみ"), ("karaoke", "カラオケ")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, k=key: self.mixer.preset(k))
            self.preset_btns.append(b)
        self.time_label = QLabel("0:00.0")
        self.time_label.setStyleSheet("font-family:monospace; font-size:13pt; color:#feca57; padding:0 8px;")
        self.follow = QCheckBox("追従")
        self.follow.setToolTip("再生位置に合わせて音程バーをスクロール")
        self.follow.setChecked(True)
        self.chord_mode = QComboBox()
        self.chord_mode.addItem("コード名 (Cm7)", "name")
        self.chord_mode.addItem("度数 (VIm7)", "degree")
        self.chord_mode.currentIndexChanged.connect(lambda _i: self.roll.set_chord_mode(self.chord_mode.currentData()))
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(int(round(self.player.volume() * 100)))
        self.vol.setMinimumWidth(70)
        self.vol.setMaximumWidth(110)
        self.vol.valueChanged.connect(lambda v: self.player.set_volume(v / 100))
        self.vis_btn = QPushButton(vis_text)
        self.vis_btn.setToolTip("この解析結果をビジュアライザー (倍音・スペクトラム・コード) で見る")
        self.vis_btn.clicked.connect(lambda _=False: self.visualizerRequested.emit())
        for w in (self.play_btn, *self.preset_btns, self.time_label, self.follow, QLabel("コード表示"), self.chord_mode):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(QLabel("音量"))
        bar.addWidget(self.vol)
        bar.addWidget(self.vis_btn)
        rl.addLayout(bar)

        legend = QLabel(
            "<span style='color:#ff9f43'>■</span> しゃくり　<span style='color:#a29bfe'>■</span> 上から入る　"
            "<span style='color:#48dbfb'>■</span> ポルタメント　<span style='color:#1dd1a1'>■</span> ストレート　"
            "｜ バー左の濃い部分 = 目標音に落ち着くまで　｜ 下端の帯: "
            "<span style='color:#ff6b6b'>■</span> フォール <span style='color:#feca57'>■</span> 語尾上げ "
            "<span style='color:#f368e0'>■</span> 予備動作　｜ 〰 = ビブラート　｜ ▲▼ = 上ずり/こぶし/落ち込み"
            "　｜ 縦線 = 拍")
        legend.setStyleSheet("color:#9aa4b8; font-size:9pt;")
        legend.setWordWrap(True)
        rl.addWidget(legend)

        self.tabs = QTabWidget()
        rl.addWidget(self.tabs, 1)
        roll_split = QSplitter(Qt.Horizontal)
        self.roll = PianoRollView()
        self.detail = NoteDetailPanel()
        roll_split.addWidget(self.roll)
        roll_split.addWidget(self.detail)
        roll_split.setSizes([980, 560])
        self.tabs.addTab(roll_split, "音程バー + コード + ノート詳細")
        self.stems_view = StemsView(self.mixer)
        self.tabs.addTab(self.stems_view, "パート (楽器分離)")
        self.table = NotesTable()
        self.tabs.addTab(self.table, "ノート一覧")
        self.summary = SummaryView()
        self.tabs.addTab(self.summary, "歌い方の傾向")
        self.spec = SpectrogramView()
        self.tabs.addTab(self.spec, "スペクトログラム")
        self.tabs.currentChanged.connect(self._tab_changed)
        split.addWidget(right)
        split.setSizes([440, 1240])
        split.setStretchFactor(1, 1)

        # ---------------------------------------------------------------- 接続
        self.source.status.connect(self.show_status)
        self.source.audioReady.connect(lambda *_: self.run_btn.setEnabled(True))
        self.roll.noteClicked.connect(self._select_note)
        self.roll.seekRequested.connect(self._seek_abs)
        self.spec.seekRequested.connect(self._seek_abs)
        self.stems_view.seekRequested.connect(self._seek_abs)
        self.table.noteSelected.connect(self._select_note_and_show)
        self.detail.playRequested.connect(self._play_range)
        self.detail.stepRequested.connect(lambda d: self._select_note(self.sel_note + d))
        self.player.positionChanged.connect(self._on_pos)
        self.player.playingChanged.connect(self._on_playing)
        self.mixer.changed.connect(self._apply_mixer)
        # ショートカットはこの画面が見えている時だけ有効 (統合版でビジュアライザーのタブと衝突しない)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._toggle_play)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self._select_note(self.sel_note - 1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self._select_note(self.sel_note + 1))
        self._set_result_enabled(False)

    def showEvent(self, ev) -> None:  # noqa: N802
        super().showEvent(ev)
        self.vol.blockSignals(True)
        self.vol.setValue(int(round(self.player.volume() * 100)))     # 統合版ではビジュアライザーと共有
        self.vol.blockSignals(False)
        self._on_pos(self.player.position())        # 隠れている間に進んだ再生位置に合わせる

    @property
    def analyzing(self) -> bool:
        return self._analyzing

    def _set_result_enabled(self, on: bool) -> None:
        for w in (self.play_btn, *self.preset_btns, self.vis_btn):
            w.setEnabled(on)

    # ------------------------------------------------------------------ 進捗
    def show_status(self, msg: str, frac: float) -> None:
        self.status_label.setText(msg)
        if frac < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(frac * 1000))
        last = self.log.toPlainText().rsplit("\n", 1)[-1] if self.log.blockCount() else ""
        if not last.endswith(msg):
            self.log.appendPlainText(msg)

    # ------------------------------------------------------------------ 解析
    def _run(self) -> None:
        if self._analyzing:
            return
        if self.source.wav is None:
            QMessageBox.information(self, "音源がありません", "先に YouTube の URL / 検索 / ファイルから音源を取得してください。")
            return
        s, e = self.source.selection()
        if e - s < 0.5:
            QMessageBox.warning(self, "区間エラー", "0.5 秒以上の区間を指定してください。")
            return
        self.player.stop()
        lo, hi = self.voice.currentData()
        self._analyzing = True
        self.run_btn.setEnabled(False)
        self.run_btn.setText("解析中...")
        from dataclasses import asdict

        start_worker(run_analysis, self.source.wav, asdict(self.source.info), s, e,
                     method=self.method.currentData(), device=self.device.currentData(), fmin=lo, fmax=hi,
                     on_done=self._analysis_done, on_error=self._analysis_failed, on_progress=self.show_status)

    def _analysis_done(self, res: AnalysisResult) -> None:
        self._analyzing = False
        self.run_btn.setEnabled(True)
        self.run_btn.setText("解析開始")
        self.show_result(res)
        self.show_status(f"解析完了: {len(res.notes)} ノート / {len(res.chords)} コード "
                         f"({res.meta['elapsed_s']:.1f} 秒)  保存先: {res.folder}", 1.0)

    def _analysis_failed(self, msg: str) -> None:
        self._analyzing = False
        self.run_btn.setEnabled(True)
        self.run_btn.setText("解析開始")
        self.show_status(f"エラー: {msg}", 0.0)
        QMessageBox.critical(self, "解析エラー", msg)

    def _result_source(self) -> str | None:
        return f"result:{self.res.folder}" if self.res else None

    def _load_result_audio(self) -> None:
        """分離した全パートをプレーヤーに読み込む (ミキサーで鳴らし分けられる)."""
        if not self.res:
            return
        self.player.load_stems(self.res.playback_parts(), self._result_source(), self.mixer.gains())

    def show_result(self, res: AnalysisResult) -> None:
        self.res = res
        self.roll.set_result(res)
        self.stems_view.set_result(res, link_to=None)
        self.table.set_result(res)
        self.summary.set_result(res)
        self.detail.clear()
        self.sel_note = -1
        self._set_result_enabled(True)
        for b, key in zip(self.preset_btns, ("all", "vocals", "karaoke")):
            b.setEnabled(self.mixer.preset_available(key))
        if res.notes:
            self._select_note(0, center=False)
        if self.tabs.currentWidget() is self.spec:
            self.spec.set_result(res)
        self._load_result_audio()
        key = res.harmony.get("key", {}) if res.harmony else {}
        title = res.meta.get("title", "")
        if key:
            title += f"  [キー {key.get('name', '-')}, {res.harmony.get('tempo', 0):.0f} BPM]"
        self.titleChanged.emit(title)
        self.resultShown.emit(res)

    def open_result(self, folder: str | Path) -> None:
        res = AnalysisResult.load(folder)
        self.show_result(res)
        vid = res.meta.get("video_id")
        hit = cached_audio(vid) if vid else None
        if hit:
            self.source.load_cached(hit[0], hit[1], res.meta["start"], res.meta["end"])
        elif Path(res.meta.get("source_wav", "")).exists():
            info = VideoInfo(video_id=vid or "", title=res.meta.get("title", ""), channel=res.meta.get("channel", ""))
            self.source.load_cached(Path(res.meta["source_wav"]), info, res.meta["start"], res.meta["end"])
        self._load_result_audio()
        self.show_status(f"解析結果を読み込みました: {res.folder.name}", 1.0)

    def open_result_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "解析結果フォルダを選択", str(config.RESULT_DIR))
        if d:
            try:
                self.open_result(d)
            except Exception as exc:
                QMessageBox.critical(self, "読み込みエラー", f"{type(exc).__name__}: {exc}")

    def export(self, kind: str) -> None:
        if not self.res:
            return
        name = f"{self.res.folder.name}_{kind}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "CSV で保存", str(Path.home() / name), "CSV (*.csv)")
        if not path:
            return
        {"notes": self.res.export_notes_csv, "chords": self.res.export_chords_csv,
         "pitch": self.res.export_pitch_csv}[kind](path)
        self.show_status(f"書き出しました: {path}", 1.0)

    # ------------------------------------------------------------------ ノート選択
    def _select_note(self, i: int, center: bool = True) -> None:
        if not self.res or not (0 <= i < len(self.res.notes)):
            return
        self.sel_note = i
        self.roll.select_note(i, center=center)
        self.detail.set_note(self.res, i)
        self.table.select_note(i)

    def _select_note_and_show(self, i: int) -> None:
        self._select_note(i)
        self.tabs.setCurrentIndex(0)

    def _tab_changed(self, idx: int) -> None:
        if self.tabs.widget(idx) is self.spec and self.res is not None:
            self.spec.set_result(self.res)

    # ------------------------------------------------------------------ 再生
    def _is_result_loaded(self) -> bool:
        return self.res is not None and self.player.path == self._result_source()

    def _apply_mixer(self) -> None:
        if self._is_result_loaded() and not self._temp_gains:
            self.player.set_gains(self.mixer.gains())

    def _toggle_play(self) -> None:
        if self.player.is_playing():
            self.player.pause()
            return
        if self.res is None:
            return
        if not self._is_result_loaded():
            self._load_result_audio()
            rel = self._current_abs() - self.res.offset_s
            self.player.seek(max(0.0, rel))
        self._restore_gains()
        self.player.play()

    def _restore_gains(self) -> None:
        if self._temp_gains:
            self._temp_gains = False
            self.player.set_gains(self.mixer.gains())

    def _play_range(self, a: float, b: float, which: str) -> None:
        """ノート詳細の ♪ ボタン: 'vocal' は一時的にボーカルだけ, 'mix' は現在のミキサー設定で."""
        if not self.res:
            return
        if not self._is_result_loaded():
            self._load_result_audio()
        if which == "vocal" and "vocals" in self.player.stem_names():
            self.player.set_gains({s: (1.0 if s == "vocals" else 0.0) for s in self.player.stem_names()})
            self._temp_gains = True
        else:
            self._restore_gains()
        off = self.res.offset_s
        self.player.play(max(0.0, a - off), b - off)

    def _on_playing(self, on: bool) -> None:
        self.play_btn.setText("⏸ 一時停止" if on else "▶ 再生")
        if not on:
            self._restore_gains()

    def _current_abs(self) -> float:
        return float(self.roll.cursor.value())

    def current_rel(self) -> float:
        """音程バーのカーソル位置 (解析区間の先頭からの秒数)."""
        if not self.res:
            return 0.0
        return max(0.0, self._current_abs() - self.res.offset_s)

    def seek_rel(self, rel: float) -> None:
        """解析区間の先頭から rel 秒の位置へ移動する."""
        if self.res:
            self._seek_abs(self.res.offset_s + rel)

    def _seek_abs(self, t_abs: float) -> None:
        if not self.res:
            return
        rel = t_abs - self.res.offset_s
        if not self._is_result_loaded():
            self._load_result_audio()
        self.player.seek(max(0.0, rel))
        self._set_cursors(t_abs, follow=False)

    def _set_cursors(self, t_abs: float, follow: bool) -> None:
        self.roll.set_cursor(t_abs, follow=follow)
        self.spec.set_cursor(t_abs)
        self.stems_view.set_cursor(t_abs)

    def _on_pos(self, pos: float) -> None:
        if not self.isVisible():
            return          # 統合版でビジュアライザーのタブを見ている間は描き直さない (表示時に合わせる)
        if self._is_result_loaded():
            t_abs = pos + self.res.offset_s
            self._set_cursors(t_abs, follow=self.follow.isChecked() and self.player.is_playing())
        else:
            t_abs = pos
        self.time_label.setText(fmt_time(t_abs, 1))
