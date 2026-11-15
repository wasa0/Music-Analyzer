"""音源の選択 (YouTube URL / 検索 / ローカルファイル) と解析区間の指定."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .. import youtube
from ..youtube import VideoInfo
from .common import TimeSpinBox, fmt_time, start_worker
from .player import AudioPlayer


class TimeAxis(pg.AxisItem):
    """秒を m:ss 表記にする軸."""

    def tickStrings(self, values, scale, spacing):  # noqa: N802
        digits = 1 if spacing < 1 else 0
        return [fmt_time(v, digits) for v in values]


def compute_overview(wav: Path, n_bins: int = 3000) -> dict:
    """波形の概形 (区間ごとの最小/最大). 長い動画でもメモリを食わないよう少しずつ読む."""
    import soundfile as sf

    info = sf.info(str(wav))
    sr, frames = info.samplerate, info.frames
    size = max(1, frames // n_bins)
    k = max(1, frames // size)
    mn = np.zeros(k, dtype=np.float32)
    mx = np.zeros(k, dtype=np.float32)
    i = 0
    block = size * max(1, (sr * 30) // size)            # 約 30 秒ずつ (size の倍数)
    for blk in sf.blocks(str(wav), blocksize=block, dtype="float32", always_2d=True):
        m = blk.mean(axis=1)
        n = len(m) // size
        if n == 0 or i >= k:
            break
        n = min(n, k - i)
        r = m[: n * size].reshape(n, size)
        mn[i:i + n] = r.min(axis=1)
        mx[i:i + n] = r.max(axis=1)
        i += n
    return {"t": (np.arange(i) + 0.5) * size / sr, "mn": mn[:i], "mx": mx[:i], "duration": frames / float(sr)}


class SourcePanel(QWidget):
    audioReady = Signal(object, object)       # (wav Path, VideoInfo)
    status = Signal(str, float)               # 進捗表示用

    def __init__(self, player: AudioPlayer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.player = player
        self.wav: Path | None = None
        self.info: VideoInfo | None = None
        self.duration = 0.0
        self._pending_start: float | None = None
        self._busy = False
        self._results: list[VideoInfo] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        # ---- 入力タブ
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)

        url_w = QWidget()
        ul = QVBoxLayout(url_w)
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://www.youtube.com/watch?v=...  (t=90 で開始位置も反映)")
        self.url_edit.returnPressed.connect(self._fetch_url)
        self.url_btn = QPushButton("音源を取得")
        self.url_btn.setObjectName("primary")
        self.url_btn.clicked.connect(self._fetch_url)
        row = QHBoxLayout()
        row.addWidget(self.url_edit, 1)
        row.addWidget(self.url_btn)
        ul.addLayout(row)
        ul.addWidget(QLabel("YouTube の URL を貼り付けて取得します。取得した音源は cache/ に保存され再利用されます。"))
        ul.addStretch(1)
        self.tabs.addTab(url_w, "URL")

        s_w = QWidget()
        sl = QVBoxLayout(s_w)
        row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("曲名・歌手名など (例: 曲名 歌ってみた / acapella)")
        self.search_edit.returnPressed.connect(self._search)
        self.search_btn = QPushButton("検索")
        self.search_btn.clicked.connect(self._search)
        row.addWidget(self.search_edit, 1)
        row.addWidget(self.search_btn)
        sl.addLayout(row)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["タイトル", "チャンネル", "長さ", "再生数"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._fetch_result(r))
        sl.addWidget(self.table, 1)
        self.pick_btn = QPushButton("選択した動画の音源を取得")
        self.pick_btn.setObjectName("primary")
        self.pick_btn.clicked.connect(lambda: self._fetch_result(self.table.currentRow()))
        sl.addWidget(self.pick_btn)
        self.tabs.addTab(s_w, "検索")

        f_w = QWidget()
        fl = QVBoxLayout(f_w)
        self.file_btn = QPushButton("音声 / 動画ファイルを開く...")
        self.file_btn.clicked.connect(self._open_file)
        fl.addWidget(self.file_btn)
        fl.addWidget(QLabel("自分の録音や手元の音源 (wav, mp3, m4a, flac, mp4 など) も解析できます。"))
        fl.addStretch(1)
        self.tabs.addTab(f_w, "ファイル")
        self.tabs.setMaximumHeight(300)

        # ---- 取得した音源の情報と区間指定
        self.info_label = QLabel("音源が未選択です")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color:#e8ecf4; font-weight:bold;")
        lay.addWidget(self.info_label)

        self.overview = pg.PlotWidget(axisItems={"bottom": TimeAxis("bottom")})
        self.overview.setMinimumHeight(110)
        self.overview.setMaximumHeight(150)
        self.overview.hideAxis("left")
        self.overview.setMouseEnabled(x=True, y=False)
        self.overview.setMenuEnabled(False)
        self.wave_item = pg.PlotDataItem(pen=pg.mkPen("#4b7bec", width=1))
        self.overview.addItem(self.wave_item)
        self.region = pg.LinearRegionItem(values=(0, 30), brush=pg.mkBrush(61, 126, 255, 50),
                                          pen=pg.mkPen("#3d7eff", width=2))
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self._region_moved)
        self.overview.addItem(self.region)
        self.playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen("#feca57", width=1.5))
        self.playhead.setVisible(False)
        self.overview.addItem(self.playhead)
        lay.addWidget(self.overview)

        row = QHBoxLayout()
        row.addWidget(QLabel("開始"))
        self.start_box = TimeSpinBox()
        row.addWidget(self.start_box)
        row.addWidget(QLabel("終了"))
        self.end_box = TimeSpinBox()
        self.end_box.setValue(30.0)
        row.addWidget(self.end_box)
        self.len_label = QLabel("長さ 30.0 秒")
        row.addWidget(self.len_label, 1)
        lay.addLayout(row)
        self.start_box.valueChanged.connect(self._boxes_changed)
        self.end_box.valueChanged.connect(self._boxes_changed)

        row = QHBoxLayout()
        self.preview_btn = QPushButton("▶ 区間を試聴")
        self.preview_btn.clicked.connect(self._preview)
        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.clicked.connect(self.player.stop)
        hint = QLabel("波形上の青い範囲をドラッグしても指定できます")
        hint.setStyleSheet("color:#7f8796;")
        row.addWidget(self.preview_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(hint, 1)
        lay.addLayout(row)

        self.player.positionChanged.connect(self._on_play_pos)
        self._set_range_enabled(False)

    # ------------------------------------------------------------------ 区間
    def selection(self) -> tuple[float, float]:
        return float(self.start_box.value()), float(self.end_box.value())

    def set_selection(self, s: float, e: float) -> None:
        d = self.duration or max(e, 1.0)
        s = max(0.0, min(s, d - 0.5))
        e = max(s + 0.5, min(e, d))
        for b in (self.start_box, self.end_box):
            b.blockSignals(True)
        self.start_box.setValue(s)
        self.end_box.setValue(e)
        for b in (self.start_box, self.end_box):
            b.blockSignals(False)
        self.region.blockSignals(True)
        self.region.setRegion((s, e))
        self.region.blockSignals(False)
        self._update_len()

    def _boxes_changed(self) -> None:
        s, e = self.selection()
        if e <= s:
            e = s + 0.5
        self.set_selection(s, e)

    def _region_moved(self) -> None:
        s, e = self.region.getRegion()
        for b, v in ((self.start_box, s), (self.end_box, e)):
            b.blockSignals(True)
            b.setValue(max(0.0, v))
            b.blockSignals(False)
        self._update_len()

    def _update_len(self) -> None:
        s, e = self.selection()
        L = e - s
        warn = "  ⚠ 長い区間は解析に時間がかかります" if L > 240 else ""
        self.len_label.setText(f"長さ {L:.1f} 秒{warn}")

    def _set_range_enabled(self, on: bool) -> None:
        for w in (self.start_box, self.end_box, self.preview_btn, self.stop_btn, self.overview):
            w.setEnabled(on)

    # ------------------------------------------------------------------ 取得
    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for w in (self.url_btn, self.search_btn, self.pick_btn, self.file_btn):
            w.setEnabled(not busy)

    def _fetch_url(self) -> None:
        url = self.url_edit.text().strip()
        if not url:
            return
        if not youtube.is_url(url):
            # URL でなければ検索として扱う
            self.search_edit.setText(url)
            self.tabs.setCurrentIndex(1)
            self._search()
            return
        self._pending_start = youtube.parse_start_time(url)
        self._download(url)

    def _search(self) -> None:
        q = self.search_edit.text().strip()
        if not q or self._busy:
            return
        self._set_busy(True)
        self.status.emit(f"「{q}」を検索中...", -1)
        start_worker(youtube.search, q, 20, use_progress=False,
                     on_done=self._search_done, on_error=self._error)

    def _search_done(self, results: list[VideoInfo]) -> None:
        self._set_busy(False)
        self._results = results
        self.table.setRowCount(len(results))
        for r, v in enumerate(results):
            items = [v.title, v.channel, v.duration_text, f"{v.view_count:,}" if v.view_count else "-"]
            for c, txt in enumerate(items):
                it = QTableWidgetItem(txt)
                it.setToolTip(f"{v.title}\n{v.url}")
                if c >= 2:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, it)
        self.status.emit(f"{len(results)} 件見つかりました。ダブルクリックで取得します。", 1.0)

    def _fetch_result(self, row: int) -> None:
        if 0 <= row < len(self._results):
            v = self._results[row]
            self.url_edit.setText(v.url)
            self._pending_start = None
            self._download(v.url)

    def _download(self, url: str) -> None:
        if self._busy:
            return
        self._set_busy(True)

        def job(progress):
            wav, info = youtube.download_audio(url, progress)
            return wav, info, compute_overview(wav)

        start_worker(job, on_done=self._loaded, on_error=self._error, on_progress=self.status.emit)

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "音声ファイルを開く", str(Path.home()),
                                              "音声/動画 (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.webm *.mp4 "
                                              "*.mkv *.mov);;すべて (*)")
        if not path or self._busy:
            return
        self._set_busy(True)
        self._pending_start = None

        def job(progress):
            wav, info = youtube.import_local_file(path, progress)
            return wav, info, compute_overview(wav)

        start_worker(job, on_done=self._loaded, on_error=self._error, on_progress=self.status.emit)

    def load_cached(self, wav: Path, info: VideoInfo, start: float | None = None, end: float | None = None) -> None:
        """既存の音源 (解析結果を開いた時など) をパネルに反映."""
        self._loaded((wav, info, compute_overview(wav)))
        if start is not None and end is not None:
            self.set_selection(start, end)

    def _loaded(self, res) -> None:
        wav, info, ov = res
        self._set_busy(False)
        self.wav, self.info = Path(wav), info
        self.duration = ov["duration"]
        t = np.repeat(ov["t"], 2)
        y = np.empty(t.size)
        y[0::2], y[1::2] = ov["mn"], ov["mx"]
        self.wave_item.setData(t, y, connect="pairs")
        self.overview.setXRange(0, self.duration, padding=0.01)
        self.overview.setYRange(-1, 1)
        self.overview.setLimits(xMin=0, xMax=self.duration)
        self.region.setBounds((0, self.duration))
        self.start_box.setMaximum(self.duration)
        self.end_box.setMaximum(self.duration)
        s = self._pending_start or 0.0
        self.set_selection(s, min(self.duration, s + 30.0))
        self._set_range_enabled(True)
        self.info_label.setText(f"{info.title}\n{info.channel}  ・  {fmt_time(self.duration, 0)}")
        self.status.emit("音源の準備ができました。区間を指定して「解析開始」を押してください。", 1.0)
        self.audioReady.emit(self.wav, info)

    def _error(self, msg: str) -> None:
        self._set_busy(False)
        self.status.emit(f"エラー: {msg}", 0.0)

    # ------------------------------------------------------------------ 試聴
    def _preview_prefix(self) -> str:
        return f"preview:{self.wav}:"

    def _preview(self) -> None:
        """指定区間だけを読み込んで試聴 (長い動画でも全体をメモリに載せない)."""
        if self.wav is None:
            return
        from ..audio_io import load_wav

        s, e = self.selection()
        sid = f"{self._preview_prefix()}{s:.3f}:{e:.3f}"
        if self.player.path != sid:
            y, sr = load_wav(self.wav, s, e)
            self.player.load_arrays({"mix": y}, sr, sid)
        self._preview_start = s
        self.player.play(0.0)

    def _on_play_pos(self, pos: float) -> None:
        if self.wav is not None and (self.player.path or "").startswith(self._preview_prefix()):
            self.playhead.setVisible(True)
            self.playhead.setValue(getattr(self, "_preview_start", 0.0) + pos)
        else:
            self.playhead.setVisible(False)
