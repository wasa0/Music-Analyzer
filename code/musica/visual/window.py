"""オーディオビジュアライザーの画面.

- VisualizerPanel: 可視化の本体 (ツールバー + 描画パネル)。単体ウィンドウと、統合版のタブの両方で使う
- VisualizerWindow: ビジュアライザー単体のウィンドウ (解析しなくてもファイル / YouTube をすぐ可視化できる)

表示パート (ボーカル / フロント楽器 / カラオケ / 全パート) を選ぶと、コード・五度圏以外のパネルが
そのパートだけの音で描かれる (解析結果を開いたとき)。「パート別に並べる」レイアウトでは同じ種類の
パネルを全パートぶん並べて同時に見比べられる。鳴らす音は「再生:」のボタン (ミキサー) で別に選ぶ。
"""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QGridLayout,
                               QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSlider,
                               QVBoxLayout, QWidget)

from .. import config
from ..audio_io import load_wav, to_mono
from ..gui.common import app_icon, fmt_time, start_worker
from ..gui.player import AudioPlayer
from ..gui.source_panel import SourcePanel
from ..gui.stems_view import MixerModel
from ..separation import STEM_COLORS, STEM_LABELS
from ..youtube import import_local_file
from .dsp import VIEW_COLORS, VIEW_LABELS, VIEW_ORDER, FrameAnalyzer, Track, build_track, build_views, view_parts
from .widgets import (ChordWheelView, HarmonicStackView, MandalaView, OscilloscopeView, Panel, SpecViewport,
                      SpectrogramView, SpectrumView)

PANEL_CLASSES = {"mandala": MandalaView, "stack": HarmonicStackView, "scope": OscilloscopeView,
                 "spectrum": SpectrumView, "spectro": SpectrogramView, "chord": ChordWheelView}

# (表示名, パート別に並べるパネルの種類 / None = 表示パートを 1 つ選ぶ通常のレイアウト)
LAYOUTS = [
    ("全パネル", None),
    ("マンダラ", None),
    ("倍音分解 + 波形", None),
    ("スペクトラム + スペクトログラム", None),
    ("コード + スペクトログラム", None),
    ("パート別に並べる: スペクトログラム + ピッチ軌跡", "spectro"),
    ("パート別に並べる: スペクトラム", "spectrum"),
    ("パート別に並べる: 倍音分解", "stack"),
    ("パート別に並べる: マンダラ", "mandala"),
    ("パート別に並べる: 波形", "scope"),
]

VIEW_TIPS = {
    "vocals": "分離したボーカルだけで描きます (ピッチ軌跡と音程バーは解析結果)",
    "front": "フロント楽器 = ボーカルとリズム隊 (ドラム・ベース) 以外の楽器だけで描きます。\n"
             "ピッチはこのパートの音から推定します",
    "karaoke": "ボーカル以外 (伴奏全体) だけで描きます。ピッチはこのパートの音から推定します",
    "all": "すべてのパートを足した音 (元の曲) で描きます",
}

NO_RESULT_TEXT = ("「歌声アナライザー」のタブで解析すると、ここでパートごと (ボーカル / フロント楽器 / カラオケ / "
                  "全パート) に可視化できます。解析せずに見るときは「ツール → ビジュアライザー単体を開く」")


def _short_stem(name: str) -> str:
    return STEM_LABELS.get(name, name).split(" ")[0]


class YouTubeDialog(QDialog):
    """解析アプリと同じ音源パネル (URL / 検索 / ファイル + 区間) を使う取得ダイアログ."""

    def __init__(self, player: AudioPlayer, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("YouTube / ファイルから音源を取得")
        self.resize(720, 620)
        lay = QVBoxLayout(self)
        self.panel = SourcePanel(player)
        lay.addWidget(self.panel, 1)
        self.status = QLabel("URL を貼るか検索してください。取得後、区間の開始位置から再生します。")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.panel.status.connect(lambda m, _f: self.status.setText(m))
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("この音源を可視化")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.panel.audioReady.connect(lambda *_: self.buttons.button(QDialogButtonBox.Ok).setEnabled(True))


class CompactMixer(QWidget):
    """ツールバー用の小さなミキサー (鳴らす音): パートごとのオン/オフと 3 つのプリセット.

    ミキサーのパートが変わったら (別の解析結果を開いたときなど) ボタンを作り直す。
    """

    def __init__(self, model: MixerModel, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.btns: dict[str, QPushButton] = {}
        self._built: list[str] | None = None
        model.changed.connect(self._sync)

    def rebuild(self) -> None:
        while self.lay.count():
            it = self.lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.btns.clear()
        self._built = list(self.model.stems)
        if len(self.model.stems) <= 1:
            return
        head = QLabel("再生:")
        head.setToolTip("鳴らすパート (表示するパートとは別に選べます)")
        self.lay.addWidget(head)
        for s in self.model.stems:
            b = QPushButton(_short_stem(s))
            b.setCheckable(True)
            b.setChecked(True)
            col = STEM_COLORS.get(s, "#888")
            b.setStyleSheet(f"QPushButton:checked{{background:{col}; color:#101010; font-weight:bold;}}")
            b.setToolTip("クリックでこのパートの音のオン/オフ")
            b.toggled.connect(lambda on, k=s: self.model.set_mute(k, not on))
            self.lay.addWidget(b)
            self.btns[s] = b
        for key, text in (("all", "全部"), ("vocals", "歌だけ"), ("karaoke", "カラオケ")):
            b = QPushButton(text)
            b.setToolTip("鳴らす音のプリセット")
            b.clicked.connect(lambda _=False, k=key: self.model.preset(k))
            b.setEnabled(self.model.preset_available(key))
            self.lay.addWidget(b)
        self._sync()

    def _sync(self) -> None:
        if self._built != list(self.model.stems):
            self.rebuild()              # 最後に _sync をもう一度呼ぶ (パートは一致しているので作り直さない)
            return
        g = self.model.gains()
        for s, b in self.btns.items():
            b.blockSignals(True)
            b.setChecked(g.get(s, 0) > 0)
            b.blockSignals(False)


class VisualizerPanel(QWidget):
    """ビジュアライザーの本体.

    standalone=True : ファイル / YouTube / 解析結果を自分で開く (単体ウィンドウ)。読み込むとすぐ再生する。
    standalone=False: 統合版のタブ。歌声アナライザーの解析結果を show_result() で受け取る。
                      プレーヤーとミキサーはアナライザーと共有し、タブを切り替えても再生は続く。
    """

    fullScreenRequested = Signal(bool)
    fpsChanged = Signal(float)

    def __init__(self, player: AudioPlayer, mixer: MixerModel, standalone: bool = True, parent=None) -> None:
        super().__init__(parent)
        config.ensure_dirs()
        self.player = player
        self.mixer = mixer
        self.standalone = standalone
        self.track: Track | None = None             # 全パートの Track
        self.views: dict[str, Track] = {}            # 表示パート → Track
        self.analyzers: dict[str, FrameAnalyzer] = {}
        self.view_key = "all"
        self.result = None                           # 表示中 (または準備中) の解析結果
        self._stem_paths: dict[str, Path] = {}
        self._source_id: str | None = None
        self._pos = 0.0                              # プレーヤーが別の音を読み込んでいるときの表示位置
        self._pending = None                         # (解析結果, 開始位置): タブが表示されたら準備する
        self._loading = None                         # 準備中の解析結果
        self._want_start: float | None = None        # 準備が終わったときの表示位置
        self._token = 0                              # 読み込みの世代 (古い読み込み結果を捨てる)
        self._full = False
        self._fps_t = time.perf_counter()
        self._fps_n = 0
        self.spec_vp = SpecViewport(self)            # スペクトログラムの表示範囲 (全パネルで共有)
        self._panels: dict[tuple[str, str], Panel] = {}   # (種類, "main" または表示パート) → パネル
        self._shown: list[tuple[Panel, str, str]] = []    # 表示中の (パネル, 種類, スロット)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # ---- ツールバー
        bar = QHBoxLayout()
        self.open_btn = QPushButton("📂 ファイル")
        self.open_btn.clicked.connect(self._open_file)
        self.yt_btn = QPushButton("▶ YouTube から")
        self.yt_btn.clicked.connect(self._open_youtube)
        self.res_btn = QPushButton("📊 解析結果を開く")
        self.res_btn.clicked.connect(self._open_result_dialog)
        self.play_btn = QPushButton("▶ 再生")
        self.play_btn.setToolTip("再生 / 一時停止 (Space)")
        self.play_btn.clicked.connect(self._toggle)
        self.seek = QSlider(Qt.Horizontal)
        self.seek.setRange(0, 1000)
        self.seek.valueChanged.connect(lambda v: self._seek_to(v / 1000.0))
        self.time_label = QLabel("0:00.0 / 0:00.0")
        self.time_label.setStyleSheet("font-family:monospace; color:#feca57; padding:0 6px;")
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(int(round(player.volume() * 100)))
        self.vol.setMaximumWidth(100)
        self.vol.valueChanged.connect(lambda v: self.player.set_volume(v / 100))
        self.layout_combo = QComboBox()
        for name, _kind in LAYOUTS:
            self.layout_combo.addItem(name)
        self.layout_combo.setToolTip("表示するパネルの組み合わせ。「パート別に並べる」は同じパネルを全パートぶん並べます")
        self.layout_combo.currentIndexChanged.connect(self._apply_layout)
        self.full_btn = QPushButton("⛶ 全画面 (F11)")
        self.full_btn.clicked.connect(self.toggle_fullscreen)
        if standalone:
            for wdg in (self.open_btn, self.yt_btn, self.res_btn):
                bar.addWidget(wdg)
        else:
            for wdg in (self.open_btn, self.yt_btn, self.res_btn):
                wdg.hide()
        bar.addWidget(self.play_btn)
        bar.addWidget(self.seek, 1)
        for wdg in (self.time_label, QLabel("音量"), self.vol, self.layout_combo, self.full_btn):
            bar.addWidget(wdg)
        self.bar_widget = QWidget()
        self.bar_widget.setLayout(bar)
        root.addWidget(self.bar_widget)

        # ---- 2 段目: 表示パート / 情報 / 鳴らすパート
        row2 = QHBoxLayout()
        self.view_box = QWidget()
        vb = QHBoxLayout(self.view_box)
        vb.setContentsMargins(0, 0, 8, 0)
        head = QLabel("表示:")
        head.setToolTip("どのパートの音で描くか (コード・五度圏はいつも全体)")
        vb.addWidget(head)
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        self.view_btns: dict[str, QPushButton] = {}
        for key in VIEW_ORDER:
            b = QPushButton(VIEW_LABELS[key])
            b.setCheckable(True)
            b.setStyleSheet(f"QPushButton:checked{{background:{VIEW_COLORS[key]}; color:#101010; font-weight:bold;}}")
            b.setToolTip(VIEW_TIPS[key])
            b.clicked.connect(lambda _=False, k=key: self.set_view(k))
            self.view_group.addButton(b)
            vb.addWidget(b)
            self.view_btns[key] = b
        self.view_btns["all"].setChecked(True)
        self.view_box.setVisible(False)
        row2.addWidget(self.view_box)
        self.info = QLabel("音声ファイルをドラッグ&ドロップ、または YouTube から取得してください" if standalone
                           else NO_RESULT_TEXT)
        self.info.setStyleSheet("color:#9aa4b8;")
        self.info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # 長い曲名でも窓を広げない
        row2.addWidget(self.info, 1)
        self.fps_label = QLabel("")
        self.fps_label.setStyleSheet("color:#5c6270; padding:0 6px;")
        row2.addWidget(self.fps_label)
        self.compact_mixer = CompactMixer(self.mixer)
        self.compact_mixer.rebuild()
        row2.addWidget(self.compact_mixer)
        self.info_widget = QWidget()
        self.info_widget.setLayout(row2)
        root.addWidget(self.info_widget)

        # ---- パネル
        self.grid = QGridLayout()
        self.grid.setSpacing(4)
        root.addLayout(self.grid, 1)
        self._apply_layout(0)
        self._update_layout_items()

        self.timer = QTimer(self)                    # 表示されている間だけ動かす (showEvent / hideEvent)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.player.playingChanged.connect(self._on_playing)
        if standalone:
            self.mixer.changed.connect(self._mixer_changed)
        self.setAcceptDrops(standalone)
        # ショートカットはこのパネルが見えている時だけ有効 (統合版でアナライザーのタブと衝突しない)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._toggle)
        QShortcut(QKeySequence(Qt.Key_F11), self, activated=self.toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=lambda: self._full and self.set_fullscreen(False))
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self._seek_to(self._current_time() - 5))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self._seek_to(self._current_time() + 5))

    # ------------------------------------------------------------------ パネルの配置
    def _panel(self, kind: str, slot: str) -> Panel:
        """slot = "main" (選んだ表示パートで描く) / 表示パートのキー (パート別に並べる時)."""
        p = self._panels.get((kind, slot))
        if p is None:
            p = PANEL_CLASSES[kind](self)
            p.hide()
            if isinstance(p, SpectrogramView):
                p.set_viewport(self.spec_vp)
            self._panels[(kind, slot)] = p
        return p

    def _panel_view(self, kind: str, slot: str) -> str:
        if kind == "chord":
            return "all"                 # コード・五度圏はいつも全パート
        return self.view_key if slot == "main" else slot

    def _bind(self, p: Panel, kind: str, slot: str) -> None:
        vk = self._panel_view(kind, slot)
        tr = self.views.get(vk)
        if hasattr(p, "set_track"):
            p.set_track(tr)
        p.set_tag(VIEW_LABELS[vk] if len(self.views) > 1 and kind != "chord" else "", VIEW_COLORS.get(vk))

    def _compare_kind(self) -> str | None:
        idx = self.layout_combo.currentIndex()
        return LAYOUTS[idx][1] if 0 <= idx < len(LAYOUTS) else None

    def _apply_layout(self, idx: int) -> None:
        if not (0 <= idx < len(LAYOUTS)):
            return
        for p in self._panels.values():
            self.grid.removeWidget(p)
            p.hide()
        g = self.grid
        for r in range(6):
            g.setRowStretch(r, 0)
        for c in range(4):
            g.setColumnStretch(c, 0)
        placed: list[tuple[str, str, tuple]] = []      # (種類, スロット, グリッド位置)
        compare = LAYOUTS[idx][1]
        if compare is None:
            if idx == 0:
                placed = [("mandala", "main", (0, 0, 2, 1)), ("stack", "main", (0, 1)), ("scope", "main", (1, 1)),
                          ("chord", "main", (0, 2, 2, 1)), ("spectrum", "main", (2, 0, 1, 3)),
                          ("spectro", "main", (3, 0, 1, 3))]
                rows, cols = (3, 2, 2, 3), (5, 5, 4)
            elif idx == 1:
                placed, rows, cols = [("mandala", "main", (0, 0))], (1,), (1,)
            elif idx == 2:
                placed, rows, cols = [("stack", "main", (0, 0)), ("scope", "main", (1, 0))], (3, 2), (1,)
            elif idx == 3:
                placed, rows, cols = [("spectrum", "main", (0, 0)), ("spectro", "main", (1, 0))], (2, 3), (1,)
            else:
                placed, rows, cols = [("chord", "main", (0, 0)), ("spectro", "main", (1, 0))], (3, 2), (1,)
        else:
            keys = [k for k in VIEW_ORDER if k in self.views] or ["all"]
            n = len(keys)
            if compare in ("spectro", "spectrum"):          # 横長のパネルは縦に積む (時間・周波数の軸がそろう)
                placed = [(compare, k, (i, 0)) for i, k in enumerate(keys)]
                rows, cols = (1,) * n, (1,)
            elif n <= 2:
                placed = [(compare, k, (0, i)) for i, k in enumerate(keys)]
                rows, cols = (1,), (1,) * n
            else:
                placed = [(compare, k, (i // 2, i % 2)) for i, k in enumerate(keys)]
                rows, cols = (1, 1), (1, 1)
        for r, s in enumerate(rows):
            g.setRowStretch(r, s)
        for c, s in enumerate(cols):
            g.setColumnStretch(c, s)
        self._shown = []
        for kind, slot, pos in placed:
            p = self._panel(kind, slot)
            g.addWidget(p, *pos)
            self._bind(p, kind, slot)
            p.show()
            self._shown.append((p, kind, slot))
        self.view_box.setEnabled(compare is None)
        self.view_box.setToolTip("" if compare is None else "「パート別に並べる」では全パートを同時に表示しています")
        self._update_info()

    def _update_layout_items(self) -> None:
        """パートが 1 つしか無いときは「パート別に並べる」を選べなくする."""
        multi = len(self.views) > 1
        model = self.layout_combo.model()
        for i, (_name, kind) in enumerate(LAYOUTS):
            item = model.item(i) if hasattr(model, "item") else None
            if kind is not None and item is not None:
                item.setEnabled(multi)
        if not multi and self._compare_kind() is not None:
            self.layout_combo.setCurrentIndex(0)

    # ------------------------------------------------------------------ 表示パート
    def set_view(self, key: str) -> None:
        """表示パートを切り替える (コード・五度圏以外のパネルが、そのパートだけの音で描かれる)."""
        if self.views and key not in self.views:
            return
        self.view_key = key
        b = self.view_btns.get(key)
        if b is not None and not b.isChecked():
            b.setChecked(True)
        for (kind, slot), p in self._panels.items():
            if slot == "main":
                self._bind(p, kind, slot)
        self._update_info()

    def available_views(self) -> list[str]:
        return list(self.views)

    def _view_tip(self, key: str) -> str:
        tip = VIEW_TIPS[key]
        parts = view_parts(key, self.track.stems) if self.track is not None else None
        if parts:
            tip += "\n使うパート: " + "、".join(STEM_LABELS.get(s, s) for s in parts)
        return tip

    def _update_info(self) -> None:
        tr = self.track
        if tr is None:
            return
        v = self.views.get(self.view_key, tr)
        # 幅が足りないと右側が切れるので、大事なものから並べる (全文はツールチップ)
        items = [tr.label]
        if tr.key:
            items.append(f"キー {tr.key.get('name')}")
        if self._compare_kind() is None:
            if len(self.views) > 1:
                parts = view_parts(self.view_key, tr.stems) or []
                items.append(f"表示: {v.view_label} ({' + '.join(_short_stem(s) for s in parts)})")
            items.append(f"f0: {v.f0_source}")
        items.append(fmt_time(tr.duration))
        text = "   ・   ".join(items)
        self.info.setText(text)
        self.info.setToolTip(text)

    # ------------------------------------------------------------------ 読み込み
    def _clear_track(self, msg: str = "") -> None:
        self.track, self.views, self.analyzers = None, {}, {}
        self._source_id, self._stem_paths = None, {}
        for p in self._panels.values():
            if hasattr(p, "set_track"):
                p.set_track(None)
            p.frame = None
            p.set_tag("")
        self.view_box.setVisible(False)
        self._update_layout_items()
        if msg:
            self.info.setText(msg)

    def _set_track(self, track: Track, views: dict[str, Track], stems: dict[str, Path], source_id: str,
                   start: float | None = None, autoplay: bool = False,
                   playback_gains: dict[str, float] | None = None) -> None:
        self.track = track
        self.views = views or {"all": track}
        self.analyzers = {k: FrameAnalyzer(v) for k, v in self.views.items()}
        self._stem_paths = dict(stems)
        self._source_id = source_id
        if self.view_key not in self.views:
            self.view_key = "all"
        for k, b in self.view_btns.items():
            b.setVisible(k in self.views)
            b.setToolTip(self._view_tip(k))
        self.view_btns[self.view_key].setChecked(True)
        self.view_box.setVisible(len(self.views) > 1)
        self._update_layout_items()
        if self.standalone:
            # 単体版はミキサーもこの曲用に作り直して、すぐ鳴らせるようにする
            self.mixer.set_stems(list(track.stems))
            for k, v in (playback_gains or {}).items():
                if k in self.mixer.mute and v <= 0:
                    self.mixer.mute[k] = True
            self.mixer.changed.emit()
            self.player.load_stems(stems, source_id, self.mixer.gains())
        self._apply_layout(self.layout_combo.currentIndex())
        if start is not None:
            self._set_pos(start)
        elif self._audio_is_ours():
            self._pos = self.player.position()
        else:
            self._pos = 0.0
        if autoplay and self._audio_is_ours():
            self.player.play(self._pos)

    def _busy(self, msg: str, _frac: float = -1) -> None:
        self.info.setText(msg)

    def _error(self, msg: str) -> None:
        self.info.setText(f"エラー: {msg}")
        QMessageBox.critical(self, "エラー", msg)

    def _start_job(self, job, on_done) -> None:
        """別スレッドで読み込む. 後から別の読み込みが始まったら、古い方の結果は捨てる."""
        self._token += 1
        token = self._token

        def done(r):
            if token == self._token:
                self._loading = None
                on_done(r)

        def fail(msg):
            if token == self._token:
                self._loading = None
                self._error(msg)

        def prog(msg, frac):
            if token == self._token:
                self._busy(msg, frac)

        start_worker(job, on_done=done, on_error=fail, on_progress=prog)

    def load_file(self, path: str | Path, start: float | None = None) -> None:
        def job(progress):
            wav, info = import_local_file(path, progress)
            y, sr = load_wav(wav)
            return wav, build_track({"mix": to_mono(y)}, sr, info.title, progress=progress)

        self.result = None
        self._start_job(job, lambda r: self._set_track(r[1], {"all": r[1]}, {"mix": r[0]}, str(r[0]), start,
                                                       autoplay=True))

    def load_wav(self, wav: Path, label: str, start: float | None = None) -> None:
        def job(progress):
            y, sr = load_wav(wav)
            return build_track({"mix": to_mono(y)}, sr, label, progress=progress)

        self.result = None
        self._start_job(job, lambda tr: self._set_track(tr, {"all": tr}, {"mix": wav}, str(wav), start,
                                                        autoplay=True))

    def load_url(self, url: str, start: float | None = None) -> None:
        """YouTube の URL から取得して可視化 (t= の開始位置も反映)."""
        from .. import youtube

        if start is None:
            start = youtube.parse_start_time(url)
        self._start_job(lambda progress: youtube.download_audio(url, progress),
                        lambda r: self.load_wav(r[0], r[1].title, start=start))

    def load_result(self, folder: str | Path, source: str = "mix", gains: dict[str, float] | None = None,
                    start: float | None = None) -> None:
        """単体版: 解析結果 (分離した全パート・ピッチ・コード) を読み込んで再生する.

        source='vocal' ならボーカルだけを鳴らして表示する。gains は鳴らす音の初期設定。
        """
        from ..analysis import AnalysisResult

        res = AnalysisResult.load(folder)
        stems = res.playback_parts()
        if gains is None:
            if source == "vocal":
                gains = {s: (1.0 if s == "vocals" else 0.0) for s in stems}
            elif "mix" in stems:              # 旧形式の結果: ミックスだけ鳴らす
                gains = {s: (1.0 if s == "mix" else 0.0) for s in stems}
            else:
                gains = {s: 1.0 for s in stems}
        if source == "vocal":
            self.view_key = "vocals"
        self._load_result(res, start=start if start is not None else 0.0, autoplay=True, playback_gains=gains)

    def show_result(self, res, start: float | None = None) -> None:
        """統合版: アナライザーの解析結果を表示する. 準備 (数秒) はこのタブが表示されたときに行う."""
        if res is self.result:
            if self._pending is not None:              # まだ準備していない
                if start is not None:
                    self._pending = (res, start)
                return
            if self._loading is res:                   # 準備中: 終わったらこの位置から
                if start is not None:
                    self._want_start = start
                return
            if self.track is not None:                 # 表示中
                if start is not None:
                    self._set_pos(start)
                return
        self.result = res
        self._token += 1                       # 準備中だった前の結果は捨てる
        self._clear_track("解析結果の準備待ち (このタブを開くと準備します)")
        self._pending = (res, start)
        if self.isVisible():
            self._build_pending()

    def _build_pending(self) -> None:
        res, start = self._pending
        self._pending = None
        self._load_result(res, start=start)

    def _load_result(self, res, start: float | None = None, autoplay: bool = False,
                     playback_gains: dict[str, float] | None = None) -> None:
        self.result = res
        stems = res.playback_parts()
        tune = float(res.meta.get("tuning_offset", 0.0))
        # 音程バーはチューニング補正後の整数音高なので、補正量を戻して実際の周波数に合わせる
        notes = [(n.features["start_s"], n.features["end_s"], n.midi + tune) for n in res.notes]
        label = f"{res.meta.get('title', '')} [{fmt_time(res.meta['start'])}〜{fmt_time(res.meta['end'])}]"
        harmony = res.harmony or {}
        f0_t, f0, offset = res.times, res.f0_hz, res.offset_s

        def job(progress):
            data = {}
            sr = 44100
            for s, p in stems.items():
                y, sr = load_wav(p)
                data[s] = to_mono(y)
            tr = build_track(data, sr, label, f0_t=f0_t, f0=f0, notes=notes, chords=harmony.get("chords"),
                             key=harmony.get("key"), beats=harmony.get("beats"), time_offset=offset,
                             progress=progress)
            return tr, build_views(tr, progress)

        self._busy("解析結果を可視化の準備中...")
        self._want_start = start
        self._loading = res
        self._start_job(job, lambda r: self._set_track(r[0], r[1], stems, f"result:{res.folder}", self._want_start,
                                                       autoplay=autoplay, playback_gains=playback_gains))

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "音声ファイルを開く", str(Path.home()),
                                              "音声/動画 (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.webm *.mp4 "
                                              "*.mkv *.mov);;すべて (*)")
        if path:
            self.load_file(path)

    def _open_youtube(self) -> None:
        dlg = YouTubeDialog(self.player, self)
        if dlg.exec() == QDialog.Accepted and dlg.panel.wav is not None:
            s, _e = dlg.panel.selection()
            self.load_wav(dlg.panel.wav, dlg.panel.info.title if dlg.panel.info else dlg.panel.wav.name, start=s)

    def _open_result_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "解析結果フォルダを選択", str(config.RESULT_DIR))
        if d:
            try:
                self.load_result(d)
            except Exception as exc:
                QMessageBox.critical(self, "読み込みエラー", f"{type(exc).__name__}: {exc}")

    def dragEnterEvent(self, ev) -> None:  # noqa: N802
        if self.standalone and ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev) -> None:  # noqa: N802
        urls = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        if self.standalone and urls:
            p = Path(urls[0])
            if p.is_dir() and (p / "analysis.json").exists():
                self.load_result(p)
            else:
                self.load_file(p)

    # ------------------------------------------------------------------ 再生
    def _audio_is_ours(self) -> bool:
        return self._source_id is not None and self.player.path == self._source_id

    def _ensure_audio(self) -> None:
        """プレーヤーに別の音 (アナライザーの試聴など) が読み込まれていたら、この曲に戻す."""
        if self.track is None or self._audio_is_ours():
            return
        self.player.load_stems(self._stem_paths, self._source_id, self.mixer.gains())
        self.player.seek(self._pos)

    def _current_time(self) -> float:
        if self._audio_is_ours():
            self._pos = self.player.position()
        return self._pos

    def _set_pos(self, t: float) -> None:
        """表示位置を合わせる (この曲を再生中ならプレーヤーも. ほぼ同じ位置ならそのまま)."""
        dur = self.track.duration if self.track is not None else max(0.0, t)
        self._pos = max(0.0, min(float(t), dur))
        if self._audio_is_ours() and abs(self.player.position() - self._pos) > 0.05:
            self.player.seek(self._pos)

    def _seek_to(self, t: float) -> None:
        if self.track is None:
            return
        self._ensure_audio()
        self._pos = max(0.0, min(float(t), self.track.duration))
        self.player.seek(self._pos)

    def _toggle(self) -> None:
        if self.player.is_playing():
            self.player.pause()
            return
        if self.track is None:
            return
        self._ensure_audio()
        self.player.set_gains(self.mixer.gains())
        pos = self.player.position()
        self.player.play(0.0 if pos >= self.track.duration - 0.05 else None)

    def _on_playing(self, on: bool) -> None:
        self.play_btn.setText("⏸ 一時停止" if on else "▶ 再生")

    def _mixer_changed(self) -> None:
        if self._audio_is_ours():
            self.player.set_gains(self.mixer.gains())

    # ------------------------------------------------------------------ 全画面
    def toggle_fullscreen(self) -> None:
        self.set_fullscreen(not self._full)

    def set_fullscreen(self, on: bool, notify: bool = True) -> None:
        """ツールバーを隠してパネルだけにする. ウィンドウ側の全画面化は fullScreenRequested で."""
        if on == self._full:
            return
        self._full = on
        self.bar_widget.setVisible(not on)
        self.info_widget.setVisible(not on)
        if notify:
            self.fullScreenRequested.emit(on)

    @property
    def is_fullscreen(self) -> bool:
        return self._full

    # ------------------------------------------------------------------ 描画ループ
    def showEvent(self, ev) -> None:  # noqa: N802
        super().showEvent(ev)
        self.vol.blockSignals(True)
        self.vol.setValue(int(round(self.player.volume() * 100)))     # 統合版ではアナライザーと共有
        self.vol.blockSignals(False)
        self._fps_t, self._fps_n = time.perf_counter(), 0
        self.timer.start()
        if self._pending is not None:
            self._build_pending()

    def hideEvent(self, ev) -> None:  # noqa: N802
        super().hideEvent(ev)
        self.timer.stop()

    def _tick(self) -> None:
        if self.track is None or not self._shown:
            return
        t = self._current_time()
        # 表示パートごとに、そのパートを描くパネルが必要な分だけ解析する
        need: dict[str, int] = {}
        for p, kind, slot in self._shown:
            vk = self._panel_view(kind, slot)
            need[vk] = max(need.get(vk, -1), p.detail)
        frames = {vk: self.analyzers[vk].analyze(t, d) for vk, d in need.items() if vk in self.analyzers}
        for p, kind, slot in self._shown:
            fr = frames.get(self._panel_view(kind, slot))
            if fr is not None:
                p.set_frame(fr)
        dur, off = self.track.duration, self.track.time_offset
        if not self.seek.isSliderDown():
            self.seek.blockSignals(True)
            self.seek.setRange(0, int(dur * 1000))
            self.seek.setValue(int(t * 1000))
            self.seek.blockSignals(False)
        self.time_label.setText(f"{fmt_time(off + t)} / {fmt_time(off + dur)}")
        self._fps_n += 1
        now = time.perf_counter()
        if now - self._fps_t >= 1.0:
            fps = self._fps_n / (now - self._fps_t)
            self._fps_t, self._fps_n = now, 0
            self.fps_label.setText(f"{fps:.0f} fps")
            self.fpsChanged.emit(fps)

    def shutdown(self) -> None:
        """閉じるとき用: 描画を止め、読み込み中の結果は捨てる."""
        self.timer.stop()
        self._token += 1
        self._pending = None


class VisualizerWindow(QMainWindow):
    """ビジュアライザー単体のウィンドウ (解析しなくてもファイル / YouTube をすぐ可視化できる)."""

    TITLE = "MUSICA オーディオビジュアライザー"

    def __init__(self) -> None:
        super().__init__()
        config.ensure_dirs()
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(app_icon())
        self.resize(1600, 1000)
        self.player = AudioPlayer(self)
        self.mixer = MixerModel(self)
        self.panel = VisualizerPanel(self.player, self.mixer, standalone=True)
        self.setCentralWidget(self.panel)
        self._was_max = False
        self.panel.fullScreenRequested.connect(self._set_fullscreen)
        self.panel.fpsChanged.connect(lambda fps: self.setWindowTitle(f"{self.TITLE}  ({fps:.0f} fps)"))

    # 読み込み (app.py / アナライザーから呼ばれる)
    def load_file(self, path: str | Path, start: float | None = None) -> None:
        self.panel.load_file(path, start=start)

    def load_wav(self, wav: Path, label: str, start: float | None = None) -> None:
        self.panel.load_wav(wav, label, start=start)

    def load_url(self, url: str, start: float | None = None) -> None:
        self.panel.load_url(url, start=start)

    def load_result(self, folder: str | Path, source: str = "mix", gains: dict[str, float] | None = None,
                    start: float | None = None) -> None:
        self.panel.load_result(folder, source, gains=gains, start=start)

    def _set_fullscreen(self, on: bool) -> None:
        if on:
            self._was_max = self.isMaximized()
            self.showFullScreen()
        elif self._was_max:
            self.showMaximized()
        else:
            self.showNormal()

    def changeEvent(self, ev) -> None:  # noqa: N802
        super().changeEvent(ev)
        # ウィンドウマネージャー側の操作で全画面が解除されたら、隠したツールバーも戻す
        if ev.type() == QEvent.WindowStateChange and not self.isFullScreen() and self.panel.is_fullscreen:
            self.panel.set_fullscreen(False, notify=False)

    def closeEvent(self, ev) -> None:  # noqa: N802
        self.panel.shutdown()
        self.player.shutdown()
        super().closeEvent(ev)


__all__ = ["VisualizerPanel", "VisualizerWindow", "LAYOUTS"]
