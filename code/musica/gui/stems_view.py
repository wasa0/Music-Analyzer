"""パート (ステム) のミキサーと波形表示."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider,
                               QVBoxLayout, QWidget)

from ..separation import STEM_COLORS, STEM_LABELS, sort_stems
from .source_panel import TimeAxis


class MixerModel(QObject):
    """各パートのミュート・ソロ・音量を保持し、実際に鳴らすゲインを計算する."""

    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.stems: list[str] = []
        self.mute: dict[str, bool] = {}
        self.solo: dict[str, bool] = {}
        self.level: dict[str, float] = {}

    def set_stems(self, names) -> None:
        self.stems = sort_stems(names)
        # 旧形式の結果 (ミックス + ボーカル) はミックスだけ鳴らすのが「元の音」
        self.mute = {s: ("mix" in self.stems and s != "mix") for s in self.stems}
        self.solo = {s: False for s in self.stems}
        self.level = {s: 1.0 for s in self.stems}
        self.changed.emit()

    def gains(self) -> dict[str, float]:
        any_solo = any(self.solo.values())
        out = {}
        for s in self.stems:
            on = self.solo[s] if any_solo else not self.mute[s]
            out[s] = self.level[s] if on else 0.0
        return out

    def audible(self) -> list[str]:
        return [s for s, g in self.gains().items() if g > 0]

    def set_mute(self, s: str, on: bool) -> None:
        self.mute[s] = on
        self.changed.emit()

    def set_solo(self, s: str, on: bool) -> None:
        self.solo[s] = on
        self.changed.emit()

    def set_level(self, s: str, v: float) -> None:
        self.level[s] = v
        self.changed.emit()

    def preset_available(self, name: str) -> bool:
        """カラオケ (伴奏だけ) は伴奏が分離されているときだけ. 旧形式の結果 (mix + vocals) では不可."""
        if name == "all":
            return bool(self.stems)
        if "vocals" not in self.stems or len(self.stems) < 2:
            return False
        return not (name == "karaoke" and "mix" in self.stems)

    def preset(self, name: str) -> None:
        """'all' 全パート / 'vocals' ボーカルのみ / 'karaoke' ボーカル以外."""
        legacy = "mix" in self.stems           # 伴奏が分離されていない旧形式: ミックス か ボーカル
        for s in self.stems:
            self.solo[s] = False
            if legacy:
                self.mute[s] = (s != "vocals") if name == "vocals" else (s != "mix")
            else:
                self.mute[s] = (name == "vocals" and s != "vocals") or (name == "karaoke" and s == "vocals")
        self.changed.emit()


class MixerPanel(QWidget):
    """M (ミュート) / S (ソロ) / 音量スライダーを並べたミキサー."""

    def __init__(self, model: MixerModel, vertical: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.vertical = vertical
        self._lay = QGridLayout(self)
        self._lay.setContentsMargins(4, 4, 4, 4)
        self._lay.setHorizontalSpacing(6)
        self._lay.setVerticalSpacing(4)
        self._widgets: dict[str, tuple] = {}
        model.changed.connect(self._sync)

    def rebuild(self) -> None:
        while self._lay.count():
            it = self._lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._widgets.clear()
        presets = QHBoxLayout()
        for key, text, tip in (("all", "全パート", "すべてのパートを鳴らす"),
                               ("vocals", "ボーカルのみ", "歌だけを聴く"),
                               ("karaoke", "カラオケ", "ボーカル以外 (伴奏) を聴く")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self.model.preset(k))
            b.setEnabled(self.model.preset_available(key))
            presets.addWidget(b)
        holder = QWidget()
        holder.setLayout(presets)
        self._lay.addWidget(holder, 0, 0, 1, 5)
        for r, s in enumerate(self.model.stems, start=1):
            sw = QLabel()
            sw.setFixedSize(12, 12)
            sw.setStyleSheet(f"background:{STEM_COLORS.get(s, '#888')}; border-radius:6px;")
            name = QLabel(STEM_LABELS.get(s, s))
            name.setMinimumWidth(110)
            m = QPushButton("M")
            m.setCheckable(True)
            m.setFixedWidth(30)
            m.setToolTip("ミュート (このパートを消す)")
            m.setStyleSheet("QPushButton:checked{background:#c0392b;color:white;}")
            so = QPushButton("S")
            so.setCheckable(True)
            so.setFixedWidth(30)
            so.setToolTip("ソロ (このパートだけ聴く)")
            so.setStyleSheet("QPushButton:checked{background:#f1c40f;color:black;}")
            vol = QSlider(Qt.Horizontal)
            vol.setRange(0, 150)
            vol.setValue(100)
            vol.setMinimumWidth(80)
            vol.setToolTip("音量 (100 = 元の大きさ)")
            m.toggled.connect(lambda on, k=s: self.model.set_mute(k, on))
            so.toggled.connect(lambda on, k=s: self.model.set_solo(k, on))
            vol.valueChanged.connect(lambda v, k=s: self.model.set_level(k, v / 100.0))
            for c, w in enumerate((sw, name, m, so, vol)):
                self._lay.addWidget(w, r, c)
            self._widgets[s] = (name, m, so, vol)
        self._lay.setRowStretch(len(self.model.stems) + 1, 1)
        self._sync()

    def _sync(self) -> None:
        g = self.model.gains()
        for s, (name, m, so, vol) in self._widgets.items():
            for w, val in ((m, self.model.mute.get(s, False)), (so, self.model.solo.get(s, False))):
                w.blockSignals(True)
                w.setChecked(val)
                w.blockSignals(False)
            vol.blockSignals(True)
            vol.setValue(int(round(self.model.level.get(s, 1.0) * 100)))
            vol.blockSignals(False)
            name.setStyleSheet("color:#ffffff;" if g.get(s, 0) > 0 else "color:#5c6270;")


def envelope(y: np.ndarray, sr: int, n_bins: int = 2400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = y.mean(axis=0) if y.ndim == 2 else y
    size = max(1, len(m) // n_bins)
    k = len(m) // size
    blk = m[: k * size].reshape(k, size)
    return (np.arange(k) + 0.5) * size / sr, blk.min(1), blk.max(1)


class StemsView(QWidget):
    """左: ミキサー, 右: パートごとの波形 (時間軸は音程バーと連動)."""

    seekRequested = Signal(float)

    def __init__(self, model: MixerModel, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        lay = QHBoxLayout(self)
        left = QFrame()
        left.setFrameShape(QFrame.StyledPanel)
        ll = QVBoxLayout(left)
        title = QLabel("ミキサー")
        title.setStyleSheet("font-weight:bold; color:#9ecbff;")
        ll.addWidget(title)
        self.mixer = MixerPanel(model)
        ll.addWidget(self.mixer)
        hint = QLabel("M = ミュート, S = ソロ。再生中でもすぐ反映されます。\n"
                      "分離したパートは結果フォルダの stems/ に WAV で保存されています。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#7f8796; font-size:9pt;")
        ll.addWidget(hint)
        ll.addStretch(1)
        left.setMaximumWidth(380)
        left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        lay.addWidget(left)
        self.plots = pg.GraphicsLayoutWidget()
        lay.addWidget(self.plots, 1)
        self.items: dict[str, tuple] = {}
        self.cursors: list = []
        self.plots.scene().sigMouseClicked.connect(self._clicked)
        model.changed.connect(self._dim)

    def set_result(self, res, link_to=None) -> None:
        from ..audio_io import load_wav

        self.plots.clear()
        self.items.clear()
        self.cursors = []
        parts = res.playback_parts()
        self.model.set_stems(parts)
        self.mixer.rebuild()
        off = res.offset_s
        first = None
        for r, s in enumerate(self.model.stems):
            y, sr = load_wav(parts[s])
            t, mn, mx = envelope(y, sr)
            p = self.plots.addPlot(row=r, col=0, axisItems={"bottom": TimeAxis("bottom")})
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=True, y=False)
            p.hideAxis("left")
            p.setYRange(-1, 1, padding=0)
            if r < len(self.model.stems) - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            col = STEM_COLORS.get(s, "#888888")
            tt = np.repeat(t + off, 2)
            yy = np.empty(tt.size)
            yy[0::2], yy[1::2] = mn, mx
            it = pg.PlotDataItem(tt, yy, connect="pairs", pen=pg.mkPen(col, width=1))
            p.addItem(it)
            lab = pg.TextItem(STEM_LABELS.get(s, s), color=col, anchor=(0, 0))
            p.addItem(lab, ignoreBounds=True)
            lab.setPos(off, 1.0)
            cur = pg.InfiniteLine(angle=90, pen=pg.mkPen("#feca57", width=1.5))
            p.addItem(cur, ignoreBounds=True)
            self.cursors.append(cur)
            if first is None:
                first = p
                if link_to is not None:
                    p.setXLink(link_to)
            else:
                p.setXLink(first)
            self.items[s] = (p, it, lab)
        self._dim()

    def set_cursor(self, t_abs: float) -> None:
        for c in self.cursors:
            c.setValue(t_abs)

    def _dim(self) -> None:
        g = self.model.gains()
        for s, (_p, it, lab) in self.items.items():
            col = pg.mkColor(STEM_COLORS.get(s, "#888888"))
            if g.get(s, 0) <= 0:
                col.setAlpha(50)
            it.setPen(pg.mkPen(col, width=1))

    def _clicked(self, ev) -> None:
        if ev.button() != Qt.LeftButton or ev.isAccepted():
            return
        for p, _it, _lab in self.items.values():
            vb = p.getViewBox()
            if vb.sceneBoundingRect().contains(ev.scenePos()):
                self.seekRequested.emit(float(vb.mapSceneToView(ev.scenePos()).x()))
                return
