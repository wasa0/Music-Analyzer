"""全体傾向のサマリー / ノート一覧表 / 全体スペクトログラム."""
from __future__ import annotations

import html

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QSplitter, QTableWidget, QTableWidgetItem,
                               QTextBrowser, QVBoxLayout, QWidget)

from .. import config
from ..analysis import AnalysisResult
from ..features import APPROACH_LABELS, RELEASE_LABELS
from .common import APPROACH_COLORS, RELEASE_COLORS, fmt_time
from .pianoroll import chord_color
from .source_panel import TimeAxis


class _NumItem(QTableWidgetItem):
    """数値でソートできるセル."""

    def __init__(self, text: str, value: float) -> None:
        super().__init__(text)
        self._v = value if value is not None and np.isfinite(value) else -1e18
        self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def __lt__(self, other):
        return self._v < getattr(other, "_v", 0)


class SummaryView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        lay.addWidget(split)
        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        split.addWidget(self.text)
        self.charts = pg.GraphicsLayoutWidget()
        split.addWidget(self.charts)
        split.setSizes([560, 640])
        self.p_app = self.charts.addPlot(row=0, col=0, title="入り方の内訳")
        self.p_rel = self.charts.addPlot(row=0, col=1, title="語尾処理の内訳")
        self.p_vib = self.charts.addPlot(row=1, col=0, title="ビブラート: 速さ × 深さ")
        self.p_att = self.charts.addPlot(row=1, col=1, title="しゃくり: 深さ × 所要時間")
        self.p_int = self.charts.addPlot(row=2, col=0, colspan=2, title="音程ずれ (中心音高 − バー) の分布")
        for p in (self.p_app, self.p_rel, self.p_vib, self.p_att, self.p_int):
            p.setMenuEnabled(False)
            p.showGrid(x=True, y=True, alpha=0.15)

    def set_result(self, res: AnalysisResult) -> None:
        S = res.summary
        m = res.meta
        lines = [f"<h2 style='color:#ffffff'>{html.escape(m.get('title', ''))}</h2>",
                 f"<p style='color:#9aa4b8'>{html.escape(m.get('channel', ''))} ・ 区間 {fmt_time(m['start'])} 〜 "
                 f"{fmt_time(m['end'])} ・ 分離: {m['method']} ・ 解析 {m.get('elapsed_s', 0):.1f} 秒</p>"]
        if m.get("url", "").startswith("http"):
            lines.append(f"<p><a style='color:#70a1ff' href='{html.escape(m['url'])}'>{html.escape(m['url'])}</a></p>")
        lines.append("<h3 style='color:#9ecbff'>歌い方の傾向</h3><ul>")
        for d in S.get("description", []):
            if d.startswith("【"):
                lines.append(f"</ul><h3 style='color:#feca57'>{html.escape(d)}</h3><ul>")
            else:
                lines.append(f"<li style='margin-bottom:4px'>{html.escape(d.lstrip('・ '))}</li>")
        lines.append("</ul>")
        if S.get("n_notes"):
            ev = S["event_counts"]
            rows = [
                ("ノート数 / レガート率", f"{S['n_notes']} / {S['legato_ratio'] * 100:.0f}%"),
                ("音程ずれ 平均 / 絶対平均", f"{S['intonation']['mean']:+.1f} / {S['abs_intonation']['mean']:.1f} cent"),
                ("しゃくり率 / オーバーシュート率", f"{S['scoop_ratio'] * 100:.0f}% / {S['overshoot_ratio'] * 100:.0f}%"),
                ("フォール率 / 語尾上げ率", f"{S['fall_ratio'] * 100:.0f}% / {S['rise_ratio'] * 100:.0f}%"),
                ("ビブラート率 (400ms以上)", f"{S['vib_ratio'] * 100:.0f}% ({S['n_long_notes']} ノート中)"),
                ("イベント (上ずり/こぶし/落ち/跳躍)",
                 f"{ev['spike_up']} / {ev['kobushi']} / {ev['spike_down'] + ev['dip']} / {ev['flip']}"),
            ]
            lines.append("<h3 style='color:#9ecbff'>主要な数値</h3><table cellspacing=0 cellpadding=4>")
            for k, v in rows:
                lines.append(f"<tr><td style='color:#9aa4b8'>{k}</td><td style='color:#ffffff'><b>{v}</b></td></tr>")
            lines.append("</table>")
        chords = [c for c in res.chords if c.get("root") is not None]
        if chords:
            key = res.harmony.get("key", {})
            lines.append(f"<h3 style='color:#9ecbff'>コード進行</h3><p style='color:#9aa4b8'>キー {html.escape(key.get('label', '-'))}"
                         f" ・ テンポ 約 {res.harmony.get('tempo', 0):.0f} BPM ・ 区間内 {len(chords)} コード</p>"
                         "<p style='line-height:170%'>")
            cells = []
            for c in chords:
                col = pg.mkColor(chord_color(c["root"])).name()
                cells.append(f"<span style='background:{col}; color:white; padding:2px 6px; border-radius:3px;'>"
                             f"<b>{html.escape(c['label'])}</b></span> <span style='color:#7f8796; font-size:8pt'>"
                             f"{html.escape(c['degree'])} {fmt_time(c['start'] + res.offset_s, 1)}</span>")
            lines.append(" &nbsp; ".join(cells) + "</p>")
        lines.append("<p style='color:#6c7280'>※ 数値は分離後の音声からの自動推定です。伴奏の混入や子音で誤差が出ることがあります。"
                     " コードは伴奏パートのクロマ (12 音名の強さ) からの推定で、テンションや転回形は省略されることがあります。"
                     " DYN/BRE/BRI は曲内の相対値を 0〜127 に割り当てた目安です。</p>")
        self.text.setHtml("\n".join(lines))
        self._charts(res)

    SHORT = {"scoop": "しゃくり", "fall_in": "上から", "portamento": "ポルタ", "straight": "直入り",
             "fall": "フォール", "rise": "上げ", "preparation": "予備", "flat": "そのまま"}

    def _bar(self, p, labels: dict, counts: dict, colors: dict) -> None:
        p.clear()
        keys = list(labels)
        vals = [counts.get(k, 0) for k in keys]
        bg = pg.BarGraphItem(x=np.arange(len(keys)), height=vals, width=0.7,
                             brushes=[pg.mkBrush(colors[k]) for k in keys], pens=[pg.mkPen(None)] * len(keys))
        p.addItem(bg)
        ax = p.getAxis("bottom")
        ax.setTicks([[(i, self.SHORT.get(k, labels[k])) for i, k in enumerate(keys)]])
        p.setXRange(-0.6, len(keys) - 0.4, padding=0)

    def _charts(self, res: AnalysisResult) -> None:
        S = res.summary
        fs = [n.features for n in res.notes]
        if not fs:
            for p in (self.p_app, self.p_rel, self.p_vib, self.p_att, self.p_int):
                p.clear()
            return
        self._bar(self.p_app, APPROACH_LABELS, S["approach_counts"], APPROACH_COLORS)
        self._bar(self.p_rel, RELEASE_LABELS, S["release_counts"], RELEASE_COLORS)
        self.p_vib.clear()
        vb = [f for f in fs if f["vib"]]
        if vb:
            self.p_vib.plot([f["vib_rate_hz"] for f in vb], [f["vib_extent_cents"] for f in vb], pen=None,
                            symbol="o", symbolSize=8, symbolBrush=(255, 159, 243, 180))
        self.p_vib.setLabel("bottom", "速さ", units="Hz")
        self.p_vib.setLabel("left", "深さ ±cent")
        self.p_att.clear()
        sc = [f for f in fs if f["approach"] == "scoop"]
        if sc:
            self.p_att.plot([f["attack_ms"] for f in sc], [f["attack_depth_cents"] for f in sc], pen=None,
                            symbol="t1", symbolSize=9, symbolBrush=(255, 159, 67, 200))
        self.p_att.setLabel("bottom", "所要時間", units="ms")
        self.p_att.setLabel("left", "深さ cent")
        self.p_int.clear()
        vals = np.array([f["intonation_cents"] for f in fs])
        hist, edges = np.histogram(vals, bins=np.arange(-50, 52.5, 5))
        self.p_int.addItem(pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=hist, brush="#48dbfb", pen=None))
        self.p_int.addItem(pg.InfiniteLine(angle=90, pos=0, pen=pg.mkPen("#ffffff", width=1)))
        self.p_int.setLabel("bottom", "cent (+ = 高め)")


class NotesTable(QTableWidget):
    noteSelected = Signal(int)
    COLS = ["#", "開始", "長さ ms", "音名", "音程ずれ", "入り", "入り深さ", "入り ms", "OS", "語尾", "語尾深さ",
            "語尾 ms", "ビブ Hz", "ビブ ±c", "ビブ開始", "HNR", "重心 Hz", "コード", "役割", "イベント"]

    def __init__(self, parent=None) -> None:
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)
        self.cellClicked.connect(self._clicked)

    def set_result(self, res: AnalysisResult) -> None:
        self.setSortingEnabled(False)
        self.setRowCount(len(res.notes))
        off = res.offset_s

        def num(v, fmt="{:.0f}"):
            return _NumItem("-" if v is None or not np.isfinite(v) else fmt.format(v), v)

        for r, n in enumerate(res.notes):
            f = n.features
            items = [
                num(f["index"]), _NumItem(fmt_time(f["start_s"] + off, 2), f["start_s"]), num(f["dur_ms"]),
                _NumItem(f["name"], n.midi), num(f["intonation_cents"], "{:+.0f}"),
                QTableWidgetItem(f["approach_label"]), num(f["attack_depth_cents"], "{:+.0f}"), num(f["attack_ms"]),
                num(f["overshoot_cents"]), QTableWidgetItem(f["release_label"]),
                num(f["release_depth_cents"], "{:+.0f}"), num(f["release_ms"]),
                num(f["vib_rate_hz"] if f["vib"] else np.nan, "{:.2f}"),
                num(f["vib_extent_cents"] if f["vib"] else np.nan), num(f["vib_onset_ms"] if f["vib"] else np.nan),
                num(f["hnr_db"], "{:.1f}"), num(f["centroid_hz"]),
                QTableWidgetItem(f.get("chord") or "-"), QTableWidgetItem(f.get("chord_role") or "-"),
                QTableWidgetItem(", ".join(e["label"] for e in f.get("events", []))),
            ]
            if f.get("chord_tone") is False:
                items[18].setForeground(pg.mkColor("#feca57"))
            items[5].setForeground(pg.mkColor(APPROACH_COLORS[f["approach"]]))
            items[9].setForeground(pg.mkColor(RELEASE_COLORS[f["release"]]))
            for c, it in enumerate(items):
                it.setData(Qt.UserRole, r)
                self.setItem(r, c, it)
        self.setSortingEnabled(True)

    def select_note(self, i: int) -> None:
        for r in range(self.rowCount()):
            it = self.item(r, 0)
            if it is not None and it.data(Qt.UserRole) == i:
                self.blockSignals(True)
                self.selectRow(r)
                self.scrollToItem(it)
                self.blockSignals(False)
                return

    def _clicked(self, r: int, _c: int) -> None:
        it = self.item(r, 0)
        if it is not None:
            self.noteSelected.emit(int(it.data(Qt.UserRole)))


class SpectrogramView(pg.GraphicsLayoutWidget):
    """区間全体のボーカルのスペクトログラム + f0 と倍音の軌跡."""

    seekRequested = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.plot = self.addPlot(axisItems={"bottom": TimeAxis("bottom")})
        self.plot.setLabel("left", "周波数", units="Hz")
        self.plot.setMenuEnabled(False)
        self.plot.setTitle("<span style='color:#7f8796'>ホイール: 拡大縮小 (軸の上でその軸だけ) ・ ドラッグ: 移動 ・ "
                           "右ドラッグ: 拡大縮小 ・ 左下の A: 全体に戻す ・ クリック: その位置へ移動</span>", size="9pt")
        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get("magma"))
        self.plot.addItem(self.img)
        self.cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#feca57", width=2))
        self.plot.addItem(self.cursor, ignoreBounds=True)
        self.harm_items: list = []
        self._res_id = None
        self.scene().sigMouseClicked.connect(self._clicked)

    def set_result(self, res: AnalysisResult) -> None:
        if self._res_id == id(res):
            return
        self._res_id = id(res)
        y = res.vocal_analysis_signal()
        sr = config.ANALYSIS_SR
        win, nfft, hop = 1024, 2048, config.HOP
        w = np.hanning(win)
        n = 1 + len(y) // hop
        yp = np.pad(y, (win // 2, win // 2 + hop))
        fmax = 5000.0
        kmax = int(fmax / (sr / nfft))
        D = np.empty((n, kmax), dtype=np.float32)
        for b0 in range(0, n, 2048):
            b1 = min(n, b0 + 2048)
            idx = (np.arange(b0, b1) * hop)[:, None] + np.arange(win)[None, :]
            D[b0:b1] = 20 * np.log10(np.abs(np.fft.rfft(yp[idx] * w, nfft, axis=1))[:, :kmax] + 1e-6)
        D -= D.max()
        self.img.setImage(np.clip(D, -80, 0), autoLevels=False, levels=(-80, 0))
        off = res.offset_s
        self.img.setRect(QRectF(off, 0, n * config.HOP_SEC, fmax))
        for it in self.harm_items:
            self.plot.removeItem(it)
        self.harm_items = []
        t = res.times + off
        for k in range(1, 6):
            it = pg.PlotDataItem(t, res.f0_hz * k, pen=pg.mkPen(0, 255, 200, 180 if k == 1 else 70, width=1.3),
                                 connect="finite")
            self.plot.addItem(it)
            self.harm_items.append(it)
        dur = n * config.HOP_SEC
        # マウスで拡大縮小しても曲の外 / 0Hz 未満へは行かないように (最大で 0.1 秒・50Hz 幅まで拡大)
        self.plot.setLimits(xMin=off, xMax=off + dur, yMin=0, yMax=fmax, minXRange=0.1, minYRange=50)
        self.plot.setXRange(off, off + dur, padding=0)
        self.plot.setYRange(0, fmax, padding=0)

    def set_cursor(self, t_abs: float) -> None:
        self.cursor.setValue(t_abs)

    def _clicked(self, ev) -> None:
        if ev.button() == Qt.LeftButton and not ev.isAccepted():
            vb = self.plot.getViewBox()
            if vb.sceneBoundingRect().contains(ev.scenePos()):
                self.seekRequested.emit(float(vb.mapSceneToView(ev.scenePos()).x()))
