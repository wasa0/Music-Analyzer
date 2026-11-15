"""カラオケ風の音程バー (ピアノロール) と VOCALOID 風コントロールレーン."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPicture  # noqa: F401

from ..analysis import AnalysisResult
from ..features import APPROACH_LABELS, EVENT_LABELS, RELEASE_LABELS
from ..pitch import NOTE_NAMES, midi_to_hz, midi_to_name
from .common import APPROACH_COLORS, EVENT_COLORS, LANE_COLORS, RELEASE_COLORS, fmt_time
from .source_panel import TimeAxis

BLACK_KEYS = {1, 3, 6, 8, 10}


class NoteAxis(pg.AxisItem):
    """縦軸に音名 (C4 など) を表示."""

    def tickValues(self, minVal, maxVal, size):  # noqa: N802
        span = maxVal - minVal
        lo, hi = int(np.floor(minVal)), int(np.ceil(maxVal))
        if span > 30:
            major = [m for m in range(lo, hi + 1) if m % 12 == 0]
            minor = [m for m in range(lo, hi + 1) if m % 12 in (4, 7)]
        else:
            major = [m for m in range(lo, hi + 1) if m % 12 == 0]
            minor = [m for m in range(lo, hi + 1) if m % 12 != 0]
        return [(12, major), (1, minor)]

    def tickStrings(self, values, scale, spacing):  # noqa: N802
        return [f"{NOTE_NAMES[int(v) % 12]}{int(v) // 12 - 1}" for v in values]


class KeyboardBackground(pg.GraphicsObject):
    """黒鍵の段を暗く塗り、C の境目に線を引く背景."""

    def __init__(self) -> None:
        super().__init__()
        self.setZValue(-100)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(-1e5, 0, 2e5, 128)

    def paint(self, p: QPainter, *args) -> None:
        vb = self.getViewBox()
        if vb is None:
            return
        (x0, x1), (y0, y1) = vb.viewRange()
        dark = QColor("#111318")
        light = QColor("#1a1d24")
        pen_c = QPen(QColor("#3a4050"))
        pen_c.setCosmetic(True)
        pen_e = QPen(QColor("#232731"))
        pen_e.setCosmetic(True)
        for m in range(max(0, int(np.floor(y0)) - 1), min(127, int(np.ceil(y1)) + 1) + 1):
            r = QRectF(x0, m - 0.5, x1 - x0, 1.0)
            p.fillRect(r, dark if (m % 12) in BLACK_KEYS else light)
            if m % 12 == 0:
                p.setPen(pen_c)
                p.drawLine(QPointF(x0, m - 0.5), QPointF(x1, m - 0.5))
            elif m % 12 == 5:
                p.setPen(pen_e)
                p.drawLine(QPointF(x0, m - 0.5), QPointF(x1, m - 0.5))


class NoteBarsItem(pg.GraphicsObject):
    """音程バー. 色 = 入り方, 下端の帯 = 語尾の処理, 波線 = ビブラート区間."""

    sigNoteClicked = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptHoverEvents(True)
        self.notes: list[dict] = []
        self.selected = -1
        self._pic = QPicture()
        self._rect = QRectF()

    def set_notes(self, notes: list[dict]) -> None:
        self.prepareGeometryChange()
        self.notes = notes
        self.selected = -1
        self._build()
        self.update()

    def _build(self) -> None:
        self._pic = QPicture()
        p = QPainter(self._pic)
        p.setRenderHint(QPainter.Antialiasing)
        xs, ys = [], []
        for n in self.notes:
            x0, x1, m = n["x0"], n["x1"], n["midi"]
            xs += [x0, x1]
            ys += [m - 0.5, m + 0.5]
            col = QColor(APPROACH_COLORS.get(n["approach"], "#1dd1a1"))
            fill = QColor(col)
            fill.setAlpha(150)
            pen = QPen(col.lighter(130))
            pen.setCosmetic(True)
            pen.setWidthF(1.2)
            p.setPen(pen)
            p.setBrush(QBrush(fill))
            p.drawRect(QRectF(x0, m - 0.40, x1 - x0, 0.80))
            # アタック部分 (目標音に落ち着くまで) を濃く
            if n["settle_x"] > x0:
                c2 = QColor(col.darker(160))
                c2.setAlpha(200)
                p.setPen(Qt.NoPen)
                p.setBrush(c2)
                p.drawRect(QRectF(x0, m - 0.40, n["settle_x"] - x0, 0.80))
            # 語尾の処理を下端の帯で表示
            if n["release"] != "flat" and n["release_x"] < x1:
                rc = QColor(RELEASE_COLORS.get(n["release"], "#8395a7"))
                p.setPen(Qt.NoPen)
                p.setBrush(rc)
                p.drawRect(QRectF(n["release_x"], m - 0.40, x1 - n["release_x"], 0.22))
            # ビブラート区間を波線で表示
            if n["vib"] and np.isfinite(n["vib_x"]) and np.isfinite(n["vib_rate"]):
                pen_v = QPen(QColor(255, 255, 255, 200))
                pen_v.setCosmetic(True)
                pen_v.setWidthF(1.3)
                p.setPen(pen_v)
                p.setBrush(Qt.NoBrush)
                path = QPainterPath()
                xv = np.arange(n["vib_x"], n["release_x"], 1.0 / (n["vib_rate"] * 12))
                if xv.size > 2:
                    yv = m + 0.18 * np.sin(2 * np.pi * n["vib_rate"] * (xv - xv[0]))
                    path.moveTo(xv[0], yv[0])
                    for a, b in zip(xv[1:], yv[1:]):
                        path.lineTo(a, b)
                    p.drawPath(path)
        p.end()
        if xs:
            self._rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        else:
            self._rect = QRectF()

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self._rect

    def paint(self, p: QPainter, *args) -> None:
        p.drawPicture(0, 0, self._pic)
        if 0 <= self.selected < len(self.notes):
            n = self.notes[self.selected]
            pen = QPen(QColor("#ffffff"))
            pen.setCosmetic(True)
            pen.setWidthF(2.5)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(n["x0"], n["midi"] - 0.48, n["x1"] - n["x0"], 0.96))

    def set_selected(self, i: int) -> None:
        self.selected = i
        self.update()

    def note_at(self, pos: QPointF) -> int:
        for i, n in enumerate(self.notes):
            if n["x0"] <= pos.x() <= n["x1"] and abs(pos.y() - n["midi"]) <= 0.5:
                return i
        return -1

    def mouseClickEvent(self, ev) -> None:  # noqa: N802
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return
        i = self.note_at(ev.pos())
        if i >= 0:
            ev.accept()
            self.sigNoteClicked.emit(i)
        else:
            ev.ignore()

    def hoverEvent(self, ev) -> None:  # noqa: N802
        if ev.isExit():
            self.setToolTip("")
            return
        i = self.note_at(ev.pos())
        self.setToolTip(self.notes[i]["tip"] if i >= 0 else "")


def chord_color(root: int | None, alpha: int = 255) -> QColor:
    """コードの根音 → 色 (ビジュアライザーと同じ「音名 = 色相」の対応)."""
    if root is None:
        c = QColor(60, 64, 76)
    else:
        c = QColor.fromHsvF((root % 12) / 12.0, 0.55, 0.62)
    c.setAlpha(alpha)
    return c


class ChordLaneItem(pg.GraphicsObject):
    """コード名 (または度数) を時間軸上の箱で表示する. 文字は画面のピクセル単位で描く."""

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptHoverEvents(True)
        self.chords: list[dict] = []
        self.beats: list[float] = []
        self.offset = 0.0
        self.mode = "name"
        self._rect = QRectF()

    def set_data(self, chords: list[dict], beats: list[float], offset: float) -> None:
        self.prepareGeometryChange()
        self.chords, self.beats, self.offset = chords, beats, offset
        if chords:
            self._rect = QRectF(chords[0]["start"] + offset, 0, chords[-1]["end"] - chords[0]["start"], 1)
        else:
            self._rect = QRectF()
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self._rect

    def paint(self, p: QPainter, *args) -> None:
        vb = self.getViewBox()
        if vb is None or not self.chords:
            return
        (x0, x1), _ = vb.viewRange()
        tr = p.transform()
        p.resetTransform()
        big = QFont(p.font())
        big.setPointSizeF(10)
        big.setBold(True)
        small = QFont(big)
        small.setPointSizeF(7.5)
        fm_big, fm_small = QFontMetricsF(big), QFontMetricsF(small)
        view = tr.mapRect(QRectF(x0, 0.0, x1 - x0, 1.0))
        for c in self.chords:
            a, b = c["start"] + self.offset, c["end"] + self.offset
            if b < x0 or a > x1:
                continue
            box = tr.mapRect(QRectF(a, 0.06, b - a, 0.88))
            p.setPen(QPen(QColor(20, 22, 28), 1))
            p.setBrush(chord_color(c.get("root"), 235))
            p.drawRoundedRect(box.adjusted(1, 0, -1, 0), 4, 4)
            # 画面の端で切れている箱は、見えている部分の中央に文字を置く
            r = QRectF(max(box.left(), view.left()), box.top(),
                       min(box.right(), view.right()) - max(box.left(), view.left()), box.height())
            label = c["degree"] if self.mode == "degree" and c.get("root") is not None else c["label"]
            # 収まらないときは 小さい文字 → 分数部分 (/Bb) を省略 の順に試す。それでも入らなければ出さない
            # (途中で切ると別のコード名に見えてしまうため。ツールチップで確認できる)
            short = label.split("/")[0]
            for font, fm, cand in ((big, fm_big, label), (small, fm_small, label),
                                   (big, fm_big, short), (small, fm_small, short)):
                if fm.horizontalAdvance(cand) + 4 <= r.width():
                    p.setFont(font)
                    p.setPen(QColor(255, 255, 255))
                    p.drawText(r, Qt.AlignCenter, cand)
                    break
        # 拍の目印
        pen = QPen(QColor(255, 255, 255, 90), 1)
        p.setPen(pen)
        for bt in self.beats:
            x = bt + self.offset
            if x0 <= x <= x1:
                pt = tr.map(QPointF(x, 0.0))
                p.drawLine(QPointF(pt.x(), pt.y()), QPointF(pt.x(), pt.y() - 5))
        p.setTransform(tr)

    def chord_at_x(self, x: float) -> dict | None:
        for c in self.chords:
            if c["start"] + self.offset <= x < c["end"] + self.offset:
                return c
        return None

    def hoverEvent(self, ev) -> None:  # noqa: N802
        if ev.isExit():
            self.setToolTip("")
            return
        c = self.chord_at_x(ev.pos().x())
        if c is None:
            self.setToolTip("")
            return
        self.setToolTip(f"{c['label']}  ({c['degree']})\n{fmt_time(c['start'] + self.offset, 2)} 〜 "
                        f"{fmt_time(c['end'] + self.offset, 2)}\n一致度 {c.get('score') or 0:.2f}")


class BeatGridItem(pg.GraphicsObject):
    """拍の縦線 (4 拍ごとに少し濃く)."""

    def __init__(self) -> None:
        super().__init__()
        self.beats: list[float] = []
        self.setZValue(-50)

    def set_beats(self, beats_abs: list[float]) -> None:
        self.prepareGeometryChange()
        self.beats = list(beats_abs)
        self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802
        if not self.beats:
            return QRectF()
        return QRectF(self.beats[0], 0, self.beats[-1] - self.beats[0] + 1e-3, 128)

    def paint(self, p: QPainter, *args) -> None:
        vb = self.getViewBox()
        if vb is None or not self.beats:
            return
        (x0, x1), (y0, y1) = vb.viewRange()
        weak = QPen(QColor(255, 255, 255, 18))
        weak.setCosmetic(True)
        strong = QPen(QColor(255, 255, 255, 45))
        strong.setCosmetic(True)
        for i, b in enumerate(self.beats):
            if x0 <= b <= x1:
                p.setPen(strong if i % 4 == 0 else weak)
                p.drawLine(QPointF(b, y0), QPointF(b, y1))


def note_tooltip(f: dict, offset: float) -> str:
    lines = [f"#{f['index']}  {f['name']}  ({fmt_time(f['start_s'] + offset, 2)}, {f['dur_ms']:.0f} ms)",
             f"音程ずれ: {f['intonation_cents']:+.0f} cent",
             f"入り: {f['approach_label']}  {f['attack_depth_cents']:+.0f} cent / {f['attack_ms']:.0f} ms"]
    if f.get("overshoot_cents"):
        lines.append(f"オーバーシュート: {f['overshoot_cents']:.0f} cent")
    if f.get("vib"):
        lines.append(f"ビブラート: {f['vib_rate_hz']:.1f} Hz ±{f['vib_extent_cents']:.0f} cent "
                     f"(開始 {f['vib_onset_ms']:.0f} ms)")
    lines.append(f"語尾: {f['release_label']}  {f['release_depth_cents']:+.0f} cent / {f['release_ms']:.0f} ms")
    if f.get("chord") and f.get("chord") != "-":
        tone = "コードトーン" if f.get("chord_tone") else "テンション/経過音" if f.get("chord_tone") is False else ""
        lines.append(f"コード: {f['chord']} ({f.get('chord_degree', '-')}) の {f.get('chord_role', '-')}  {tone}")
    for e in f.get("events", []):
        lines.append(f"★ {e['label']} {e['cents']:+.0f} cent ({e['dur_ms']:.0f} ms)")
    lines.append("クリックで詳細 (FFT) を表示")
    return "\n".join(lines)


class PianoRollView(pg.GraphicsLayoutWidget):
    noteClicked = Signal(int)
    seekRequested = Signal(float)   # 絶対時刻 (元音源基準)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.res: AnalysisResult | None = None
        self.offset = 0.0
        self.ci.layout.setSpacing(2)

        # コード表示レーン (最上段)
        self.chord_plot = self.addPlot(row=0, col=0)
        self.chord_plot.setMenuEnabled(False)
        self.chord_plot.setMouseEnabled(x=True, y=False)
        self.chord_plot.hideAxis("bottom")
        self.chord_plot.getAxis("left").setTicks([[]])
        self.chord_plot.setLabel("left", "コード")
        self.chord_plot.setYRange(0, 1, padding=0)
        self.chord_plot.setMaximumHeight(46)
        self.chord_plot.setMinimumHeight(40)
        self.chord_lane = ChordLaneItem()
        self.chord_plot.addItem(self.chord_lane)
        self.chord_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#feca57", width=2))
        self.chord_plot.addItem(self.chord_cursor, ignoreBounds=True)

        self.roll = self.addPlot(row=1, col=0, axisItems={"left": NoteAxis("left"), "bottom": TimeAxis("bottom")})
        self.chord_plot.setXLink(self.roll)
        self.roll.setLabel("left", "音程")
        self.roll.showGrid(x=True, y=False, alpha=0.15)
        self.roll.setMenuEnabled(False)
        self.roll.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.bg = KeyboardBackground()
        self.roll.addItem(self.bg, ignoreBounds=True)
        self.beat_grid = BeatGridItem()
        self.roll.addItem(self.beat_grid, ignoreBounds=True)
        self.bars = NoteBarsItem()
        self.bars.sigNoteClicked.connect(self._on_note)
        self.roll.addItem(self.bars)
        self.curve = pg.PlotDataItem(pen=pg.mkPen("#f5f6fa", width=1.8), connect="finite")
        self.curve.setZValue(20)
        self.roll.addItem(self.curve)
        self.events = pg.ScatterPlotItem(hoverable=True, tip=self._event_tip, hoverSize=16)
        self.events.setZValue(30)
        self.roll.addItem(self.events)
        self.cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#feca57", width=2))
        self.cursor.setZValue(50)
        self.roll.addItem(self.cursor, ignoreBounds=True)

        self.pit = self.addPlot(row=2, col=0, axisItems={"bottom": TimeAxis("bottom")})
        self.pit.setLabel("left", "PIT (cent)")
        self.pit.setXLink(self.roll)
        self.pit.setMouseEnabled(x=True, y=False)
        self.pit.setMenuEnabled(False)
        self.pit.showGrid(x=True, y=True, alpha=0.15)
        self.pit.addItem(pg.InfiniteLine(angle=0, pos=0, pen=pg.mkPen("#576075", width=1)))
        self.pit_curve = pg.PlotDataItem(pen=pg.mkPen(LANE_COLORS["PIT"], width=1.5), connect="finite")
        self.pit.addItem(self.pit_curve)
        self.pit_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#feca57", width=1))
        self.pit.addItem(self.pit_cursor, ignoreBounds=True)
        self.pit.setYRange(-300, 300)

        self.ctl = self.addPlot(row=3, col=0, axisItems={"bottom": TimeAxis("bottom")})
        self.ctl.setLabel("left", "DYN/BRE/BRI")
        self.ctl.setXLink(self.roll)
        self.ctl.setMouseEnabled(x=True, y=False)
        self.ctl.setMenuEnabled(False)
        self.ctl.setYRange(0, 127)
        self.ctl.showGrid(x=True, y=True, alpha=0.12)
        leg = self.ctl.addLegend(offset=(6, 4), labelTextColor="#c8ccd4", brush=pg.mkBrush(20, 22, 28, 200))
        self.lane_items = {}
        for k in ("DYN", "BRE", "BRI"):
            it = pg.PlotDataItem(pen=pg.mkPen(LANE_COLORS[k], width=1.4), name=k, connect="finite")
            self.ctl.addItem(it)
            self.lane_items[k] = it
        leg.setLabelTextSize("9pt")
        self.ctl_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#feca57", width=1))
        self.ctl.addItem(self.ctl_cursor, ignoreBounds=True)

        lay = self.ci.layout
        lay.setRowStretchFactor(0, 0)
        lay.setRowStretchFactor(1, 6)
        lay.setRowStretchFactor(2, 2)
        lay.setRowStretchFactor(3, 2)
        for p in (self.chord_plot, self.roll, self.pit, self.ctl):
            p.getAxis("left").setWidth(64)
        self.scene().sigMouseClicked.connect(self._scene_clicked)

    def set_chord_mode(self, mode: str) -> None:
        """'name' = コード名 (Cm7), 'degree' = 度数 (VIm7)."""
        self.chord_lane.set_mode(mode)

    # ------------------------------------------------------------------ データ
    def set_result(self, res: AnalysisResult) -> None:
        self.res = res
        self.offset = off = res.offset_s
        t = res.times + off
        self.curve.setData(t, res.midi_raw)
        notes = []
        pit = np.full(len(t), np.nan)
        spots = []
        for n in res.notes:
            f = n.features
            x0, x1 = f["start_s"] + off, f["end_s"] + off
            settle_x = x0 + f["settle_idx"] * 0.01
            rel_x = x0 + f["release_idx"] * 0.01
            vib_x = x0 + f["vib_onset_ms"] / 1000 if f.get("vib") and np.isfinite(f.get("vib_onset_ms", np.nan)) else np.nan
            notes.append({"x0": x0, "x1": x1, "midi": n.midi, "approach": f["approach"],
                          "release": f["release"], "settle_x": settle_x, "release_x": rel_x,
                          "vib": f.get("vib", False), "vib_x": vib_x, "vib_rate": f.get("vib_rate_hz", np.nan),
                          "tip": note_tooltip(f, off)})
            pit[n.start:n.end] = (res.midi_raw[n.start:n.end] - n.midi) * 100.0
            for e in f.get("events", []):
                ti = int(round(e["time_s"] * 100))
                y = res.midi_raw[min(ti, len(t) - 1)]
                if not np.isfinite(y):
                    y = n.midi
                sym = "t1" if e["cents"] > 0 else "t"
                spots.append({"pos": (e["time_s"] + off, y + (0.6 if e["cents"] > 0 else -0.6)),
                              "symbol": "star" if e["type"] == "flip" else sym, "size": 11,
                              "brush": pg.mkBrush(EVENT_COLORS[e["type"]]), "pen": pg.mkPen("#000000", width=0.5),
                              "data": f"{e['label']}  {e['cents']:+.0f} cent / {e['dur_ms']:.0f} ms\n"
                                      f"{fmt_time(e['time_s'] + off, 2)} (#{f['index']} {f['name']})"})
        self.bars.set_notes(notes)
        self.events.setData(spots)
        self.chord_lane.set_data(res.chords, res.harmony.get("beats", []) if res.harmony else [], off)
        self.beat_grid.set_beats([b + off for b in (res.harmony.get("beats", []) if res.harmony else [])])
        self.chord_plot.setVisible(bool(res.chords))
        self.pit_curve.setData(t, pit)
        for k, it in self.lane_items.items():
            it.setData(t, res.lanes[k])
        valid = res.midi_raw[np.isfinite(res.midi_raw)]
        lo, hi = (np.percentile(valid, 1) - 3, np.percentile(valid, 99) + 3) if valid.size else (55, 80)
        self.roll.setXRange(off, off + min(12.0, t[-1] - off + 0.1 if len(t) else 12), padding=0)
        self.roll.setYRange(lo, hi, padding=0)
        self.roll.setLimits(xMin=off - 1, xMax=t[-1] + 1 if len(t) else off + 10, yMin=0, yMax=128)
        self.set_cursor(off)

    def _event_tip(self, x, y, data):
        return data

    # ------------------------------------------------------------------ 操作
    def set_cursor(self, t_abs: float, follow: bool = False) -> None:
        for c in (self.cursor, self.pit_cursor, self.ctl_cursor, self.chord_cursor):
            c.setValue(t_abs)
        if follow:
            (x0, x1), _ = self.roll.viewRange()
            w = x1 - x0
            if t_abs > x1 - w * 0.15 or t_abs < x0:
                self.roll.setXRange(t_abs - w * 0.15, t_abs + w * 0.85, padding=0)

    def select_note(self, i: int, center: bool = True) -> None:
        self.bars.set_selected(i)
        if center and self.res and 0 <= i < len(self.res.notes):
            f = self.res.notes[i].features
            (x0, x1), (y0, y1) = self.roll.viewRange()
            a, b = f["start_s"] + self.offset, f["end_s"] + self.offset
            if a < x0 or b > x1:
                w = x1 - x0
                self.roll.setXRange((a + b) / 2 - w / 2, (a + b) / 2 + w / 2, padding=0)
            m = self.res.notes[i].midi
            if m < y0 + 1 or m > y1 - 1:
                h = y1 - y0
                self.roll.setYRange(m - h / 2, m + h / 2, padding=0)

    def _on_note(self, i: int) -> None:
        self.select_note(i, center=False)
        self.noteClicked.emit(i)

    def _scene_clicked(self, ev) -> None:
        if ev.button() != Qt.LeftButton or ev.isAccepted() or ev.double():
            return
        for p in (self.chord_plot, self.roll, self.pit, self.ctl):
            vb = p.getViewBox()
            if vb.sceneBoundingRect().contains(ev.scenePos()):
                x = vb.mapSceneToView(ev.scenePos()).x()
                self.seekRequested.emit(float(x))
                return


__all__ = ["PianoRollView", "APPROACH_LABELS", "RELEASE_LABELS", "EVENT_LABELS", "midi_to_hz", "midi_to_name"]
