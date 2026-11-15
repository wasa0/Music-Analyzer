"""ビジュアライザーの描画パネル (QPainter).

色は「音名 → 色相」で統一している (C=赤 … B=紫)。同じ音名 (オクターブ違い) は同じ色なので、
倍音のうち 2・4・8 倍音は基音と同色、3・6 倍音は 5 度上、5 倍音は長 3 度上の色になる。
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QImage, QLinearGradient, QPainter, QPainterPath,
                           QPen, QPolygonF, QRadialGradient)
from PySide6.QtWidgets import QSizePolicy, QWidget

from pyqtgraph.functions import arrayToQPath, create_qpolygonf, ndarray_from_qpolygonf
from PySide6.QtCore import QLineF

from ..gui.common import fmt_time
from ..pitch import NOTE_NAMES
from .dsp import SPEC_FMAX, SPEC_FMIN, SPEC_RANGE_DB, Track, VisFrame, pitch_class_hue

BG = QColor(8, 9, 13)


def hue_color(h: float, s: float = 0.85, v: float = 1.0, a: float = 1.0) -> QColor:
    c = QColor.fromHsvF(h % 1.0, s, v)
    c.setAlphaF(max(0.0, min(1.0, a)))
    return c


def note_label(freq: float) -> str:
    if not np.isfinite(freq) or freq <= 0:
        return "-"
    m = 69 + 12 * np.log2(freq / 440.0)
    n = int(round(m))
    cents = (m - n) * 100
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1} {cents:+.0f}c"


def _poly(xs: np.ndarray, ys: np.ndarray) -> QPolygonF:
    """numpy 配列から QPolygonF を高速生成 (QPointF を 1 個ずつ作らずメモリに直接書き込む)."""
    poly = create_qpolygonf(len(xs))
    arr = ndarray_from_qpolygonf(poly)
    arr[:, 0] = xs
    arr[:, 1] = ys
    return poly


def _decimate(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) <= n:
        return x
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx]


def glow_line(p: QPainter, shape, color: QColor, glow: tuple[float, ...] = (8.0,), glow_alpha: float = 0.18,
              core_alpha: float = 1.0, underlay: bool = True) -> None:
    """発光する線. Qt のラスタ描画は「アンチエイリアス + 太線」が非常に重いので、
    ぼかし役の太線はアンチエイリアス無し、芯だけアンチエイリアス付き 1px で描く."""
    draw = p.drawPath if isinstance(shape, QPainterPath) else p.drawPolyline
    p.setBrush(Qt.NoBrush)
    p.setRenderHint(QPainter.Antialiasing, False)
    for w in glow:
        c = QColor(color)
        c.setAlphaF(max(0.0, min(1.0, glow_alpha)))
        p.setPen(QPen(c, w))
        draw(shape)
    c = QColor(color)
    if underlay:
        c.setAlphaF(max(0.0, min(1.0, core_alpha * 0.55)))
        p.setPen(QPen(c, 2.0))
        draw(shape)
    p.setRenderHint(QPainter.Antialiasing, True)
    c.setAlphaF(max(0.0, min(1.0, core_alpha)))
    p.setPen(QPen(c, 1.0))
    draw(shape)


def _use_gl() -> bool:
    """GPU (OpenGL) で描画するか.

    MUSICA_GL=0 / オフスクリーン環境 / OpenGL 2.0 以上のコンテキストが作れない環境
    (リモートデスクトップ、仮想マシン、古いドライバ等) ではソフトウェア描画にする。
    """
    if os.environ.get("MUSICA_GL", "1") == "0":
        return False
    if "offscreen" in os.environ.get("QT_QPA_PLATFORM", ""):
        return False
    try:
        from PySide6.QtGui import QGuiApplication, QOffscreenSurface, QOpenGLContext

        if QGuiApplication.instance() is None:
            return True                      # 判定できない (アプリ作成前): 既定は GPU
        ctx = QOpenGLContext()
        if not ctx.create():
            return False
        surf = QOffscreenSurface()
        surf.setFormat(ctx.format())
        surf.create()
        ok = surf.isValid() and ctx.makeCurrent(surf)
        if ok:
            ver = ctx.format().version()
            renderer = ""
            try:
                r = ctx.functions().glGetString(0x1F01)        # GL_RENDERER
                renderer = r.decode(errors="ignore") if isinstance(r, (bytes, bytearray)) else str(r or "")
            except Exception:
                pass
            ctx.doneCurrent()
            ok = ver >= (2, 0)
            # ソフトウェア実装の OpenGL (Mesa llvmpipe, Windows の GDI Generic / Basic Render 等) は
            # Qt のラスタ描画より遅いので使わない
            soft = ("llvmpipe", "softpipe", "swiftshader", "gdi generic", "basic render", "software")
            if any(s in renderer.lower() for s in soft):
                ok = False
            global GL_RENDERER
            GL_RENDERER = renderer
        surf.destroy()
        return bool(ok)
    except Exception:
        return False


GL_RENDERER = ""


USE_GL = _use_gl()
if USE_GL:
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtOpenGLWidgets import QOpenGLWidget as _PanelBase
else:
    _PanelBase = QWidget


class Panel(_PanelBase):
    """描画パネルの基底. 各サブクラスは draw_panel() を実装する.

    GPU モード (QOpenGLWidget + 4x MSAA) では QPainter の線描画が GPU で行われるため、
    HiDPI (125% など) の画面でもソフトウェア描画より大幅に速い。
    """

    title = ""
    # FrameAnalyzer.analyze() の detail: 0 = 時刻と f0 だけ / 1 = + スペクトラム・クロマ / 2 = + 倍音・波形
    detail = 2

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.frame: VisFrame | None = None
        self.tag = ""                       # タイトルの前に出す表示パート名 (ボーカル など)
        self.tag_color = QColor(200, 200, 200)
        if USE_GL:
            fmt = QSurfaceFormat()
            fmt.setSamples(4)
            self.setFormat(fmt)
        else:
            self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(160, 100)
        self.title_font = QFont(self.font())
        self.title_font.setPointSizeF(9.5)
        self.big_font = QFont(self.font())
        self.big_font.setPointSizeF(20)
        self.big_font.setBold(True)

    def set_frame(self, f: VisFrame) -> None:
        self.frame = f
        self.update()

    def set_tag(self, text: str, color: str | QColor | None = None) -> None:
        """どのパートを表示しているか (タイトルの左に色付きで表示). 空文字で消す."""
        self.tag = text
        if color is not None:
            self.tag_color = QColor(color)
        self.update()

    def draw_panel(self) -> None:  # サブクラスで実装
        p = self.begin()
        self.draw_title(p)
        p.end()

    if USE_GL:
        def paintGL(self) -> None:  # noqa: N802
            self.draw_panel()
    else:
        def paintEvent(self, _ev) -> None:  # noqa: N802
            self.draw_panel()

    def begin(self) -> QPainter:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)
        return p

    def draw_title(self, p: QPainter, extra: str = "", hint: str | tuple[str, ...] = "") -> None:
        p.setFont(self.title_font)
        fm = QFontMetricsF(self.title_font)
        x = 10.0
        if self.tag:
            r = QRectF(7, 5, fm.horizontalAdvance(self.tag) + 12, 17)
            p.setPen(Qt.NoPen)
            p.setBrush(self.tag_color)
            p.drawRoundedRect(r, 4, 4)
            p.setPen(QColor(16, 17, 22))
            p.drawText(r, Qt.AlignCenter, self.tag)
            x = r.right() + 8
        text = self.title + ("   " + extra if extra else "")
        p.setPen(QColor(150, 160, 180))
        p.drawText(QPointF(x, 18), text)
        # 操作のヒントは右端に薄く. 入りきる最初の候補を使い、どれもタイトルと重なるなら出さない
        for h in ((hint,) if isinstance(hint, str) else hint):
            hw = fm.horizontalAdvance(h)
            if h and x + fm.horizontalAdvance(text) + 24 + hw < self.width() - 8:
                p.setPen(QColor(100, 108, 126))
                p.drawText(QRectF(self.width() - 8 - hw, 4, hw, 18), Qt.AlignRight | Qt.AlignVCenter, h)
                break
        pen = QPen(QColor(40, 44, 56))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))


# ======================================================================== ハーモニック・マンダラ
class MandalaView(Panel):
    """円形スペクトラム + 倍音リング (k 倍音 = k 枚の花びら) + 倍音から再合成した 1 周期波形."""

    title = "ハーモニック・マンダラ"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.rot = 0.0
        self.theta = np.linspace(0, 2 * np.pi, 181)
        self.hue = 0.0
        self._spoke_key = None
        self._spoke_ang = np.zeros(0)
        self._spoke_pens: list = []

    def draw_panel(self) -> None:
        p = self.begin()
        f = self.frame
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2 + 8
        R = min(w, h - 20) / 2 * 0.95
        if f is None:
            self.draw_title(p)
            p.end()
            return
        self.rot += 0.004 + 0.02 * max(0.0, (f.level_db + 60) / 60)
        if f.voiced:
            target = pitch_class_hue(f.f0)
            d = (target - self.hue + 0.5) % 1.0 - 0.5
            self.hue = (self.hue + d * 0.15) % 1.0
        energy = float(np.clip((f.level_db + 55) / 50, 0, 1))

        # 背景のグロー
        g = QRadialGradient(QPointF(cx, cy), R * 1.1)
        g.setColorAt(0.0, hue_color(self.hue, 0.9, 0.55, 0.35 * energy + 0.05))
        g.setColorAt(0.55, hue_color(self.hue + 0.08, 0.9, 0.35, 0.12 * energy))
        g.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(g))
        p.drawEllipse(QPointF(cx, cy), R * 1.1, R * 1.1)

        # 円形スペクトラム (左右対称). ペンは (帯域, 明るさ段階) ごとにキャッシュして drawLines でまとめ描き
        bands = f.bands
        nb = len(bands)
        r0 = R * 0.66
        if self._spoke_key != (nb, round(R)):
            self._spoke_key = (nb, round(R))
            self._spoke_ang = np.linspace(-np.pi / 2, np.pi * 1.5, 2 * nb, endpoint=False)
            freqs = np.concatenate([f.band_freqs, f.band_freqs[::-1]])
            width = max(1.5, 2 * np.pi * r0 / (2 * nb) * 0.6)
            self._spoke_pens = []
            for fr in freqs:
                hue = pitch_class_hue(fr)
                row = []
                for lvl in range(8):
                    body = QPen(hue_color(hue, 0.8, 1.0, 0.25 + 0.55 * (lvl + 1) / 8), width)
                    body.setCapStyle(Qt.FlatCap)
                    core = QPen(hue_color(hue, 0.45, 1.0, 0.4 + 0.6 * (lvl + 1) / 8), 1.0)
                    row.append((body, core))
                self._spoke_pens.append(row)
        vals = np.concatenate([bands, bands[::-1]])
        ang = self._spoke_ang + self.rot * 0.3
        ca, sa = np.cos(ang), np.sin(ang)
        L = R * 0.32 * vals
        idx = np.nonzero(vals >= 0.02)[0]
        lines = [(i, QLineF(cx + r0 * ca[i], cy + r0 * sa[i], cx + (r0 + L[i]) * ca[i], cy + (r0 + L[i]) * sa[i]))
                 for i in idx]
        # 太い本体はアンチエイリアス無し (高速), 芯の 1px 線だけアンチエイリアス
        for aa, which in ((False, 0), (True, 1)):
            p.setRenderHint(QPainter.Antialiasing, aa)
            for i, ln in lines:
                p.setPen(self._spoke_pens[i][min(7, int(vals[i] * 8))][which])
                p.drawLine(ln)

        # 倍音リング: k 倍音を k 枚の花びらで表現, 色 = その倍音の音名
        # 表示用の強さは振幅の平方根 (高次の弱い倍音も見えるように)
        K = min(12, len(f.harm_amp))
        th = self.theta
        ct, st = np.cos(th), np.sin(th)
        p.setBrush(Qt.NoBrush)
        for k in range(K, 0, -1):
            a = float(np.sqrt(f.harm_amp[k - 1]))
            if a < 0.08:
                continue
            rk = R * (0.22 + 0.034 * k)
            fk = f.f0 * k if np.isfinite(f.f0) else np.nan
            hue = pitch_class_hue(fk) if np.isfinite(fk) else self.hue
            direction = 1 if k % 2 else -1
            r = rk + R * 0.045 * a * np.cos(k * th + f.harm_phase[k - 1] + direction * self.rot * (1 + k * 0.15))
            poly = _poly(cx + r * ct, cy + r * st)
            glow_line(p, poly, hue_color(hue, 0.6, 1.0), glow=(3 + 8 * a,) if a >= 0.35 else (),
                      glow_alpha=0.16 * a, core_alpha=0.3 + 0.7 * a, underlay=a >= 0.6)

        # 中心: 倍音を合成した 1 周期の波形を極座標で閉曲線に
        wave = f.period_wave
        if f.voiced or np.abs(wave).max() > 0.05:
            th2 = np.linspace(0, 2 * np.pi, len(wave), endpoint=False)
            rr = R * (0.13 + 0.075 * wave * max(0.25, energy))
            xs = cx + rr * np.cos(th2 - np.pi / 2)
            ys = cy + rr * np.sin(th2 - np.pi / 2)
            poly = _poly(np.append(xs, xs[0]), np.append(ys, ys[0]))
            glow_line(p, poly, hue_color(self.hue, 0.6, 1.0), glow=(12, 6), glow_alpha=0.14)
            glow_line(p, poly, QColor(255, 255, 255), glow=(), core_alpha=0.95)

        self.draw_title(p, "k倍音 = k枚の花びら / 中心 = 倍音を足し合わせた1周期の波形")
        p.setFont(self.big_font)
        p.setPen(QColor(235, 238, 245))
        if f.voiced:
            p.drawText(QRectF(10, 26, 260, 34), Qt.AlignLeft | Qt.AlignVCenter, note_label(f.f0))
            p.setFont(self.title_font)
            p.setPen(QColor(170, 180, 200))
            p.drawText(QPointF(12, 72), f"f0 = {f.f0:.1f} Hz")
        p.end()


# ======================================================================== 倍音分解
class HarmonicStackView(Panel):
    """各倍音の正弦波 (振幅・位相つき) と、その合計 = 声の波形 / 右に倍音の強さバー."""

    title = "倍音分解: 正弦波の足し算"
    ROWS = 8

    def draw_panel(self) -> None:
        p = self.begin()
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self.draw_title(p)
            p.end()
            return
        left_w = w * 0.64
        top = 26
        rows = self.ROWS + 2
        rh = (h - top - 8) / rows
        x0, x1 = 64, left_w - 10
        xs = np.linspace(x0, x1, 200)
        th = np.linspace(0, 4 * np.pi, 200)   # 2 周期
        p.setFont(self.title_font)
        for k in range(1, self.ROWS + 1):
            a = float(np.sqrt(f.harm_amp[k - 1]))   # 表示は平方根スケール (弱い倍音も見える)
            cy = top + (k - 0.5) * rh
            fk = f.f0 * k if np.isfinite(f.f0) else np.nan
            hue = pitch_class_hue(fk) if np.isfinite(fk) else 0.6
            col = hue_color(hue, 0.7, 1.0, 0.35 + 0.65 * a)
            p.setPen(col if a > 0.1 else QColor(110, 118, 135))
            lab = f"H{k}"
            if np.isfinite(fk):
                m = int(round(69 + 12 * np.log2(fk / 440.0)))
                lab += f" {NOTE_NAMES[m % 12]}{m // 12 - 1}"
            p.drawText(QPointF(6, cy + 4), lab)
            p.setPen(QPen(QColor(35, 38, 48), 1))
            p.drawLine(QPointF(x0, cy), QPointF(x1, cy))
            ys = cy - (rh * 0.45) * a * np.cos(k * th + f.harm_phase[k - 1])
            p.setPen(QPen(col, 1.0))
            p.drawPolyline(_poly(xs, ys))
            p.setPen(QColor(110, 118, 135))
            if f.harm_db[k - 1] > -100:
                p.drawText(QRectF(x1 - 50, cy - rh / 2, 50, rh), Qt.AlignRight | Qt.AlignTop,
                           f"{f.harm_db[k - 1]:.0f}dB")
        # 合計
        cy = top + (self.ROWS + 1) * rh
        p.setPen(QColor(230, 230, 240))
        p.drawText(QPointF(6, cy - rh * 0.35), "Σ 合成")
        p.setPen(QPen(QColor(35, 38, 48), 1))
        p.drawLine(QPointF(x0, cy), QPointF(x1, cy))
        wave = np.tile(f.period_wave, 2)
        xw = np.linspace(x0, x1, len(wave))
        yw = cy - rh * 0.9 * wave
        glow_line(p, _poly(xw, yw), QColor(255, 255, 255), glow=(6,), glow_alpha=0.22)

        # 右: 倍音の強さ (dB)
        bx0, bx1 = left_w + 50, w - 10
        by0, by1 = top + 6, h - 26
        K = len(f.harm_db)
        bw = (bx1 - bx0) / K
        p.setPen(QColor(90, 98, 115))
        for dbl in (0, -20, -40, -60):
            y = by0 + (by1 - by0) * (-dbl / 60)
            p.setPen(QPen(QColor(35, 38, 48), 1))
            p.drawLine(QPointF(bx0, y), QPointF(bx1, y))
            p.setPen(QColor(90, 98, 115))
            p.drawText(QRectF(left_w, y - 8, 46, 16), Qt.AlignRight | Qt.AlignVCenter, f"{dbl}dB")
        for k in range(1, K + 1):
            v = float(np.clip((f.harm_db[k - 1] + 60) / 60, 0, 1))
            fk = f.f0 * k if np.isfinite(f.f0) else np.nan
            col = hue_color(pitch_class_hue(fk) if np.isfinite(fk) else 0.6, 0.75, 1.0, 0.9)
            x = bx0 + (k - 1) * bw
            y = by1 - (by1 - by0) * v
            grad = QLinearGradient(QPointF(0, by1), QPointF(0, by0))
            grad.setColorAt(0, col.darker(250))
            grad.setColorAt(1, col)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(grad))
            p.drawRect(QRectF(x + 1, y, bw - 2, by1 - y))
            p.setPen(QColor(150, 160, 180))
            if k <= 9 or k % 2 == 0:
                p.drawText(QRectF(x, by1 + 2, bw, 16), Qt.AlignHCenter, str(k))
        self.draw_title(p, "左: 各倍音 (2周期) / 下: 合計 = 声の形 / 右: 倍音の強さ")
        p.end()


# ======================================================================== オシロスコープ
class OscilloscopeView(Panel):
    title = "波形 (基音の位相でトリガー)"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.trails: deque = deque(maxlen=5)

    def draw_panel(self) -> None:
        p = self.begin()
        f = self.frame
        w, h = self.width(), self.height()
        cy = h / 2 + 8
        p.setPen(QPen(QColor(28, 34, 30), 1))
        for i in range(1, 8):
            x = w * i / 8
            p.drawLine(QPointF(x, 24), QPointF(x, h))
        for i in range(1, 4):
            y = 24 + (h - 24) * i / 4
            p.drawLine(QPointF(0, y), QPointF(w, y))
        if f is not None and len(f.scope) > 2:
            sc = _decimate(np.asarray(f.scope, dtype=np.float64), max(64, w // 2))
            peak = max(np.abs(sc).max(), 0.02)
            ys = cy - (h * 0.42) * sc / peak
            xs = np.linspace(4, w - 4, len(sc))
            self.trails.append(_poly(xs, ys))
            n = len(self.trails)
            for i, poly in enumerate(self.trails):
                age = (i + 1) / n
                if i == n - 1:
                    glow_line(p, poly, QColor(120, 255, 190), glow=(8,), glow_alpha=0.22)
                else:
                    p.setPen(QPen(QColor(60, 220, 140, int(70 * age)), 1.0))
                    p.drawPolyline(poly)
            if f.voiced and f.scope_periods:
                p.setPen(QPen(QColor(255, 255, 255, 40), 1, Qt.DashLine))
                for i in range(1, int(f.scope_periods)):
                    x = 4 + (w - 8) * i / f.scope_periods
                    p.drawLine(QPointF(x, 24), QPointF(x, h))
        extra = ""
        if f is not None and f.voiced:
            extra = f"3 周期表示 / 周期 {1000 / f.f0:.2f} ms"
        self.draw_title(p, extra)
        p.end()


# ======================================================================== スペクトラム
class SpectrumView(Panel):
    title = "スペクトラム (FFT, 対数周波数)"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._bar_key = None
        self._bar_brushes: list[QBrush] = []

    def _brushes(self, freqs: np.ndarray, y0: float, y1: float) -> list[QBrush]:
        """帯域ごとのグラデーション (色と縦位置だけで決まるので、パネルの大きさが変わるまで使い回す)."""
        key = (len(freqs), y0, y1, float(freqs[0]), float(freqs[-1]))
        if key != self._bar_key:
            self._bar_key = key
            self._bar_brushes = []
            for fq in freqs:
                col = hue_color(pitch_class_hue(fq), 0.75, 0.95, 0.9)
                grad = QLinearGradient(QPointF(0, y1), QPointF(0, y0))
                grad.setColorAt(0, col.darker(300))
                grad.setColorAt(1, col.lighter(130))
                self._bar_brushes.append(QBrush(grad))
        return self._bar_brushes

    def draw_panel(self) -> None:
        p = self.begin()
        f = self.frame
        w, h = self.width(), self.height()
        x0, x1, y0, y1 = 8, w - 8, 26, h - 20
        if f is None:
            self.draw_title(p)
            p.end()
            return
        fr = f.band_freqs
        lo, hi = np.log(fr[0]), np.log(fr[-1])

        def fx(freq):
            return x0 + (x1 - x0) * (np.log(freq) - lo) / (hi - lo)

        # オクターブ (C) の目盛り
        p.setFont(self.title_font)
        for octv in range(1, 10):
            fc = 440 * 2 ** ((12 * (octv + 1) - 69) / 12)
            if fr[0] <= fc <= fr[-1]:
                x = fx(fc)
                p.setPen(QPen(QColor(32, 36, 46), 1))
                p.drawLine(QPointF(x, y0), QPointF(x, y1))
                p.setPen(QColor(100, 110, 130))
                p.drawText(QPointF(x + 2, h - 5), f"C{octv}")
        n = len(f.bands)
        bw = (x1 - x0) / n
        rw = max(1.0, bw - 1.5)
        brushes = self._brushes(fr, y0, y1)
        p.setPen(Qt.NoPen)
        peaks = []
        for i in range(n):
            x = x0 + i * bw
            y = y1 - (y1 - y0) * float(f.bands[i])
            if y < y1 - 0.25:                       # 高さ 0 の帯は描かない
                p.setBrush(brushes[i])
                p.drawRect(QRectF(x + 0.5, y, rw, y1 - y))
            peaks.append(QRectF(x + 0.5, y1 - (y1 - y0) * float(f.peaks[i]) - 2, rw, 2))
        # ピーク (白い線) は同じ色なのでまとめて描く
        p.setBrush(QColor(255, 255, 255, 200))
        p.drawRects(peaks)
        # 倍音の位置マーカー
        if f.voiced:
            for k in range(1, 17):
                fk = f.f0 * k
                if fk > fr[-1]:
                    break
                x = fx(fk)
                a = float(f.harm_amp[k - 1])
                col = hue_color(pitch_class_hue(fk), 0.8, 1.0, 0.4 + 0.6 * a)
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                p.drawPolygon(QPolygonF([QPointF(x - 5, y0 - 2), QPointF(x + 5, y0 - 2), QPointF(x, y0 + 7)]))
                if k <= 6:
                    p.setPen(col)
                    p.drawText(QPointF(x + 5, y0 + 8), f"H{k}")
        self.draw_title(p, "▼ = 倍音 (k×f0) の位置")
        p.end()


# ======================================================================== スクロール・スペクトログラム
def _note_name(freq: float) -> str:
    m = int(round(69 + 12 * np.log2(freq / 440.0)))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


class SpecViewport(QObject):
    """スペクトログラムの表示範囲 (再生位置より過去 / 未来の秒数と、周波数の範囲).

    パート別に並べたときに同じ範囲で見比べられるよう、複数のパネルで共有する。
    """

    changed = Signal()
    DEFAULT = (6.0, 2.0, 50.0, 8000.0)          # 過去 6 秒〜未来 2 秒, 50Hz〜8kHz
    SPAN_MIN, SPAN_MAX = 0.5, 60.0              # 表示する秒数の範囲
    OCT_MIN = 0.25                              # 周波数方向の最大拡大 (1/4 オクターブ = 3 半音)
    F_LO, F_HI = SPEC_FMIN, SPEC_FMAX

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.past, self.future, self.fmin, self.fmax = self.DEFAULT

    @property
    def span(self) -> float:
        return self.past + self.future

    def state(self) -> tuple[float, float, float, float]:
        return self.past, self.future, self.fmin, self.fmax

    def is_default(self) -> bool:
        return all(abs(a - b) <= 1e-6 * max(1.0, b) for a, b in zip(self.state(), self.DEFAULT))

    def reset(self) -> None:
        self.set_state(*self.DEFAULT)

    def set_state(self, past: float, future: float, fmin: float, fmax: float) -> None:
        """範囲を制限内に収めて設定する (再生位置は必ず画面の中に入る)."""
        total = past + future
        frac = float(np.clip(past / total, 0.0, 1.0)) if total > 1e-9 else 0.75
        span = float(np.clip(total, self.SPAN_MIN, self.SPAN_MAX))
        lo_b, hi_b = np.log(self.F_LO), np.log(self.F_HI)
        lo, hi = np.log(max(float(fmin), 1e-3)), np.log(max(float(fmax), 1e-3))
        width = float(np.clip(hi - lo, self.OCT_MIN * np.log(2.0), hi_b - lo_b))
        lo = float(np.clip(0.5 * (lo + hi) - width / 2, lo_b, hi_b - width))
        new = (span * frac, span * (1.0 - frac), float(np.exp(lo)), float(np.exp(lo + width)))
        if any(abs(a - b) > 1e-9 * max(1.0, abs(b)) for a, b in zip(new, self.state())):
            self.past, self.future, self.fmin, self.fmax = new
            self.changed.emit()


class SpectrogramView(Panel):
    """スクロールするスペクトログラム (既定は過去 6 秒〜未来 2 秒). f0 軌跡と (解析結果があれば) 音程バーを重ねる.

    マウス操作:
      ホイール       拡大縮小 (時間は再生位置を中心に、周波数はマウスの位置を中心に)。
                     左の音名の上 / Shift+ホイールは周波数だけ、下の時刻の上 / Ctrl+ホイールは時間だけ
      左ドラッグ     表示範囲を動かす (上下 = 周波数, 左右 = 再生位置の前後どちらを広く見るか)
      右ドラッグ     拡大縮小 (右へ = 時間を拡大, 上へ = 周波数を拡大)
      ダブルクリック 元の範囲に戻す
    """

    title = "スペクトログラム + ピッチ軌跡"
    detail = 0
    HINT = ("ホイール: 拡大縮小 (Shift=周波数 / Ctrl=時間) ・ ドラッグ: 移動 ・ ダブルクリック: 元に戻す",
            "ホイール: 拡大縮小 ・ ドラッグ: 移動 ・ ダブルクリック: 戻す", "ホイール: 拡大縮小")

    def __init__(self, parent=None, viewport: SpecViewport | None = None) -> None:
        super().__init__(parent)
        self.track: Track | None = None
        self.vp = viewport if viewport is not None else SpecViewport(self)
        self.vp.changed.connect(self._viewport_changed)
        self.lut = self._make_lut()
        self.axis_font = QFont(self.font())
        self.axis_font.setPointSizeF(8.5)
        self._drag = None          # (ボタン, 押した位置, 押した時の表示範囲, 押した位置の周波数)
        self.setCursor(Qt.OpenHandCursor)

    @staticmethod
    def _make_lut() -> list[int]:
        import pyqtgraph as pg

        cm = pg.colormap.get("magma")
        cols = cm.getLookupTable(0.0, 1.0, 256, alpha=False)
        return [QColor(int(r), int(g), int(b)).rgb() for r, g, b in cols]

    def set_track(self, track: Track | None) -> None:
        self.track = track
        self.update()

    def set_viewport(self, vp: SpecViewport) -> None:
        """表示範囲を他のパネルと共有する."""
        if vp is self.vp:
            return
        self.vp.changed.disconnect(self._viewport_changed)
        self.vp = vp
        vp.changed.connect(self._viewport_changed)
        self.update()

    def _viewport_changed(self) -> None:
        self.update()

    # ------------------------------------------------------------------ 座標
    def _plot_rect(self) -> tuple[float, float, float, float]:
        """描画枠 (x0, x1, y0, y1). 左に音名、下に時刻の目盛りを置く."""
        return 40.0, self.width() - 6.0, 24.0, self.height() - 18.0

    def _freq_at(self, y: float) -> float:
        _x0, _x1, y0, y1 = self._plot_rect()
        lo, hi = np.log(self.vp.fmin), np.log(self.vp.fmax)
        frac = float(np.clip((y1 - y) / max(1.0, y1 - y0), 0.0, 1.0))
        return float(np.exp(lo + frac * (hi - lo)))

    @staticmethod
    def _zoomed(past: float, future: float, fmin: float, fmax: float, ft: float = 1.0, ff: float = 1.0,
                anchor: float | None = None) -> tuple[float, float, float, float]:
        """時間を ft 倍 (再生位置を中心), 周波数を ff 倍 (anchor Hz を中心) にした表示範囲."""
        total = past + future
        frac = past / total if total > 0 else 0.75
        span = total * ft
        lo, hi = np.log(fmin), np.log(fmax)
        a = float(np.clip(np.log(anchor), lo, hi)) if anchor else 0.5 * (lo + hi)
        return span * frac, span * (1 - frac), float(np.exp(a - (a - lo) * ff)), float(np.exp(a + (hi - a) * ff))

    # ------------------------------------------------------------------ マウス操作
    def wheelEvent(self, ev) -> None:  # noqa: N802
        if self.track is None:
            ev.ignore()
            return
        d = ev.angleDelta()
        steps = (d.y() or d.x()) / 120.0
        if steps:
            pos = ev.position()
            x0, _x1, _y0, y1 = self._plot_rect()
            mods = ev.modifiers()
            only_f = pos.x() < x0 or bool(mods & Qt.ShiftModifier)
            only_t = pos.y() > y1 or bool(mods & Qt.ControlModifier)
            if only_f and only_t:
                only_f = only_t = False
            k = 0.85 ** steps                          # 奥へ回す = 拡大
            self.vp.set_state(*self._zoomed(*self.vp.state(), ft=1.0 if only_f else k,
                                            ff=1.0 if only_t else k, anchor=self._freq_at(pos.y())))
        ev.accept()

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if self.track is None or ev.button() not in (Qt.LeftButton, Qt.RightButton):
            super().mousePressEvent(ev)
            return
        pos = ev.position()
        self._drag = (ev.button(), QPointF(pos), self.vp.state(), self._freq_at(pos.y()))
        self.setCursor(Qt.ClosedHandCursor if ev.button() == Qt.LeftButton else Qt.SizeAllCursor)
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._drag is None:
            super().mouseMoveEvent(ev)
            return
        button, p0, (past, future, fmin, fmax), anchor = self._drag
        pos = ev.position()
        dx, dy = pos.x() - p0.x(), pos.y() - p0.y()
        x0, x1, y0, y1 = self._plot_rect()
        if button == Qt.LeftButton:
            # 絵がマウスに付いてくるように動かす: 右へ = 過去側が広く見える, 下へ = 高い音が見える
            span = past + future
            new_past = past + dx / max(1.0, x1 - x0) * span
            k = float(np.exp(dy / max(1.0, y1 - y0) * (np.log(fmax) - np.log(fmin))))
            self.vp.set_state(new_past, span - new_past, fmin * k, fmax * k)
        else:
            self.vp.set_state(*self._zoomed(past, future, fmin, fmax, ft=float(np.exp(-dx / 150.0)),
                                            ff=float(np.exp(dy / 150.0)), anchor=anchor))
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if self._drag is not None and ev.button() == self._drag[0]:
            self._drag = None
            self.setCursor(Qt.OpenHandCursor)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton:
            self.vp.reset()
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)

    # ------------------------------------------------------------------ 描画
    def draw_panel(self) -> None:
        p = self.begin()
        f, tr = self.frame, self.track
        x0, x1, y0, y1 = self._plot_rect()
        if f is None or tr is None or x1 - x0 < 8 or y1 - y0 < 8:
            self.draw_title(p)
            p.end()
            return
        vp = self.vp
        t = f.t
        ta, tb = t - vp.past, t + vp.future
        lo, hi = np.log(vp.fmin), np.log(vp.fmax)
        pw, ph = x1 - x0, y1 - y0

        def tx(tt):
            return x0 + pw * (tt - ta) / (tb - ta)

        def fy(freq):
            return y1 - ph * (np.log(freq) - lo) / (hi - lo)

        p.save()
        p.setClipRect(QRectF(x0, y0, pw, ph))
        self._draw_image(p, tr, ta, tb, lo, hi, tx, fy)
        ticks = self._note_ticks(lo, hi, ph, fy)
        for m, y in ticks:
            p.setPen(QPen(QColor(255, 255, 255, 25 if m % 12 == 0 else 12), 1))
            p.drawLine(QPointF(x0, y), QPointF(x1, y))
        # 音程バー (解析結果)
        for (n0, n1, m) in tr.notes:
            if n1 < ta or n0 > tb:
                continue
            fm = 440 * 2 ** ((m - 69) / 12)
            yc = fy(fm)
            hh = max(8.0, abs(fy(fm * 2 ** (1 / 12)) - yc))   # 最低 8px の太さで見やすく
            hue = pitch_class_hue(fm)
            p.setPen(QPen(QColor(255, 255, 255, 170), 1))
            p.setBrush(hue_color(hue, 0.65, 1.0, 0.45 if n0 <= t <= n1 else 0.25))
            p.drawRoundedRect(QRectF(tx(n0), yc - hh / 2, tx(n1) - tx(n0), hh), 3, 3)
        # f0 軌跡 (枠の外にはみ出す分は枠で切り取る)
        sel = (tr.f0_t >= ta - 0.05) & (tr.f0_t <= tb + 0.05)
        if sel.any():
            tt, ff = tr.f0_t[sel], tr.f0[sel].copy()
            ff[(ff < vp.fmin / 2) | (ff > vp.fmax * 2)] = np.nan
            with np.errstate(all="ignore"):
                yy = fy(ff)
            path = arrayToQPath(tx(tt), yy, connect="finite")
            glow_line(p, path, QColor(120, 255, 220), glow=(6,), glow_alpha=0.28)
        # コード帯 (上端) と拍
        if tr.chords:
            p.setFont(self.title_font)
            for c in tr.chords:
                if c["end"] < ta or c["start"] > tb:
                    continue
                xa, xb = max(x0, tx(c["start"])), min(x1, tx(c["end"]))
                if xb - xa < 2:
                    continue
                r = QRectF(xa + 1, y0 + 1, xb - xa - 2, 17)
                col = hue_color((c["root"] % 12) / 12.0, 0.55, 0.62, 0.92) if c.get("root") is not None \
                    else QColor(60, 64, 76, 230)
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                p.drawRoundedRect(r, 3, 3)
                if r.width() > 24:
                    p.setPen(QColor(255, 255, 255))
                    p.drawText(r, Qt.AlignCenter, c["label"])
        if tr.beats:
            p.setPen(QPen(QColor(255, 255, 255, 60), 1))
            for b in tr.beats:
                if ta <= b <= tb:
                    xb = tx(b)
                    p.drawLine(QPointF(xb, y1 - 6), QPointF(xb, y1))
        # 再生位置
        xp = tx(t)
        p.setPen(QPen(QColor(254, 202, 87), 2))
        p.drawLine(QPointF(xp, y0), QPointF(xp, y1))
        if f.voiced and vp.fmin <= f.f0 <= vp.fmax:
            p.setBrush(QColor(254, 202, 87))
            p.drawEllipse(QPointF(xp, fy(f.f0)), 5, 5)
        p.restore()
        # 目盛りの文字 (枠の外)
        p.setFont(self.axis_font)
        for m, y in ticks:
            if y0 + 6 <= y <= y1 - 1:
                p.setPen(QColor(120, 130, 150) if m % 12 == 0 else QColor(92, 100, 118))
                p.drawText(QPointF(4, y + 4), f"{NOTE_NAMES[m % 12]}{m // 12 - 1}")
        self._draw_time_axis(p, tr, ta, tb, tx, x0, y1)
        extra = f"f0: {tr.f0_source}"
        if not vp.is_default():
            extra += f"   ・   {vp.span:.1f} 秒 / {_note_name(vp.fmin)}〜{_note_name(vp.fmax)}"
        self.draw_title(p, extra, hint=self.HINT)
        p.end()

    def _draw_image(self, p: QPainter, tr: Track, ta: float, tb: float, lo: float, hi: float, tx, fy) -> None:
        hop = tr.spec_hop_s
        ia = int(np.floor(ta / hop)) - 1
        ib = int(np.ceil(tb / hop)) + 2
        lb = np.log(tr.band_freqs)
        half = 0.5 * (lb[1] - lb[0]) if len(lb) > 1 else 0.05     # 帯域は対数で等間隔
        rows = np.nonzero((lb + half > lo) & (lb - half < hi))[0]
        if rows.size == 0 or ib <= ia:
            return
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        img = np.zeros((r1 - r0, ib - ia), dtype=np.uint8)
        s, e = max(0, ia), min(len(tr.spec), ib)
        if e > s:
            img[:, s - ia: e - ia] = tr.spec[s:e, r0:r1].T
        # コントラスト: 曲中の大きい成分 (spec_ref) から下 65dB を 0..255 に割り当てる
        # (int16 だと 255 倍したときに桁あふれして、いちばん強い成分が黒くなるので int32 で計算)
        span = int(65.0 / SPEC_RANGE_DB * 255)
        lo_v = tr.spec_ref - span
        img = np.clip((img[::-1].astype(np.int32) - lo_v) * 255 // span, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img)
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Indexed8)
        qimg.setColorTable(self.lut)
        # 帯域・フレームの境目がちょうど目盛りに合う位置に置く (拡大しても f0 や音程バーとずれない)。
        # フレーム i は時刻 i*hop が中心
        top, bottom = fy(np.exp(lb[r1 - 1] + half)), fy(np.exp(lb[r0] - half))
        left, right = tx((ia - 0.5) * hop), tx((ib - 0.5) * hop)
        p.drawImage(QRectF(left, top, right - left, bottom - top), qimg)

    @staticmethod
    def _note_ticks(lo: float, hi: float, ph: float, fy) -> list[tuple[int, float]]:
        """音名の目盛り [(MIDI, y)]. 拡大すると半音ごと、引くと C と G → C だけ."""
        semis = 12.0 * (hi - lo) / np.log(2.0)
        px = ph / max(semis, 1e-6)                       # 半音あたりのピクセル数
        pcs = set(range(12)) if px >= 12 else ({0, 7} if px * 5 >= 14 else {0})
        l440, l2 = np.log(440.0), np.log(2.0)
        m_lo = int(np.ceil(69 + 12 * (lo - l440) / l2))
        m_hi = int(np.floor(69 + 12 * (hi - l440) / l2))
        return [(m, float(fy(440.0 * 2 ** ((m - 69) / 12)))) for m in range(m_lo, m_hi + 1) if m % 12 in pcs]

    def _draw_time_axis(self, p: QPainter, tr: Track, ta: float, tb: float, tx, x0: float, y1: float) -> None:
        """下端の時刻の目盛り (元の音源での時刻)."""
        fm = QFontMetricsF(self.axis_font)
        n_max = max(2.0, (self.width() - x0) / 90.0)
        step = next((s for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0)
                     if (tb - ta) / s <= n_max), 60.0)
        digits = 2 if step < 0.1 else (1 if step < 1 else 0)
        off, dur = tr.time_offset, tr.duration
        p.setFont(self.axis_font)
        for k in range(int(np.ceil((ta + off) / step)), int(np.floor((tb + off) / step)) + 1):
            t_abs = k * step
            if not (-1e-6 <= t_abs - off <= dur + 1e-6):
                continue
            x = tx(t_abs - off)
            p.setPen(QPen(QColor(90, 98, 115), 1))
            p.drawLine(QPointF(x, y1), QPointF(x, y1 + 4))
            label = fmt_time(t_abs, digits)
            lw = fm.horizontalAdvance(label)
            if x - lw / 2 < x0 - 2 or x + lw / 2 > self.width() - 2:
                continue
            p.setPen(QColor(120, 130, 150))
            p.drawText(QRectF(x - lw / 2 - 2, y1 + 3, lw + 4, 14), Qt.AlignCenter, label)


# ======================================================================== コード / 五度圏
FIFTHS = [(i * 7) % 12 for i in range(12)]     # C G D A E B F# C# G# D# A# F
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


class ChordWheelView(Panel):
    """五度圏の上に、いま鳴っている 12 音名の強さ (クロマ) とコードの構成音を表示.

    解析結果があればそのコード進行 (次のコードまでのカウントダウン付き)、
    無ければクロマからその場でコードを推定する。
    """

    title = "コード / 五度圏"
    detail = 1                     # クロマ (12 音名の強さ) だけ使う

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.track: Track | None = None
        self._live_score: np.ndarray | None = None
        self._tmpl = None
        self.mid_font = QFont(self.font())
        self.mid_font.setPointSizeF(13)
        self.mid_font.setBold(True)
        self.huge_font = QFont(self.font())
        self.huge_font.setPointSizeF(30)
        self.huge_font.setBold(True)

    def set_track(self, track: Track | None) -> None:
        self.track = track
        self._live_score = None
        self.update()

    def _live_chord(self, chroma: np.ndarray) -> dict | None:
        """クロマとテンプレートの照合でその場のコードを推定 (解析結果が無いとき用)."""
        from ..chords import QUALITIES, _templates

        if self._tmpl is None:
            T, keys = _templates()
            pen = np.array([QUALITIES[q][1] for _r, q in keys])
            self._tmpl = (T, keys, pen)
        T, keys, pen = self._tmpl
        if chroma.max() < 1e-6:
            return None
        x = chroma / (np.linalg.norm(chroma) + 1e-9)
        s = T @ x - pen
        self._live_score = s if self._live_score is None else self._live_score * 0.85 + s * 0.15
        i = int(np.argmax(self._live_score))
        root, q = keys[i]
        return {"root": root, "quality": q, "label": SHARP_NAMES[root] + q, "degree": "", "start": 0, "end": 0}

    def draw_panel(self) -> None:
        from ..chords import QUALITIES

        p = self.begin()
        f, tr = self.frame, self.track
        w, h = self.width(), self.height()
        if f is None or tr is None:
            self.draw_title(p)
            p.end()
            return
        key = tr.key or {}
        names = FLAT_NAMES if key.get("flats") else SHARP_NAMES
        # 現在のコード
        cur = None
        live = False
        if tr.chords and 0 <= f.chord_index < len(tr.chords):
            cur = tr.chords[f.chord_index]
        elif not tr.chords:
            cur = self._live_chord(f.chroma)
            live = True
        side = min(w * 0.55, h - 40)
        cx, cy = 14 + side / 2, 30 + side / 2
        R = side / 2 - 18
        # 調 (キー) の 7 音を薄い帯で
        if key.get("relative_major") is not None:
            tonic = key["relative_major"]
            scale = {(tonic + iv) % 12 for iv in (0, 2, 4, 5, 7, 9, 11)}
            for i, pc in enumerate(FIFTHS):
                if pc in scale:
                    a = -np.pi / 2 + i * np.pi / 6
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(255, 255, 255, 14))
                    p.drawEllipse(QPointF(cx + R * np.cos(a), cy + R * np.sin(a)), 22, 22)
        p.setPen(QPen(QColor(50, 55, 70), 1))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), R, R)
        pos = {pc: QPointF(cx + R * np.cos(-np.pi / 2 + i * np.pi / 6), cy + R * np.sin(-np.pi / 2 + i * np.pi / 6))
               for i, pc in enumerate(FIFTHS)}
        # 構成音を結ぶ多角形
        tones: list[int] = []
        if cur is not None and cur.get("root") is not None:
            tones = [(cur["root"] + iv) % 12 for iv in QUALITIES[cur["quality"]][0]]
            ordered = [pc for pc in FIFTHS if pc in tones]
            poly = QPolygonF([pos[pc] for pc in ordered])
            col = hue_color(cur["root"] / 12.0, 0.6, 1.0, 0.28)
            p.setPen(QPen(hue_color(cur["root"] / 12.0, 0.5, 1.0, 0.9), 2))
            p.setBrush(col)
            p.drawPolygon(poly)
        # 12 音名のノード (大きさ = いまの強さ)
        p.setFont(self.title_font)
        for pc, pt in pos.items():
            v = float(f.chroma[pc])
            r = 7 + 15 * v
            p.setPen(Qt.NoPen)
            p.setBrush(hue_color(pc / 12.0, 0.75, 1.0, 0.25 + 0.75 * v))
            p.drawEllipse(pt, r, r)
            if tones and pc == tones[0]:
                p.setPen(QPen(QColor(255, 255, 255), 2.5))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(pt, r + 4, r + 4)
            p.setPen(QColor(235, 238, 245) if pc in tones else QColor(140, 150, 170))
            p.drawText(QRectF(pt.x() - 20, pt.y() - 9, 40, 18), Qt.AlignCenter, names[pc])
        # 右側: コード名, 度数, 次のコード
        x0 = cx + side / 2 + 12
        y = 34
        p.setPen(QColor(150, 160, 180))
        p.setFont(self.title_font)
        info = []
        if key.get("label"):
            info.append(f"キー {key.get('name', '')}")
        if tr.beats and len(tr.beats) > 2:
            bpm = 60.0 / max(1e-3, float(np.median(np.diff(tr.beats))))
            info.append(f"{bpm:.0f} BPM")
        p.drawText(QPointF(x0, y + 4), "  ".join(info) if info else ("推定 (リアルタイム)" if live else ""))
        p.setFont(self.huge_font)
        p.setPen(QColor(255, 255, 255))
        label = cur["label"] if cur else "N.C."
        p.drawText(QRectF(x0, y + 10, w - x0 - 8, 52), Qt.AlignLeft | Qt.AlignVCenter, label)
        p.setFont(self.mid_font)
        if cur and cur.get("degree"):
            p.setPen(QColor(254, 202, 87))
            p.drawText(QPointF(x0, y + 86), cur["degree"])
        if tr.chords:
            p.setFont(self.title_font)
            yy = y + 112
            p.setPen(QColor(120, 130, 150))
            p.drawText(QPointF(x0, yy), "次のコード")
            nxt = tr.chords[f.chord_index + 1: f.chord_index + 5] if f.chord_index >= 0 else \
                [c for c in tr.chords if c["start"] > f.t][:4]
            bar_w = min(360.0, max(40.0, w - x0 - 70))
            for c in nxt:
                yy += 24
                dt = c["start"] - f.t
                if c.get("root") is not None:
                    p.setBrush(hue_color(c["root"] / 12.0, 0.55, 0.62, 0.95))
                else:
                    p.setBrush(QColor(60, 64, 76))
                p.setPen(Qt.NoPen)
                frac = float(np.clip(1.0 - dt / 4.0, 0.0, 1.0))
                p.drawRoundedRect(QRectF(x0, yy - 15, bar_w * max(frac, 0.04), 19), 3, 3)
                p.setPen(QColor(255, 255, 255))
                p.drawText(QPointF(x0 + 6, yy), f"{c['label']}")
                p.setPen(QColor(150, 160, 180))
                p.drawText(QPointF(x0 + bar_w + 6, yy), f"{dt:.1f}s")
        self.draw_title(p, "大きさ = 音名の強さ / 多角形 = 構成音")
        p.end()
