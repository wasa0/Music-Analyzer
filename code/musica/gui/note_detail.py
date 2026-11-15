"""選択したノートの詳細: ピッチ軌跡 / ピッチ変動の FFT / 倍音スペクトル / スペクトログラム."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .. import config
from ..analysis import AnalysisResult
from ..features import analyze_vibrato, detrend_slow, modulation_spectrum, vibrato_envelope
from ..pitch import average_spectrum, midi_to_hz, midi_to_name
from .common import APPROACH_COLORS, EVENT_COLORS, RELEASE_COLORS, fmt_time


def _f(v, fmt="{:.0f}", unit=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return fmt.format(v) + unit


class NoteDetailPanel(QWidget):
    playRequested = Signal(float, float, str)   # (start_abs, end_abs, "vocal"|"mix")
    stepRequested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.res: AnalysisResult | None = None
        self.idx = -1
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 0, 0, 0)

        head = QHBoxLayout()
        self.title = QLabel("ノートをクリックすると詳細を表示します")
        self.title.setStyleSheet("font-size:13pt; font-weight:bold; color:#ffffff;")
        head.addWidget(self.title, 1)
        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶")
        self.prev_btn.setToolTip("前のノート")
        self.next_btn.setToolTip("次のノート")
        self.prev_btn.clicked.connect(lambda: self.stepRequested.emit(-1))
        self.next_btn.clicked.connect(lambda: self.stepRequested.emit(+1))
        self.play_v = QPushButton("♪ ボーカル")
        self.play_m = QPushButton("♪ ミックス")
        self.play_v.clicked.connect(lambda: self._play("vocal"))
        self.play_m.clicked.connect(lambda: self._play("mix"))
        for b in (self.prev_btn, self.next_btn, self.play_v, self.play_m):
            head.addWidget(b)
        lay.addLayout(head)

        split = QSplitter(Qt.Vertical)
        lay.addWidget(split, 1)

        self.plots = pg.GraphicsLayoutWidget()
        self.p_pitch = self.plots.addPlot(row=0, col=0, title="ピッチ軌跡 (音程バーからのずれ)")
        self.p_pitch.setLabel("left", "cent")
        self.p_pitch.setLabel("bottom", "ノート開始からの時間", units="ms")
        self.p_pitch.showGrid(x=True, y=True, alpha=0.2)
        self.p_mod = self.plots.addPlot(row=1, col=0, title="ピッチ変動の FFT (ビブラート解析)")
        self.p_mod.setLabel("left", "振幅 ±cent")
        self.p_mod.setLabel("bottom", "変調周波数", units="Hz")
        self.p_mod.showGrid(x=True, y=True, alpha=0.2)
        self.p_harm = self.plots.addPlot(row=2, col=0, title="倍音スペクトル (サステイン部の平均 FFT)")
        self.p_harm.setLabel("left", "dBFS")
        self.p_harm.setLabel("bottom", "周波数", units="Hz")
        self.p_harm.showGrid(x=True, y=True, alpha=0.2)
        self.p_spec = self.plots.addPlot(row=3, col=0, title="スペクトログラム (倍音の動き)")
        self.p_spec.setLabel("left", "Hz")
        self.p_spec.setLabel("bottom", "ノート開始からの時間", units="ms")
        for p in (self.p_pitch, self.p_mod, self.p_harm, self.p_spec):
            p.setMenuEnabled(False)
            p.getAxis("left").setWidth(52)
            p.titleLabel.setMaximumHeight(22)
        lay_p = self.plots.ci.layout
        for row, s in enumerate((4, 2, 3, 3)):
            lay_p.setRowStretchFactor(row, s)
        self.spec_img = pg.ImageItem()
        self.spec_img.setColorMap(pg.colormap.get("inferno"))
        self.p_spec.addItem(self.spec_img)
        split.addWidget(self.plots)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["項目", "値"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        split.addWidget(self.table)
        split.setSizes([760, 200])
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)

    # ------------------------------------------------------------------ 表示
    def clear(self) -> None:
        self.res, self.idx = None, -1
        for p in (self.p_pitch, self.p_mod, self.p_harm):
            p.clear()
        self.spec_img.clear()
        self.table.setRowCount(0)
        self.title.setText("ノートをクリックすると詳細を表示します")

    def set_note(self, res: AnalysisResult, i: int) -> None:
        if not (0 <= i < len(res.notes)):
            return
        self.res, self.idx = res, i
        n = res.notes[i]
        f = n.features
        off = res.offset_s
        self.title.setText(f"#{i}  {f['name']}  {fmt_time(f['start_s'] + off, 2)} 〜 {fmt_time(f['end_s'] + off, 2)}"
                           f"  ({f['dur_ms']:.0f} ms)")
        self._plot_pitch(res, n)
        self._plot_mod(res, n)
        self._plot_harm(res, n)
        self._plot_spec(res, n)
        self._fill_table(f, off)

    def _plot_pitch(self, res: AnalysisResult, n) -> None:
        p = self.p_pitch
        p.clear()
        f = n.features
        pad = 15
        a, b = max(0, n.start - pad), min(len(res.midi_raw), n.end + pad)
        t_ms = (np.arange(a, b) - n.start) * 10.0
        cents = (res.midi_raw[a:b] - n.midi) * 100.0
        N = n.end - n.start
        settle_ms, rel_ms = f["settle_idx"] * 10.0, f["release_idx"] * 10.0
        att = pg.LinearRegionItem((0, settle_ms), movable=False, brush=pg.mkBrush(APPROACH_COLORS[f["approach"]] + "40"))
        rel = pg.LinearRegionItem((rel_ms, N * 10.0), movable=False,
                                  brush=pg.mkBrush(RELEASE_COLORS[f["release"]] + "40"))
        for r in (att, rel):
            for ln in r.lines:
                ln.setPen(pg.mkPen(None))
            p.addItem(r)
        p.addItem(pg.InfiniteLine(angle=0, pos=0, pen=pg.mkPen("#7f8fa6", width=1, style=Qt.DashLine)))
        center_c = f["intonation_cents"]
        p.plot([0, N * 10], [center_c, center_c], pen=pg.mkPen("#1dd1a1", width=1, style=Qt.DotLine), name="中心音高")
        p.plot(t_ms, cents, pen=pg.mkPen("#f5f6fa", width=2), connect="finite", name="ピッチ")
        if f.get("vib"):
            c = (res.midi[n.start:n.end] - n.center) * 100.0
            c = np.nan_to_num(c)
            _, env = vibrato_envelope(c, f["vib_rate_hz"])
            tt = np.arange(N) * 10.0
            k0 = int(f["vib_onset_ms"] / 10) if np.isfinite(f["vib_onset_ms"]) else 0
            pen = pg.mkPen("#ff9ff3", width=1.2, style=Qt.DashLine)
            p.plot(tt[k0:], center_c + env[k0:], pen=pen, name="ビブラート包絡")
            p.plot(tt[k0:], center_c - env[k0:], pen=pen)
            p.addItem(pg.InfiniteLine(angle=90, pos=f["vib_onset_ms"], pen=pg.mkPen("#ff9ff3", width=1)))
        for e in f.get("events", []):
            x = e["time_s"] * 1000 - f["start_s"] * 1000
            k = int(round(x / 10))
            yv = cents[k + (n.start - a)] if 0 <= k + (n.start - a) < len(cents) else 0
            p.plot([x], [yv if np.isfinite(yv) else 0], pen=None, symbol="star",
                   symbolBrush=EVENT_COLORS[e["type"]], symbolSize=13)
        ac, rc = APPROACH_COLORS[f["approach"]], RELEASE_COLORS[f["release"]]
        p.setTitle(f"ピッチ軌跡　<span style='color:{ac}'>入り: {f['approach_label']}</span>　"
                   f"<span style='color:{rc}'>語尾: {f['release_label']}</span>"
                   + ("　<span style='color:#ff9ff3'>〰 ビブラート</span>" if f.get("vib") else ""), size="10pt")
        valid = cents[np.isfinite(cents)]
        lo = max(-1200.0, min(-150.0, float(np.percentile(valid, 1)) - 30)) if valid.size else -150
        hi = min(1200.0, max(150.0, float(np.percentile(valid, 99)) + 30)) if valid.size else 150
        p.setYRange(lo, hi)
        p.setXRange(t_ms[0], t_ms[-1], padding=0.02)

    def _plot_mod(self, res: AnalysisResult, n) -> None:
        p = self.p_mod
        p.clear()
        f = n.features
        c = np.nan_to_num((res.midi[n.start:n.end] - n.center) * 100.0)
        s, e = f["settle_idx"], f["release_idx"]
        if f.get("vib") and np.isfinite(f.get("vib_onset_ms", np.nan)):
            s = max(s, int(f["vib_onset_ms"] / 10))
        seg = c[s:e]
        band = pg.LinearRegionItem((3, 10), movable=False, brush=pg.mkBrush(255, 159, 243, 25))
        for ln in band.lines:
            ln.setPen(pg.mkPen(None))
        p.addItem(band)
        p.setXRange(0, 16, padding=0)
        if len(seg) < 16:
            p.setTitle("ピッチ変動の FFT　<span style='color:#a4b0be'>(サステインが短いため解析できません)</span>",
                       size="10pt")
            return
        fr, amp = modulation_spectrum(detrend_slow(seg), 100.0)
        m = fr <= 16
        p.plot(fr[m], amp[m], pen=pg.mkPen("#ff9ff3", width=2), fillLevel=0, brush=(255, 159, 243, 50))
        vib = analyze_vibrato(seg)
        title = "ピッチ変動の FFT (ビブラート解析)"
        if np.isfinite(vib["vib_rate_hz"]):
            k = np.argmin(np.abs(fr - vib["vib_rate_hz"]))
            p.plot([fr[k]], [amp[k]], pen=None, symbol="o", symbolBrush="#ffffff", symbolSize=8)
            verdict = "<span style='color:#ff9ff3'>ビブラート</span>" if f.get("vib") else "ビブラートなし"
            title = (f"ピッチ変動の FFT　ピーク {vib['vib_rate_hz']:.2f} Hz / ±{amp[k]:.0f} cent　"
                     f"規則性 {vib['vib_regularity']:.2f} → {verdict}")
        p.setTitle(title, size="10pt")
        p.setYRange(0, max(float(amp[m].max()) * 1.15, 5.0))

    def _plot_harm(self, res: AnalysisResult, n) -> None:
        p = self.p_harm
        p.clear()
        f = n.features
        y = res.vocal_analysis_signal()
        hop, sr = config.HOP, config.ANALYSIS_SR
        s0 = (n.start + f["settle_idx"]) * hop
        s1 = (n.start + max(f["release_idx"], f["settle_idx"] + 5)) * hop
        freqs, db = average_spectrum(y, sr, s0, s1)
        m = freqs <= 8000
        p.plot(freqs[m], db[m], pen=pg.mkPen("#70a1ff", width=1.3), fillLevel=-120, brush=(112, 161, 255, 40))
        f0 = float(np.nanmedian(res.f0_hz[n.start + f["settle_idx"]: n.start + max(f["release_idx"], f["settle_idx"] + 1)]))
        if not np.isfinite(f0):
            f0 = float(midi_to_hz(n.center + res.meta.get("tuning_offset", 0.0)))
        hd = f.get("harmonic_profile_db") or []
        top = float(np.max(db[m])) if m.any() else 0
        xs, ys = [], []
        for k in range(1, 21):
            fk = k * f0
            if fk > 8000:
                break
            i = np.argmin(np.abs(freqs - fk))
            lo, hi = max(0, i - 6), min(len(db), i + 7)
            j = lo + int(np.argmax(db[lo:hi]))
            xs.append(freqs[j])
            ys.append(db[j])
            p.addItem(pg.InfiniteLine(angle=90, pos=fk, pen=pg.mkPen(255, 255, 255, 30)))
            if k <= 10:
                t = pg.TextItem(f"H{k}", color="#feca57", anchor=(0.5, 1))
                t.setPos(freqs[j], db[j] + 2)
                p.addItem(t)
        p.plot(xs, ys, pen=pg.mkPen("#feca57", width=1, style=Qt.DotLine), symbol="o", symbolSize=6,
               symbolBrush="#feca57")
        p.setTitle(f"倍音スペクトル (FFT)　f0 ≈ {f0:.1f} Hz　H1-H2 {_f(f['h1h2_db'], '{:+.1f}')} dB　"
                   f"傾斜 {_f(f['tilt_db_oct'], '{:.1f}')} dB/oct　HNR {_f(f['hnr_db'], '{:.1f}')} dB", size="10pt")
        p.setXRange(0, 8000, padding=0)
        p.setYRange(top - 80, top + 12)
        _ = hd

    def _plot_spec(self, res: AnalysisResult, n) -> None:
        p = self.p_spec
        for it in list(p.items):
            if it is not self.spec_img:
                p.removeItem(it)
        y = res.vocal_analysis_signal()
        sr, hop = config.ANALYSIS_SR, config.HOP
        pad = 15
        a = max(0, (n.start - pad) * hop)
        b = min(len(y), (n.end + pad) * hop)
        win, nfft, h = 1024, 4096, 110
        w = np.hanning(win)
        seg = np.pad(y[a:b], (win // 2, win // 2))
        nfr = max(1, (len(seg) - win) // h)
        idx = np.arange(win)[None, :] + h * np.arange(nfr)[:, None]
        S = np.abs(np.fft.rfft(seg[idx] * w, nfft, axis=1))
        fmax = 4000.0
        kmax = int(fmax / (sr / nfft))
        D = 20 * np.log10(S[:, :kmax] + 1e-6)
        D = np.clip(D - D.max(), -60, 0)
        self.spec_img.setImage(D, autoLevels=False, levels=(-55, 0))
        t0 = (a / sr - n.start * config.HOP_SEC) * 1000
        self.spec_img.setRect(QRectF(t0, 0, nfr * h / sr * 1000, fmax))
        t = (np.arange(max(0, n.start - pad), min(len(res.f0_hz), n.end + pad)) - n.start) * 10.0
        f0 = res.f0_hz[max(0, n.start - pad): min(len(res.f0_hz), n.end + pad)]
        for k in range(1, 9):
            p.plot(t, f0 * k, pen=pg.mkPen(0, 255, 200, 230 if k == 1 else 110, width=2 if k == 1 else 1),
                   connect="finite")
        p.setXRange(t0, t0 + nfr * h / sr * 1000, padding=0)
        p.setYRange(0, fmax, padding=0)

    def _fill_table(self, f: dict, off: float) -> None:
        rows: list[tuple[str, str]] = [
            ("■ 音程", ""),
            ("音名 / 中心音高", f"{f['name']} / {midi_to_name(f['center_midi'])} ({f['intonation_cents']:+.0f} cent)"),
            ("安定度 (ビブラート除去後の揺れ)", _f(f["stability_cents"], "{:.1f}", " cent")),
            ("■ 入り (アタック)", ""),
            ("入り方", f["approach_label"]),
            ("開始時のずれ / 最大ずれ", f"{f['attack_onset_cents']:+.0f} / {f['attack_depth_cents']:+.0f} cent"),
            ("目標音に落ち着くまで", f"{f['attack_ms']:.0f} ms"),
            ("オーバーシュート", _f(f["overshoot_cents"], "{:.0f}", " cent") if f["overshoot_cents"] else "なし"),
            ("前のノートからの移行", _f(f.get("transition_ms"), "{:.0f}", " ms") +
             (f" ({f['transition_interval']:+d} 半音)" if f.get("transition_interval") else "")),
            ("■ ビブラート (ピッチ軌跡の FFT)", ""),
            ("判定", "あり" if f["vib"] else "なし"),
            ("速さ / 深さ", f"{_f(f['vib_rate_hz'], '{:.2f}', ' Hz')} / ±{_f(f['vib_extent_cents'], '{:.0f}', ' cent')}"),
            ("規則性 (ピーク集中度)", _f(f["vib_regularity"], "{:.2f}")),
            ("かかり始め", _f(f["vib_onset_ms"], "{:.0f}", " ms") +
             (f" (ノートの後ろ {f['vib_len_ratio'] * 100:.0f}%)" if np.isfinite(f.get("vib_len_ratio", np.nan)) else "")),
            ("深さの変化 / 速さの変化", f"{_f(f['vib_depth_slope'], '{:+.0f}', ' cent/s')} / "
                                       f"{_f(f['vib_rate_slope'], '{:+.2f}', ' Hz/s')}"),
            ("トレモロ (音量の揺れ)", _f(f["tremolo_db"], "{:.1f}", " dB")),
            ("■ 語尾 (リリース)", ""),
            ("処理", f["release_label"]),
            ("最後のずれ / 最大ずれ", f"{f['release_end_cents']:+.0f} / {f['release_depth_cents']:+.0f} cent"),
            ("語尾の長さ", f"{f['release_ms']:.0f} ms"),
            ("■ 音量 (DYN)", ""),
            ("平均 / ピーク", f"{_f(f['dyn_mean_db'], '{:.1f}')} / {_f(f['dyn_peak_db'], '{:.1f}')} dBFS"),
            ("アタックの立ち上がり", _f(f["attack_rise_db"], "{:+.1f}", " dB")),
            ("クレッシェンド", _f(f["crescendo_db_s"], "{:+.1f}", " dB/s")),
            ("語尾の抜き", _f(f["release_decay_db"], "{:+.1f}", " dB")),
            ("■ 音色 (倍音 FFT)", ""),
            ("HNR (高い=クリア, 低い=息っぽい)", _f(f["hnr_db"], "{:.1f}", " dB")),
            ("スペクトル重心 (明るさ)", _f(f["centroid_hz"], "{:.0f}", " Hz")),
            ("H1-H2 (高い=柔らか/息, 低い=張り)", _f(f["h1h2_db"], "{:+.1f}", " dB")),
            ("スペクトル傾斜", _f(f["tilt_db_oct"], "{:.1f}", " dB/oct")),
        ]
        if f.get("chord") not in (None, "-"):
            tone = f.get("chord_tone")
            rows += [
                ("■ コードとの関係", ""),
                ("鳴っているコード (度数)", f"{f['chord']}  ({f.get('chord_degree', '-')})"),
                ("この音の役割", f"{f.get('chord_role', '-')}  " + ("(コードトーン)" if tone else
                                                                 "(テンション / 経過音)" if tone is False else "")),
            ]
        ev = f.get("events", [])
        rows.append(("■ イベント", f"{len(ev)} 件" if ev else "なし"))
        for e in ev:
            rows.append((f"  {e['label']}", f"{fmt_time(e['time_s'] + off, 2)}  {e['cents']:+.0f} cent / {e['dur_ms']:.0f} ms"))
        self.table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            a, b = QTableWidgetItem(k), QTableWidgetItem(v)
            if k.startswith("■"):
                for it in (a, b):
                    it.setBackground(pg.mkColor("#262a33"))
                    it.setForeground(pg.mkColor("#9ecbff"))
            self.table.setItem(r, 0, a)
            self.table.setItem(r, 1, b)
        self.table.resizeRowsToContents()

    def _play(self, which: str) -> None:
        if self.res is None or self.idx < 0:
            return
        f = self.res.notes[self.idx].features
        off = self.res.offset_s
        self.playRequested.emit(f["start_s"] + off - 0.15, f["end_s"] + off + 0.15, which)
