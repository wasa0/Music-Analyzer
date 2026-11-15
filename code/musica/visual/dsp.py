"""ビジュアライザー用の信号処理.

- Track: 曲全体の事前計算 (パートごとの対数周波数スペクトログラム, f0 軌跡, コード)
- build_views: 表示パート (ボーカル / フロント楽器 / カラオケ / 全パート) ごとの Track
- FrameAnalyzer: 再生位置ごとの FFT (スペクトラム帯域, 倍音の振幅と位相, 1 周期波形の再合成,
  トリガー位置, クロマ = 12 音名の強さ)
表示パートの Track は、そのパートだけを足した波形・スペクトログラム・ピッチを持つ
(パートの波形とスペクトログラムは共有するので軽い)。何を鳴らすか (ミキサー) とは独立。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from scipy.ndimage import median_filter

ProgressCB = Callable[[str, float], None]

SPEC_FMIN, SPEC_FMAX, SPEC_BANDS = 30.0, 16000.0, 256
SPEC_NFFT, SPEC_HOP = 4096, 512
SPEC_RANGE_DB = 100.0   # uint8 0..255 = -100..0 dBFS

# ------------------------------------------------------------------------------------ 表示パート
VIEW_ORDER = ("vocals", "front", "karaoke", "all")
VIEW_LABELS = {"vocals": "ボーカル", "front": "フロント楽器", "karaoke": "カラオケ", "all": "全パート"}
VIEW_COLORS = {"vocals": "#ff6b81", "front": "#1dd1a1", "karaoke": "#70a1ff", "all": "#dfe4ea"}
RHYTHM_STEMS = ("drums", "bass")        # フロント楽器に含めないリズム隊


def view_parts(key: str, stems) -> list[str] | None:
    """表示パート key に含めるパート名. そのパートが無ければ None.

    全パート = すべて (旧形式の結果はミックス) / ボーカル / フロント楽器 = ボーカルとリズム隊
    (ドラム・ベース) 以外 = ギター・ピアノ・その他 (シンセ等) / カラオケ = ボーカル以外すべて。
    """
    names = list(stems)
    legacy = "mix" in names                # 伴奏が分離されていない旧形式 (ミックス + ボーカル)
    if key == "all":
        return ["mix"] if legacy else names
    if len(names) < 2:
        return None
    if key == "vocals":
        return ["vocals"] if "vocals" in names else None
    if legacy:
        return None
    if key == "front":
        sel = [s for s in names if s not in ("vocals", "mix", *RHYTHM_STEMS)]
    elif key == "karaoke":
        sel = [s for s in names if s not in ("vocals", "mix")]
    else:
        return None
    return sel or None


def view_gains(key: str, stems) -> dict[str, float] | None:
    parts = view_parts(key, stems)
    if parts is None:
        return None
    return {s: (1.0 if s in parts else 0.0) for s in stems}


def log_band_matrix(sr: int, n_fft: int, fmin: float, fmax: float, n_bands: int) -> tuple[np.ndarray, np.ndarray]:
    """FFT ビン → 対数周波数帯域への重み行列 (n_bins, n_bands) と各帯域の中心周波数."""
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    W = np.zeros((len(freqs), n_bands), dtype=np.float32)
    df = freqs[1]
    for b in range(n_bands):
        lo, hi = edges[b], edges[b + 1]
        idx = np.nonzero((freqs >= lo) & (freqs < hi))[0]
        if idx.size:
            W[idx, b] = 1.0 / idx.size
        else:  # 帯域がビン間隔より狭い低域は線形補間
            x = centers[b] / df
            i0 = int(np.floor(x))
            fr = x - i0
            if i0 + 1 < len(freqs):
                W[i0, b] += 1 - fr
                W[i0 + 1, b] += fr
    return W, centers


def chroma_matrix(sr: int, n_fft: int, fmin: float = 80.0, fmax: float = 4000.0) -> np.ndarray:
    """FFT ビン → 12 音名 (C=0) への割り当て (n_bins, 12)."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    M = np.zeros((len(freqs), 12), dtype=np.float32)
    ok = (freqs >= fmin) & (freqs <= fmax)
    midi = 69 + 12 * np.log2(np.maximum(freqs[ok], 1e-6) / 440.0)
    pc = np.round(midi).astype(int) % 12
    M[np.nonzero(ok)[0], pc] = 1.0
    return M


@dataclass
class Track:
    stems: dict[str, np.ndarray]          # パート名 → モノラル float32
    sr: int
    label: str
    spec_stems: dict[str, np.ndarray]     # パート名 → (frames, bands) float16 パワー
    spec: np.ndarray                      # 現在のミックスの (frames, bands) uint8 (0..255 = -100..0 dB)
    spec_hop_s: float
    band_freqs: np.ndarray
    f0_t: np.ndarray                      # f0 の時刻 (秒)
    f0: np.ndarray                        # Hz, NaN=無声
    notes: list[tuple[float, float, float]] = field(default_factory=list)  # (開始, 終了, MIDI 実音高)
    f0_source: str = "FFT 倍音和"
    spec_ref: int = 255                   # spec の表示上の最大値
    chords: list[dict] = field(default_factory=list)
    key: dict | None = None
    beats: list[float] = field(default_factory=list)
    gains: dict[str, float] = field(default_factory=dict)  # 表示に使うパートの重み (再生のミキサーとは別)
    view: str = "all"                     # 表示パート (VIEW_LABELS のキー)
    time_offset: float = 0.0              # 時刻表示用: 元音源での開始位置 (解析結果の区間の開始)
    f0_given: bool = False                # f0 が解析結果 (ボーカル) のものか

    @property
    def view_label(self) -> str:
        return VIEW_LABELS.get(self.view, self.view)

    @property
    def duration(self) -> float:
        n = max((len(v) for v in self.stems.values()), default=0)
        return n / self.sr

    def window(self, center: int, size: int) -> np.ndarray:
        """現在のゲインでミックスした波形 (中心 center, 長さ size)."""
        s = center - size // 2
        out = np.zeros(size, dtype=np.float64)
        for name, y in self.stems.items():
            g = self.gains.get(name, 1.0)
            if g <= 1e-4:
                continue
            a, b = max(0, s), min(len(y), s + size)
            if b > a:
                out[a - s: b - s] += g * y[a:b]
        return out

    def remix_spectrogram(self, gains: dict[str, float] | None = None) -> np.ndarray:
        """パートごとのパワーをゲインの 2 乗で足し合わせてスペクトログラムを作り直す."""
        g = self.gains if gains is None else gains
        P = None
        for name, s in self.spec_stems.items():
            w = g.get(name, 1.0) ** 2
            if w <= 1e-8:
                continue
            P = s.astype(np.float32) * w if P is None else P + s.astype(np.float32) * w
        if P is None:
            return np.zeros_like(next(iter(self.spec_stems.values())), dtype=np.uint8)
        db = 10 * np.log10(P + 1e-12)
        return np.clip((db + SPEC_RANGE_DB) / SPEC_RANGE_DB * 255.0, 0, 255).astype(np.uint8)

    def f0_at(self, t: float) -> float:
        if self.f0_t.size == 0:
            return float("nan")
        i = int(np.searchsorted(self.f0_t, t))
        i = min(max(i, 0), len(self.f0_t) - 1)
        if abs(self.f0_t[i] - t) > 0.05:
            return float("nan")
        return float(self.f0[i])

    def chord_index(self, t: float) -> int:
        for i, c in enumerate(self.chords):
            if c["start"] <= t < c["end"]:
                return i
        return -1


def estimate_f0_from_mag(mag: np.ndarray, sr: int, n_fft: int, fmin: float = 60.0, fmax: float = 1100.0,
                         n_harm: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """振幅スペクトル (frames, bins) から倍音和 (subharmonic 罰則付き) で f0 を推定する."""
    df = sr / n_fft
    cands = fmin * 2.0 ** (np.arange(int(np.log2(fmax / fmin) * 48) + 1) / 48.0)   # 25 cent 刻み
    S = np.sqrt(mag)
    nb = S.shape[1]
    score = np.zeros((S.shape[0], len(cands)), dtype=np.float32)

    def sample(mult: float) -> np.ndarray:
        b = np.clip(mult * cands / df, 0, nb - 2)
        i0 = np.floor(b).astype(int)
        fr = (b - i0).astype(np.float32)
        return S[:, i0] * (1 - fr) + S[:, i0 + 1] * fr

    for k in range(1, n_harm + 1):
        w = 0.86 ** (k - 1)
        score += w * sample(k)
        score -= 0.5 * w * sample(k - 0.5)
    best = np.argmax(score, axis=1)
    rows = np.arange(S.shape[0])
    # 放物線補間で候補間を細かく
    b0 = np.clip(best, 1, len(cands) - 2)
    a, bb, c = score[rows, b0 - 1], score[rows, b0], score[rows, b0 + 1]
    den = a - 2 * bb + c
    with np.errstate(all="ignore"):
        delta = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / den, 0.0)
    delta = np.clip(np.nan_to_num(delta), -0.5, 0.5)
    f0 = fmin * 2.0 ** ((b0 + delta) / 48.0)
    band = (np.arange(nb) * df >= 60) & (np.arange(nb) * df <= 5000)
    mean_s = S[:, band].mean(axis=1) + 1e-9
    conf = score[rows, best] / (mean_s * 3.0)
    return f0, conf


def _stft_bands(y: np.ndarray, sr: int, W: np.ndarray, want_mag_f0: bool, progress: ProgressCB | None,
                label: str, what: str = "スペクトログラムを計算中",
                ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    win = np.hanning(SPEC_NFFT).astype(np.float32)
    n = 1 + len(y) // SPEC_HOP
    ypad = np.pad(y, (SPEC_NFFT // 2, SPEC_NFFT // 2 + SPEC_HOP))
    power = np.zeros((n, W.shape[1]), dtype=np.float16)
    est_f0 = np.full(n, np.nan) if want_mag_f0 else None
    est_conf = np.zeros(n) if want_mag_f0 else None
    rms = np.zeros(n)
    norm = 2.0 / win.sum()
    batch = 1024
    for b0 in range(0, n, batch):
        if progress:
            progress(f"{what} ({label})...", b0 / n)
        b1 = min(n, b0 + batch)
        idx = (np.arange(b0, b1) * SPEC_HOP)[:, None] + np.arange(SPEC_NFFT)[None, :]
        fr = ypad[idx]
        rms[b0:b1] = np.sqrt(np.mean(fr ** 2, axis=1))
        mag = np.abs(np.fft.rfft(fr * win, axis=1)).astype(np.float32) * norm
        bands = mag @ W
        power[b0:b1] = np.minimum(bands.astype(np.float32) ** 2, 60000.0).astype(np.float16)
        if want_mag_f0:
            # 伴奏入りの曲ではベースやキックを拾いやすいので、歌の音域 (80Hz〜) に絞る
            ff, cc = estimate_f0_from_mag(mag, sr, SPEC_NFFT, fmin=80.0)
            est_f0[b0:b1] = ff
            est_conf[b0:b1] = cc
    return power, est_f0, est_conf, rms


def _band_matrix(sr: int) -> tuple[np.ndarray, np.ndarray]:
    return log_band_matrix(sr, SPEC_NFFT, SPEC_FMIN, min(SPEC_FMAX, sr * 0.48), SPEC_BANDS)


def _mix(stems: dict[str, np.ndarray], gains: dict[str, float]) -> np.ndarray:
    """パートの重み付き和 (長さが違うときは長い方に合わせる)."""
    use = [(stems[k], g) for k, g in gains.items() if g > 0 and k in stems]
    out = np.zeros(max((len(y) for y, _g in use), default=0), dtype=np.float32)
    for y, g in use:
        out[: len(y)] += g * y
    return out


def _voiced_f0(est_f0: np.ndarray, est_conf: np.ndarray, rms: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """倍音和による推定値のうち、確からしくて音量も十分なフレームだけを残した f0 軌跡 (時刻, Hz)."""
    level = 20 * np.log10(rms + 1e-9)
    ref = np.percentile(level, 95) if level.size else 0.0
    voiced = (est_conf > 1.6) & (level > ref - 30)
    lf = median_filter(np.log(est_f0), size=7, mode="nearest")
    f0 = np.where(voiced, np.exp(lf), np.nan)
    return np.arange(len(f0)) * SPEC_HOP / sr, f0


def build_track(stems: dict[str, np.ndarray], sr: int, label: str, f0_t: np.ndarray | None = None,
                f0: np.ndarray | None = None, notes: list | None = None, chords: list | None = None,
                key: dict | None = None, beats: list | None = None, gains: dict[str, float] | None = None,
                time_offset: float = 0.0, progress: ProgressCB | None = None) -> Track:
    """stems: {パート名: モノラル波形}. 単一ファイルなら {'mix': y}.

    戻り値は「全パート」を表示する Track (gains で表示に使うパートの重みを変えられる。省略時は全パート =
    旧形式の結果ならミックス)。f0 を渡さなければ音から推定する。パート別の表示は build_views() で作る。
    """
    from ..separation import STEM_LABELS

    stems = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in stems.items()}
    W, centers = _band_matrix(sr)
    all_g = view_gains("all", stems)
    own_f0 = f0 is None
    spec_stems = {}
    est = None
    for name, y in stems.items():
        single = own_f0 and len(stems) == 1
        p, ef, ec, r = _stft_bands(y, sr, W, single, progress, STEM_LABELS.get(name, name))
        spec_stems[name] = p
        if single:
            est = (ef, ec, r)
    if own_f0:
        if est is None:                    # 複数パートで f0 が無い: 全パートを足した音から推定
            est = _stft_bands(_mix(stems, all_g), sr, W, True, progress, VIEW_LABELS["all"],
                              what="ピッチを推定中")[1:]
        f0_t, f0 = _voiced_f0(*est, sr)
        src = "FFT 倍音和 (リアルタイム推定)"
    else:
        src = "解析結果 (pYIN + FFT 精密化)"
    g = dict(all_g) if gains is None else {k: float(gains.get(k, 1.0)) for k in stems}
    tr = Track(stems=stems, sr=sr, label=label, spec_stems=spec_stems, spec=np.zeros((1, SPEC_BANDS), np.uint8),
               spec_hop_s=SPEC_HOP / sr, band_freqs=centers, f0_t=np.asarray(f0_t, dtype=np.float64),
               f0=np.asarray(f0, dtype=np.float64), notes=list(notes or []), f0_source=src,
               chords=list(chords or []), key=key, beats=list(beats or []), gains=g,
               time_offset=float(time_offset), f0_given=not own_f0)
    full = tr.remix_spectrogram(all_g)
    # 表示の明るさ基準: 全パートを足したときの大きい成分 (99.5 パーセンタイル)。
    # パート別の表示も同じ基準なので、パートどうしの音量の差がそのまま明るさの差になる
    tr.spec_ref = int(np.percentile(full[:: max(1, len(full) // 2000)], 99.5)) if len(full) else 255
    tr.spec = full if g == all_g else tr.remix_spectrogram()
    if progress:
        progress("準備完了", 1.0)
    return tr


def build_views(track: Track, progress: ProgressCB | None = None) -> dict[str, Track]:
    """表示パートごとの Track {キー: Track} (VIEW_ORDER の順, 使えるものだけ). 全パートは track 自身.

    パートの波形とスペクトログラムは track と共有する。ピッチ (f0) は
    ボーカル = 解析結果 (track の f0) / フロント楽器・カラオケ = そのパートを足した音から推定。
    音程バー (解析結果のノート) はボーカルと全パートにだけ付ける。
    """
    names = list(track.stems)
    views: dict[str, Track] = {"all": track}
    used = [sorted(view_parts("all", names) or [])]
    W = None
    for key in ("vocals", "karaoke", "front"):   # 中身が同じになる表示は作らない (カラオケを優先)
        parts = view_parts(key, names)
        if parts is None or sorted(parts) in used:
            continue
        used.append(sorted(parts))
        g = {s: (1.0 if s in parts else 0.0) for s in names}
        if key == "vocals" and track.f0_given:
            f0_t, f0, src, notes, given = track.f0_t, track.f0, track.f0_source, list(track.notes), True
        else:
            if W is None:
                W, _c = _band_matrix(track.sr)
            est = _stft_bands(_mix(track.stems, g), track.sr, W, True, progress, VIEW_LABELS[key],
                              what="ピッチを推定中")[1:]
            f0_t, f0 = _voiced_f0(*est, track.sr)
            src, notes, given = "FFT 倍音和 (このパートの音から推定)", [], False
        views[key] = replace(track, view=key, gains=g, spec=track.remix_spectrogram(g),
                             f0_t=np.asarray(f0_t, dtype=np.float64), f0=np.asarray(f0, dtype=np.float64),
                             f0_source=src, notes=notes, f0_given=given)
    if progress:
        progress("準備完了", 1.0)
    return {k: views[k] for k in VIEW_ORDER if k in views}


# ------------------------------------------------------------------------------------ フレーム解析
@dataclass
class VisFrame:
    t: float = 0.0
    level_db: float = -120.0
    bands: np.ndarray = field(default_factory=lambda: np.zeros(1))      # 0..1 (平滑化済み)
    peaks: np.ndarray = field(default_factory=lambda: np.zeros(1))
    band_freqs: np.ndarray = field(default_factory=lambda: np.zeros(1))
    f0: float = float("nan")
    harm_amp: np.ndarray = field(default_factory=lambda: np.zeros(16))  # 線形, 最大=1
    harm_db: np.ndarray = field(default_factory=lambda: np.full(16, -120.0))
    harm_phase: np.ndarray = field(default_factory=lambda: np.zeros(16))  # 基音に対する相対位相
    period_wave: np.ndarray = field(default_factory=lambda: np.zeros(256))  # 倍音から再合成した 1 周期
    scope: np.ndarray = field(default_factory=lambda: np.zeros(1024))    # トリガー済み実波形
    scope_periods: float = 0.0
    voiced: bool = False
    chroma: np.ndarray = field(default_factory=lambda: np.zeros(12))    # 12 音名の強さ (最大=1)
    chord_index: int = -1


class FrameAnalyzer:
    def __init__(self, track: Track, n_bands: int = 112, n_harm: int = 16) -> None:
        self.track = track
        self.n_harm = n_harm
        self.N = 4096
        self.win = np.hanning(self.N)
        self.freqs = np.fft.rfftfreq(self.N, 1.0 / track.sr)
        self.W, self.band_freqs = log_band_matrix(track.sr, self.N, 40.0, min(16000.0, track.sr * 0.48), n_bands)
        self.CM = chroma_matrix(track.sr, self.N)
        self.bands = np.zeros(n_bands)
        self.peaks = np.zeros(n_bands)
        self.chroma = np.zeros(12)
        self.z = np.zeros(n_harm, dtype=complex)  # 平滑化した倍音 (複素振幅)
        self.last_f0 = float("nan")
        self.theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)

    def analyze(self, t: float, detail: int = 2) -> VisFrame:
        """detail 0: 時刻と f0 だけ (スペクトログラム用) / 1: + スペクトラム・クロマ (コード用) /
        2: + 倍音・波形 (全部). 表示しているパネルが必要な分だけ計算する."""
        tr = self.track
        if detail <= 0:
            f0 = tr.f0_at(t)
            return VisFrame(t=t, f0=f0, voiced=bool(np.isfinite(f0)), band_freqs=self.band_freqs,
                            chord_index=tr.chord_index(t))
        sr = tr.sr
        c = int(t * sr)
        fr = tr.window(c, self.N)
        level = 10 * np.log10(np.mean(fr ** 2) + 1e-12)
        mag = np.abs(np.fft.rfft(fr * self.win)) * (2.0 / self.win.sum())
        db = 20 * np.log10(mag @ self.W + 1e-7)
        new = np.clip((db + 75.0) / 70.0, 0.0, 1.0)
        up = new > self.bands
        self.bands = np.where(up, self.bands + (new - self.bands) * 0.75, self.bands + (new - self.bands) * 0.22)
        self.peaks = np.maximum(self.peaks - 0.008, self.bands)
        ch = np.sqrt((mag ** 2) @ self.CM)
        ch = ch / (ch.max() + 1e-9) if level > -70 else np.zeros(12)
        self.chroma = self.chroma * 0.7 + ch * 0.3

        f = VisFrame(t=t, level_db=level, bands=self.bands.copy(), peaks=self.peaks.copy(),
                     band_freqs=self.band_freqs, chroma=self.chroma.copy(), chord_index=tr.chord_index(t))
        f0 = tr.f0_at(t)
        if detail < 2:
            if np.isfinite(f0) and level > -70:
                f.f0, f.voiced = f0, True
            return f
        K = self.n_harm
        if np.isfinite(f0) and level > -70:
            # 倍音解析: 基音 6 周期ぶんの窓 (2048〜8192) を 4 倍ゼロ詰めして FFT
            Nh = int(np.clip(2 ** np.ceil(np.log2(6 * sr / f0)), 2048, 8192))
            seg = tr.window(c, Nh) * np.hanning(Nh)
            nfft = Nh * 4
            # 窓の中心を位相の原点にする: 中心が先頭に来るように回転してからゼロ詰め
            rot = np.concatenate([seg[Nh // 2:], np.zeros(nfft - Nh), seg[: Nh // 2]])
            X = np.fft.rfft(rot)
            df = sr / nfft
            amp = np.zeros(K)
            ph = np.zeros(K)
            ws = np.hanning(Nh).sum()
            for k in range(1, K + 1):
                fk = k * f0
                if fk > sr * 0.45:
                    break
                lo = max(1, int((fk - 0.25 * f0) / df))
                hi = min(len(X) - 2, int((fk + 0.25 * f0) / df) + 1)
                if hi <= lo:
                    continue
                j = lo + int(np.argmax(np.abs(X[lo:hi])))
                amp[k - 1] = np.abs(X[j]) * 2.0 / ws
                ph[k - 1] = np.angle(X[j])
            rel = ph - np.arange(1, K + 1) * ph[0]
            znew = amp * np.exp(1j * rel)
            jump = not np.isfinite(self.last_f0) or abs(np.log2(f0 / self.last_f0)) > 0.25
            self.z = znew if jump else self.z * 0.55 + znew * 0.45
            self.last_f0 = f0
            f.voiced = True
            f.f0 = f0
            # オシロスコープ: 基音の位相 0 の位置でトリガー (3 周期ぶん)
            P = sr / f0
            trig = c - ph[0] / (2 * np.pi) * P
            length = int(round(3 * P))
            start = int(round(trig - P))
            f.scope = tr.window(start + max(length, 2) // 2, max(length, 2))
            f.scope_periods = 3.0
        else:
            self.z *= 0.8
            self.last_f0 = float("nan")
            # 無声時は上向きゼロクロスでトリガー
            seg = tr.window(c, 2048)
            zc = np.nonzero((seg[:-1] < 0) & (seg[1:] >= 0))[0]
            s0 = int(zc[np.argmin(np.abs(zc - 512))]) if zc.size else 512
            f.scope = seg[s0: s0 + 1024] if s0 + 1024 <= len(seg) else seg[:1024]
        amp = np.abs(self.z)
        mx = amp.max()
        f.harm_amp = amp / mx if mx > 1e-9 else amp
        f.harm_db = 20 * np.log10(amp / mx + 1e-6) if mx > 1e-9 else np.full(K, -120.0)
        f.harm_phase = np.angle(self.z)
        kk = np.arange(1, K + 1)[:, None]
        wave = (amp[:, None] * np.cos(kk * self.theta[None, :] + f.harm_phase[:, None])).sum(0)
        wm = np.abs(wave).max()
        f.period_wave = wave / wm if wm > 1e-9 else wave
        return f


def pitch_class_hue(freq: float) -> float:
    """周波数 → 音名 (C=0) に対応する色相 0..1. 同じ音名 (オクターブ違い) は同じ色."""
    if not np.isfinite(freq) or freq <= 0:
        return 0.0
    m = 69 + 12 * np.log2(freq / 440.0)
    return (m % 12) / 12.0
