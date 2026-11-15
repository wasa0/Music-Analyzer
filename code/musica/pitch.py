"""f0 (基本周波数) 抽出と FFT による倍音解析.

1. pYIN (librosa) で頑健に f0 軌跡と有声区間を推定
2. 各フレームを FFT し、k*f0 付近の倍音ピークを放物線補間で精密化
   - 倍音周波数 f_k / k の重み付き平均で f0 をセント未満の精度に補正
   - 倍音振幅 H1..Hk (dB), H1-H2, スペクトル傾斜, スペクトル重心
3. FFT による自己相関から HNR (調波対雑音比 = 息っぽさの指標) を算出
"""
from __future__ import annotations

from typing import Callable

import numpy as np

ProgressCB = Callable[[str, float], None]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_JA = ["ド", "ド#", "レ", "レ#", "ミ", "ファ", "ファ#", "ソ", "ソ#", "ラ", "ラ#", "シ"]


def hz_to_midi(f: np.ndarray | float) -> np.ndarray | float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 69.0 + 12.0 * np.log2(np.asarray(f, dtype=np.float64) / 440.0)


def midi_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def midi_to_name(m: float, ja: bool = False) -> str:
    if m is None or not np.isfinite(m):
        return "-"
    n = int(round(m))
    names = NOTE_NAMES_JA if ja else NOTE_NAMES
    return f"{names[n % 12]}{n // 12 - 1}"


# ------------------------------------------------------------------------------------------ pYIN
def _active_intervals(y: np.ndarray, sr: int) -> list[tuple[int, int]]:
    import librosa

    iv = librosa.effects.split(y, top_db=45, frame_length=2048, hop_length=512)
    if len(iv) == 0:
        return []
    merged: list[list[int]] = []
    gap = int(0.3 * sr)
    for s, e in iv:
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = int(e)
        else:
            merged.append([int(s), int(e)])
    pad = int(0.15 * sr)
    return [(max(0, s - pad), min(len(y), e + pad)) for s, e in merged if e - s > int(0.06 * sr)]


def track_f0(y: np.ndarray, sr: int, fmin: float, fmax: float, hop: int,
             progress: ProgressCB | None = None) -> tuple[np.ndarray, np.ndarray]:
    """pYIN で f0 (Hz, 無声は NaN) と有声確率を返す. フレーム i の時刻は i*hop/sr."""
    import librosa

    n_frames = 1 + len(y) // hop
    f0 = np.full(n_frames, np.nan)
    vprob = np.zeros(n_frames)
    intervals = _active_intervals(y, sr)
    total = sum(e - s for s, e in intervals) or 1
    done = 0
    # 最低音の 3 周期が入る最小の 2^n (65Hz@22.05kHz → 1024 = 46ms)。
    # 窓が長いと一瞬の上ずり (数十 ms) が平均化されて消えるため短めにする。
    frame_length = int(2 ** np.ceil(np.log2(3.0 * sr / fmin)))
    for s, e in intervals:
        s_al = (s // hop) * hop
        seg = y[s_al:e]
        if len(seg) < frame_length // 2:
            continue
        if progress:
            progress("ピッチ (f0) 推定中 [pYIN]...", done / total)
        f, _vf, vp = librosa.pyin(seg, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length,
                                  hop_length=hop, center=True)
        i0 = s_al // hop
        i1 = min(n_frames, i0 + len(f))
        f0[i0:i1] = f[: i1 - i0]
        vprob[i0:i1] = vp[: i1 - i0]
        done += e - s
    f0 = _remove_short_islands(f0, min_len=5)
    return f0, vprob


def clean_f0_edges(f0: np.ndarray, rms_db: np.ndarray, max_step_st: float = 1.0,
                   floor_db: float = 30.0, max_trim: int = 4) -> np.ndarray:
    """有声区間の端 (声の出始め/消え際) の信頼できないフレームを除去する.

    - 区間内最大音量より floor_db 以上小さい端のフレーム
    - 隣のフレームから 10ms で max_step_st 半音以上飛んでいる端のフレーム
    """
    out = f0.copy()
    m = hz_to_midi(out)
    v = np.isfinite(out)
    n = len(out)
    i = 0
    while i < n:
        if not v[i]:
            i += 1
            continue
        j = i
        while j < n and v[j]:
            j += 1
        a, b = i, j
        peak = np.max(rms_db[a:b])
        for _ in range(max_trim):
            if b - a <= 5:
                break
            if rms_db[a] < peak - floor_db or abs(m[a] - m[a + 1]) > max_step_st:
                out[a] = np.nan
                a += 1
            else:
                break
        for _ in range(max_trim):
            if b - a <= 5:
                break
            if rms_db[b - 1] < peak - floor_db or abs(m[b - 1] - m[b - 2]) > max_step_st:
                out[b - 1] = np.nan
                b -= 1
            else:
                break
        i = j
    return out


def _remove_short_islands(f0: np.ndarray, min_len: int) -> np.ndarray:
    v = np.isfinite(f0)
    out = f0.copy()
    i = 0
    n = len(v)
    while i < n:
        if v[i]:
            j = i
            while j < n and v[j]:
                j += 1
            if j - i < min_len:
                out[i:j] = np.nan
            i = j
        else:
            i += 1
    return out


# ------------------------------------------------------------------------------- FFT 倍音解析
def harmonic_analysis(y: np.ndarray, sr: int, f0: np.ndarray, hop: int, n_harm: int = 20,
                      win: int = 1024, nfft: int = 8192, batch: int = 384,
                      progress: ProgressCB | None = None) -> dict[str, np.ndarray]:
    """フレームごとに FFT して倍音を測定する.

    Returns: dict(f0=精密化f0, harm_db=(n, K) 倍音振幅dB, hnr_db, centroid_hz, h1h2_db, tilt_db_oct,
                  rms_db)
    """
    n = len(f0)
    K = n_harm
    w = np.hanning(win + 2)[1:-1].astype(np.float64)
    wsum = w.sum()
    df = sr / nfft
    fmax_h = min(0.45 * sr, 9000.0)
    ypad = np.pad(y.astype(np.float64), (win // 2, win // 2 + hop))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    cen_band = freqs <= 8000.0

    # 自己相関用の窓の自己相関 (Boersma 1993 の正規化)
    nac = 2 * win
    rw = np.fft.irfft(np.abs(np.fft.rfft(w, nac)) ** 2, nac)[:win]
    rw = rw / rw[0]

    f0_ref = f0.copy()
    harm_db = np.full((n, K), np.nan, dtype=np.float32)
    hnr = np.full(n, np.nan)
    centroid = np.full(n, np.nan)
    rms_db = np.full(n, -120.0)
    offs_win = np.arange(win)

    max_f0 = np.nanmax(f0) if np.any(np.isfinite(f0)) else 500.0
    rmax = int(np.ceil(0.4 * max_f0 / df)) + 2
    offs = np.arange(-rmax, rmax + 1)

    for b0 in range(0, n, batch):
        if progress and (b0 // batch) % 4 == 0:
            progress("FFT 倍音解析中...", b0 / max(1, n))
        b1 = min(n, b0 + batch)
        rows = np.arange(b0, b1)
        idx = rows[:, None] * hop + offs_win[None, :]
        raw = ypad[idx]
        rms_db[b0:b1] = 10 * np.log10(np.mean(raw ** 2, axis=1) + 1e-12)
        fr = raw * w
        X = np.fft.rfft(fr, nfft, axis=1)
        mag = np.abs(X) * (2.0 / wsum)
        P = mag ** 2
        tot = P[:, cen_band].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            centroid[b0:b1] = np.where(tot > 1e-10, (P[:, cen_band] * freqs[cen_band]).sum(axis=1) / tot, np.nan)

        ff = f0[b0:b1]
        vmask = np.isfinite(ff)
        if not np.any(vmask):
            continue
        vr = np.nonzero(vmask)[0]
        fv = ff[vr]
        lm = np.log(mag[vr] + 1e-12)

        # 1st pass: 低次倍音 (k<=3) を広め (±0.3 f0) に探して f0 を粗く補正。
        #   pYIN は HMM の平滑化で数十 ms の上ずりを丸めてしまうため、FFT で取り戻す。
        est, amp, ok = _pick_harmonics(lm, fv, np.arange(1, 4), lambda k, f: 0.3 * f / df, offs, df, fmax_h)
        f1 = _weighted_f0(est, amp, ok, np.arange(1, 4), fallback=fv)
        # 2nd pass: 全倍音を補正後 f0 の近傍 (狭い窓) で精密に測定
        kk_all = np.arange(1, K + 1)
        est, amp, ok = _pick_harmonics(
            lm, f1, kk_all,
            lambda k, f: np.minimum(0.3 * f, np.maximum(3 * df, 0.03 * k * f)) / df, offs, df, fmax_h)
        kmask = kk_all <= 10
        fr_est = _weighted_f0(est[:, kmask], amp[:, kmask], ok[:, kmask], kk_all[kmask], fallback=f1)
        # 倍音同士の整合性 (f_k/k のばらつき) が小さい時だけ採用
        with np.errstate(all="ignore"):
            dev_c = 1200 * np.log2(est[:, kmask] / fr_est[:, None])
            wk = np.where(ok[:, kmask], 10 ** (amp[:, kmask] / 20) * kk_all[kmask], 0.0)
            wk = np.nan_to_num(wk)
            spread = np.sqrt((wk * np.nan_to_num(dev_c) ** 2).sum(1) / np.maximum(wk.sum(1), 1e-12))
            jump = np.abs(1200 * np.log2(fr_est / fv))
        good = (wk.sum(1) > 0) & (spread < 25) & (jump < 400)
        newf = np.where(good, fr_est, fv)
        f0_ref[b0 + vr] = newf
        harm_db[b0 + vr] = amp.astype(np.float32)

        # HNR (FFT 自己相関, 精密化 f0 の周期で評価)
        Fx = np.fft.rfft(fr[vr], nac, axis=1)
        r = np.fft.irfft(np.abs(Fx) ** 2, nac, axis=1)[:, :win]
        r0 = r[:, :1]
        with np.errstate(invalid="ignore", divide="ignore"):
            rn = r / np.where(r0 > 1e-12, r0, 1.0) / np.maximum(rw[None, :], 1e-3)
        lag = sr / newf
        L0 = np.clip(np.floor(lag * 0.97).astype(int), 1, win - 3)
        L1 = np.clip(np.ceil(lag * 1.03).astype(int) + 1, 2, win - 2)
        best = np.zeros(len(vr))
        span = int(np.max(L1 - L0)) + 1
        lidx = np.clip(L0[:, None] + np.arange(span)[None, :], 1, win - 2)
        lv = np.take_along_axis(rn, lidx, axis=1)
        lv = np.where(lidx <= L1[:, None], lv, -np.inf)
        best = np.clip(lv.max(axis=1), 1e-6, 0.999999)
        hnr[b0 + vr] = np.clip(10 * np.log10(best / (1 - best)), -10, 40)

    f0_ref = _consistency_filter(f0, f0_ref)
    h1h2 = harm_db[:, 0].astype(np.float64) - harm_db[:, 1].astype(np.float64)
    tilt = _spectral_tilt(harm_db)
    if progress:
        progress("FFT 倍音解析完了", 1.0)
    return {
        "f0": f0_ref,
        "harm_db": harm_db,
        "hnr_db": hnr,
        "centroid_hz": centroid,
        "h1h2_db": h1h2,
        "tilt_db_oct": tilt,
        "rms_db": rms_db,
    }


def _consistency_filter(f0_pyin: np.ndarray, f0_ref: np.ndarray, big_cents: float = 40.0) -> np.ndarray:
    """FFT 補正の外れ値を除く.

    分析窓 (46ms) はホップ (10ms) の 4 倍以上あるので、本物のピッチの跳ねは必ず連続した複数フレームに現れる。
    1 フレームだけ大きく補正された値はノイズとみなして pYIN 値に戻し、最後に 3 点メディアンをかける。
    """
    out = f0_ref.copy()
    with np.errstate(all="ignore"):
        d = 1200 * np.log2(f0_ref / f0_pyin)
    big = np.nan_to_num(np.abs(d)) > big_cents
    sgn = np.sign(np.nan_to_num(d))
    prev_ok = np.zeros_like(big)
    next_ok = np.zeros_like(big)
    prev_ok[1:] = big[:-1] & (sgn[:-1] == sgn[1:])
    next_ok[:-1] = big[1:] & (sgn[1:] == sgn[:-1])
    isolated = big & ~(prev_ok | next_ok)
    out[isolated] = f0_pyin[isolated]
    # 有声区間ごとに log f0 の 3 点メディアン
    lf = np.log(out)
    v = np.isfinite(lf)
    sm = lf.copy()
    inner = v[1:-1] & v[:-2] & v[2:]
    stack = np.stack([lf[:-2], lf[1:-1], lf[2:]])
    sm[1:-1][inner] = np.median(stack[:, inner], axis=0)
    return np.exp(sm)


def _pick_harmonics(lm: np.ndarray, f0: np.ndarray, ks: np.ndarray, radius_fn, offs: np.ndarray,
                    df: float, fmax_h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """対数振幅スペクトル lm (frames, bins) から k*f0 付近のピークを放物線補間で求める.

    Returns: est (f_k/k Hz), amp (dB), ok (ノイズフロアより 6dB 以上の有効ピーク)
    """
    nf, nb = lm.shape
    rr = np.arange(nf)
    est = np.full((nf, len(ks)), np.nan)
    amp = np.full((nf, len(ks)), np.nan)
    ok = np.zeros((nf, len(ks)), dtype=bool)
    aoffs = np.abs(offs)[None, :]
    for col, k in enumerate(ks):
        fk = k * f0
        valid = fk < fmax_h
        radius = radius_fn(k, f0)
        c = np.rint(fk / df).astype(int)
        idxk = np.clip(c[:, None] + offs[None, :], 1, nb - 2)
        vals = np.take_along_axis(lm, idxk, axis=1)
        inwin = aoffs <= radius[:, None]
        vals_w = np.where(inwin, vals, -np.inf)
        j = np.argmax(vals_w, axis=1)
        p = idxk[rr, j]
        a, b, g = lm[rr, p - 1], lm[rr, p], lm[rr, p + 1]
        den = a - 2 * b + g
        with np.errstate(invalid="ignore", divide="ignore"):
            delta = np.where(np.abs(den) > 1e-12, 0.5 * (a - g) / den, 0.0)
        delta = np.clip(delta, -0.5, 0.5)
        peak_ln = b - 0.25 * (a - g) * delta
        with np.errstate(all="ignore"):
            floor = np.nanmedian(np.where(inwin, vals, np.nan), axis=1)
        ok[:, col] = valid & ((peak_ln - floor) > np.log(2.0))
        est[:, col] = (p + delta) * df / k
        amp[:, col] = np.where(valid, 20 * peak_ln / np.log(10), np.nan)
    return est, amp, ok


def _weighted_f0(est: np.ndarray, amp: np.ndarray, ok: np.ndarray, ks: np.ndarray,
                 fallback: np.ndarray) -> np.ndarray:
    """f_k/k を (振幅 × k) で重み付け平均 (高次倍音ほど周波数分解能が実質 k 倍良い)."""
    w = np.where(ok, 10 ** (np.nan_to_num(amp, nan=-200) / 20) * ks[None, :], 0.0)
    sw = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (w * np.nan_to_num(est)).sum(axis=1) / sw
    return np.where(sw > 0, f, fallback)


def _spectral_tilt(harm_db: np.ndarray, kmax: int = 10) -> np.ndarray:
    """倍音振幅 (dB) を log2(k) に回帰した傾き = dB/oct (負に大きいほど暗い音色)."""
    k = min(kmax, harm_db.shape[1])
    x = np.log2(np.arange(1, k + 1, dtype=np.float64))
    Y = harm_db[:, :k].astype(np.float64)
    m = np.isfinite(Y)
    cnt = m.sum(axis=1)
    X = np.where(m, x[None, :], 0.0)
    Yz = np.where(m, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = X.sum(1) / cnt
        my = Yz.sum(1) / cnt
        cov = (np.where(m, (X - mx[:, None]) * (Yz - my[:, None]), 0.0)).sum(1)
        var = (np.where(m, (X - mx[:, None]) ** 2, 0.0)).sum(1)
        slope = cov / var
    slope[cnt < 4] = np.nan
    return slope


def frame_spectrum(y: np.ndarray, sr: int, center: int, win: int = 2048, nfft: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """1フレームの FFT 振幅スペクトル (dBFS). GUI のノート詳細表示用."""
    s = center - win // 2
    seg = np.zeros(win)
    a, b = max(0, s), min(len(y), s + win)
    if b > a:
        seg[a - s: b - s] = y[a:b]
    w = np.hanning(win)
    X = np.fft.rfft(seg * w, nfft)
    mag = np.abs(X) * 2.0 / w.sum()
    return np.fft.rfftfreq(nfft, 1.0 / sr), 20 * np.log10(mag + 1e-9)


def average_spectrum(y: np.ndarray, sr: int, s0: int, s1: int, win: int = 2048, nfft: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """区間 [s0, s1) の平均パワースペクトル (dBFS)."""
    w = np.hanning(win)
    hop = win // 4
    acc = None
    cnt = 0
    for c in range(s0 + win // 2, max(s0 + win // 2 + 1, s1 - win // 2), hop):
        seg = y[c - win // 2: c + win // 2]
        if len(seg) < win:
            break
        X = np.abs(np.fft.rfft(seg * w, nfft)) * 2.0 / w.sum()
        acc = X ** 2 if acc is None else acc + X ** 2
        cnt += 1
    if acc is None:
        return frame_spectrum(y, sr, (s0 + s1) // 2, win, nfft)
    return np.fft.rfftfreq(nfft, 1.0 / sr), 10 * np.log10(acc / cnt + 1e-18)
