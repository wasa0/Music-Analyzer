"""ノート単位の「調教」特徴量.

VOCALOID の調教で手作業で付ける要素を、実際の歌声のピッチ/音量/倍音から計測する。

ピッチ (セント, 1半音=100)
  - 入り方 (attack): しゃくり / 上からの入り / ポルタメント / ストレート, 深さと所要時間
  - オーバーシュート: 目標音を越えて行き過ぎる量
  - ビブラート: ピッチ軌跡を FFT して 速さ(Hz)・深さ(±cent)・規則性, かかり始め(遅延)・深さ変化
  - 語尾 (release): フォール(語尾下げ) / 語尾上げ / 予備動作 / ポルタメント, 深さと時間
  - 上ずり・こぶし・ひっくり返り (瞬間的なピッチの跳ね)
  - 音程の正確さ (中心音高のバーからのずれ), 安定度
音量 (DYN): アタックの立ち上がり, クレッシェンド, 語尾の抜き, トレモロ
音色: HNR (息っぽさ=BRE), スペクトル重心 (明るさ=BRI), H1-H2, スペクトル傾斜
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, hilbert, sosfiltfilt

from .notes import Note, _core_slice
from .pitch import midi_to_name

FPS = 100.0  # 解析フレームレート (HOP=10ms)

APPROACH_LABELS = {
    "scoop": "しゃくり (下から)",
    "fall_in": "上から入る",
    "portamento": "ポルタメント",
    "straight": "ストレート",
}
RELEASE_LABELS = {
    "fall": "フォール (語尾下げ)",
    "rise": "語尾上げ",
    "preparation": "予備動作",
    "portamento": "ポルタメント",
    "flat": "そのまま",
}
EVENT_LABELS = {
    "spike_up": "瞬間的な上ずり",
    "spike_down": "瞬間的なピッチ落ち",
    "kobushi": "こぶし/上ずり",
    "dip": "ピッチの落ち込み",
    "flip": "ひっくり返り (大跳躍)",
}


# ---------------------------------------------------------------------------- FFT ユーティリティ
def modulation_spectrum(x: np.ndarray, fs: float = FPS, nfft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """ピッチ (セント) 変動の振幅スペクトル. 返り値の振幅は正弦波の片振幅 (±cent) に正規化."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w, max(nfft, len(x)))
    amp = 2.0 * np.abs(X) / max(w.sum(), 1e-9)
    return np.fft.rfftfreq(max(nfft, len(x)), 1.0 / fs), amp


def detrend_slow(x: np.ndarray, sigma_frames: float = 10.0) -> np.ndarray:
    """1.5Hz 以下のゆっくりした変化を取り除く (ビブラート成分だけ残す)."""
    return x - gaussian_filter1d(x, sigma_frames, mode="nearest")


def analyze_vibrato(c: np.ndarray, fs: float = FPS) -> dict:
    """セント列 c (サステイン部) の FFT からビブラートを判定."""
    res = {"vib": False, "vib_rate_hz": np.nan, "vib_extent_cents": np.nan, "vib_regularity": np.nan}
    if len(c) < int(0.25 * fs):
        return res
    x = detrend_slow(c)
    f, a = modulation_spectrum(x, fs)
    band = (f >= 3.0) & (f <= 10.0)
    if not np.any(band):
        return res
    ib = np.nonzero(band)[0]
    k = ib[np.argmax(a[ib])]
    rate, extent = float(f[k]), float(a[k])
    p = a ** 2
    near = np.abs(f - rate) <= 0.8
    total = p[(f >= 1.0) & (f <= 20.0)].sum()
    reg = float(p[near].sum() / total) if total > 0 else 0.0
    dur = len(c) / fs
    res.update(vib_rate_hz=rate, vib_extent_cents=extent, vib_regularity=reg)
    res["vib"] = bool(extent >= 12.0 and reg >= 0.35 and dur * rate >= 1.8)
    return res


def vibrato_envelope(c: np.ndarray, rate: float, fs: float = FPS) -> tuple[np.ndarray, np.ndarray]:
    """ビブラート帯域を抜き出して (帯域信号, Hilbert 包絡=片振幅) を返す."""
    lo = max(1.5, rate - 1.8)
    hi = min(fs / 2 - 1, rate + 1.8)
    sos = butter(2, [lo, hi], btype="band", fs=fs, output="sos")
    if len(c) <= 3 * (2 * sos.shape[0] + 1) + 1:
        z = np.zeros_like(c)
        return z, z
    xb = sosfiltfilt(sos, c - np.mean(c))
    return xb, np.abs(hilbert(xb))


def _interp_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    m = np.isfinite(x)
    if m.all() or not m.any():
        return np.nan_to_num(x)
    idx = np.arange(len(x))
    x[~m] = np.interp(idx[~m], idx[m], x[m])
    return x


def _linfit_slope(y: np.ndarray, fs: float = FPS) -> float:
    m = np.isfinite(y)
    if m.sum() < 4:
        return np.nan
    t = np.arange(len(y))[m] / fs
    return float(np.polyfit(t, y[m], 1)[0])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


# ------------------------------------------------------------------------------ ノート特徴量
def note_features(notes: list[Note], midi: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    """各 Note.features を計算して埋める. midi はチューニング補正・ギャップ補間済み."""
    rms = curves["rms_db"]
    for i, n in enumerate(notes):
        prev_n = notes[i - 1] if i > 0 else None
        next_n = notes[i + 1] if i + 1 < len(notes) else None
        legato_in = prev_n is not None and n.start - prev_n.end <= 3
        legato_out = next_n is not None and next_n.start - n.end <= 3

        m = _interp_nan(midi[n.start:n.end])
        N = len(m)
        c = (m - n.center) * 100.0              # 自分の中心音高からのずれ
        F: dict = {
            "index": i,
            "start_s": n.start / FPS,
            "end_s": n.end / FPS,
            "dur_ms": N * 10.0,
            "midi": n.midi,
            "name": midi_to_name(n.midi),
            "center_midi": n.center,
            "intonation_cents": (n.center - n.midi) * 100.0,
            "legato_in": legato_in,
            "legato_out": legato_out,
        }

        core = _core_slice(n)
        core_c = c[core.start - n.start: core.stop - n.start]
        pre = analyze_vibrato(core_c) if len(core_c) >= 25 else {"vib": False, "vib_extent_cents": 0.0}
        vib_amp = pre["vib_extent_cents"] if pre.get("vib") else 0.0
        thr = max(35.0, 0.75 * vib_amp + 10.0)
        cg = gaussian_filter1d(c, 3.0, mode="nearest") if N > 3 else c

        # ---- 入り (attack)
        settle = 0
        inside = np.abs(cg) < thr
        for k in range(N):
            if inside[k: k + 4].all():
                settle = k
                break
        else:
            settle = N // 3
        onset_dev = float(np.median(c[: min(3, N)]))
        att_region = c[: settle + 1]
        att_extreme = float(att_region[np.argmax(np.abs(att_region))]) if settle > 0 else onset_dev
        attack_ms = settle * 10.0
        prev_dir = np.sign(prev_n.midi - n.midi) if prev_n is not None else 0
        if legato_in and prev_dir != 0 and np.sign(onset_dev) == prev_dir and abs(onset_dev) >= 30:
            approach = "portamento"
        elif onset_dev <= -40 and attack_ms >= 30:
            approach = "scoop"
        elif onset_dev >= 40 and attack_ms >= 30:
            approach = "fall_in"
        else:
            approach = "straight"
        F.update(approach=approach, approach_label=APPROACH_LABELS[approach], attack_ms=attack_ms,
                 attack_onset_cents=onset_dev, attack_depth_cents=att_extreme)

        # ---- オーバーシュート (近づいてきた方向の反対側へ行き過ぎる量)
        if abs(onset_dev) >= 20:
            direction = -np.sign(onset_dev)
        elif legato_in and prev_dir != 0:
            direction = -prev_dir
        else:
            direction = 0
        overshoot = 0.0
        if direction != 0:
            win = c[settle: min(N, settle + 20)]
            if win.size:
                ov = float(np.max(direction * win))
                overshoot = ov if ov > 15.0 else 0.0
        F["overshoot_cents"] = overshoot

        # ---- 語尾 (release)
        rel_start = N
        for k in range(N - 1, settle, -1):
            if inside[max(settle, k - 3): k + 1].all():
                rel_start = k + 1
                break
        else:
            rel_start = max(settle + 1, N - N // 3)
        end_dev = float(np.median(c[max(0, N - 3):]))
        rel_region = c[rel_start:]
        rel_extreme = float(rel_region[np.argmax(np.abs(rel_region))]) if rel_region.size else end_dev
        release_ms = (N - rel_start) * 10.0
        next_dir = np.sign(next_n.midi - n.midi) if next_n is not None else 0
        if legato_out and next_dir != 0 and np.sign(end_dev) == next_dir and abs(end_dev) >= 30:
            release = "portamento"
        elif legato_out and next_dir != 0 and np.sign(rel_extreme) == -next_dir and abs(rel_extreme) >= 30:
            release = "preparation"
        elif end_dev <= -50 and release_ms >= 30:
            release = "fall"
        elif end_dev >= 50 and release_ms >= 30:
            release = "rise"
        else:
            release = "flat"
        F.update(release=release, release_label=RELEASE_LABELS[release], release_ms=release_ms,
                 release_end_cents=end_dev, release_depth_cents=rel_extreme)

        # ---- ビブラート (サステイン部の FFT)
        sus = c[settle:rel_start]
        vib = analyze_vibrato(sus)
        F.update(vib=vib["vib"], vib_rate_hz=vib["vib_rate_hz"], vib_extent_cents=vib["vib_extent_cents"],
                 vib_regularity=vib["vib_regularity"], vib_onset_ms=np.nan, vib_len_ratio=np.nan,
                 vib_depth_slope=np.nan, vib_rate_slope=np.nan, tremolo_db=np.nan)
        vib_comp = np.zeros(N)
        if vib["vib"]:
            xb, env = vibrato_envelope(c, vib["vib_rate_hz"])
            vib_comp = xb
            if env.any():
                senv = env[settle:rel_start]
                peak = np.percentile(senv, 90) if senv.size else 0
                above = np.nonzero(env[settle:] >= max(8.0, 0.5 * peak))[0]
                onset_idx = settle + int(above[0]) if above.size else settle
                F["vib_onset_ms"] = onset_idx * 10.0
                F["vib_len_ratio"] = (N - onset_idx) / N
                F["vib_depth_slope"] = _linfit_slope(env[onset_idx:rel_start])
                ph = np.unwrap(np.angle(hilbert(xb)))
                inst = np.gradient(ph) * FPS / (2 * np.pi)
                F["vib_rate_slope"] = _linfit_slope(inst[onset_idx:rel_start])
                # 揺れている区間だけで FFT し直して速さ・深さを確定
                #   (揺れ始める前の部分まで窓に入れると深さが過小評価されるため)
                region = c[onset_idx:rel_start]
                if len(region) >= 20:
                    v2 = analyze_vibrato(region)
                    if np.isfinite(v2["vib_rate_hz"]) and v2["vib_extent_cents"] >= 0.5 * vib["vib_extent_cents"]:
                        F["vib_rate_hz"] = v2["vib_rate_hz"]
                        F["vib_regularity"] = max(v2["vib_regularity"], vib["vib_regularity"])
                    F["vib_extent_cents"] = float(np.median(env[onset_idx:rel_start]))
            # トレモロ: 音量 (dB) の同じ周波数の変調
            r_sus = rms[n.start + settle: n.start + rel_start]
            if len(r_sus) >= 25:
                fr, ar = modulation_spectrum(detrend_slow(r_sus))
                near = np.abs(fr - vib["vib_rate_hz"]) <= 0.6
                F["tremolo_db"] = float(ar[near].max()) if near.any() else np.nan

        # ---- 上ずり / こぶし / ひっくり返り
        events = []
        # 瞬間的な跳ね: サステイン部 (入り・語尾の遷移を除く) で 20〜40ms だけ周囲から外れる動き。
        # 150ms 未満の短いノート (速いフェイク・こぶし回し) は音型そのものなので対象外。
        if N >= 15:
            resid = c - median_filter(c, size=9, mode="nearest")
            hnr_n = curves["hnr_db"][n.start:n.end]
            lo_s, hi_s = max(0, settle - 2), min(N, rel_start + 2)
            for sign, kind in ((1, "spike_up"), (-1, "spike_down")):
                for a, b in _runs(sign * resid > 70):
                    if 2 <= b - a <= 4 and a >= lo_s and b <= hi_s and _nanmean(hnr_n[a:b]) >= 5.0:
                        val = resid[a:b].max() if sign > 0 else resid[a:b].min()
                        events.append((kind, a, b, float(val)))
        base = c - vib_comp
        lo_i, hi_i = settle, rel_start
        if hi_i - lo_i >= 6:
            seg = base[lo_i:hi_i]
            for a, b in _runs(seg > 60):
                if 5 <= b - a <= 25:
                    events.append(("kobushi", lo_i + a, lo_i + b, float(seg[a:b].max())))
            for a, b in _runs(seg < -60):
                if 5 <= b - a <= 25:
                    events.append(("dip", lo_i + a, lo_i + b, float(seg[a:b].min())))
        raw = midi[n.start:n.end]
        jumps = np.nonzero(np.abs(np.diff(raw)) > 4.0)[0]
        for j in jumps:
            events.append(("flip", int(j), int(j) + 1, float((raw[j + 1] - raw[j]) * 100)))
        F["events"] = [
            {"type": t, "label": EVENT_LABELS[t], "time_s": (n.start + a) / FPS, "dur_ms": (b - a) * 10.0,
             "cents": v}
            for t, a, b, v in sorted(events, key=lambda e: e[1])
        ]

        # ---- 安定度 (ビブラート除去後の揺れ)
        sus_base = base[settle:rel_start]
        F["stability_cents"] = float(np.std(detrend_slow(sus_base, 5.0))) if sus_base.size >= 5 else np.nan

        # ---- ポルタメント (前ノートからの移行)
        if legato_in and prev_n is not None:
            pf = prev_n.features
            F["transition_ms"] = pf.get("release_ms", 0.0) + (n.start - prev_n.end) * 10.0 + attack_ms
            F["transition_interval"] = n.midi - prev_n.midi
        else:
            F["transition_ms"] = np.nan
            F["transition_interval"] = 0 if prev_n is None else n.midi - prev_n.midi

        # ---- 音量 (DYN)
        r = rms[n.start:n.end]
        r_core = r[core.start - n.start: core.stop - n.start]
        F["dyn_mean_db"] = _nanmean(r_core)
        F["dyn_peak_db"] = float(np.max(r)) if r.size else np.nan
        k_att = min(N, 10)
        F["attack_rise_db"] = float(np.max(r[:k_att]) - r[0]) if N > 1 else 0.0
        F["crescendo_db_s"] = _linfit_slope(r[settle:rel_start]) if rel_start - settle >= 10 else np.nan
        tail = r[max(0, N - 8):]
        F["release_decay_db"] = float(np.mean(tail) - F["dyn_mean_db"]) if tail.size else np.nan

        # ---- 音色 (倍音 FFT 由来)
        sl = slice(n.start + settle, n.start + max(rel_start, settle + 1))
        F["hnr_db"] = _nanmean(curves["hnr_db"][sl])
        F["centroid_hz"] = _nanmean(curves["centroid_hz"][sl])
        F["h1h2_db"] = _nanmean(curves["h1h2_db"][sl])
        F["tilt_db_oct"] = _nanmean(curves["tilt_db_oct"][sl])
        hd = curves["harm_db"][sl].astype(np.float64)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            prof = np.nanmean(hd, axis=0) if hd.size else np.full(curves["harm_db"].shape[1], np.nan)
        F["harmonic_profile_db"] = [float(v - prof[0]) if np.isfinite(v) and np.isfinite(prof[0]) else None
                                    for v in prof[:12]]
        F["settle_idx"] = int(settle)
        F["release_idx"] = int(rel_start)
        n.features = F


# --------------------------------------------------------------------- VOCALOID 風コントロール
def control_lanes(curves: dict[str, np.ndarray], voiced: np.ndarray) -> dict[str, np.ndarray]:
    """DYN / BRE / BRI を 0..127 の目安値に変換 (曲内の相対値)."""
    rms = curves["rms_db"]
    ref = np.percentile(rms[voiced], 97) if voiced.any() else np.max(rms)
    dyn = np.clip((rms - (ref - 42.0)) / 42.0 * 127.0, 0, 127)
    hnr = curves["hnr_db"]
    bre = np.clip((22.0 - hnr) / 27.0 * 127.0, 0, 127)
    cen = curves["centroid_hz"]
    med = np.nanmedian(cen[voiced]) if voiced.any() else np.nanmedian(cen)
    with np.errstate(all="ignore"):
        bri = np.clip(64.0 + 64.0 * np.log2(cen / med), 0, 127)
    # VOCALOID のコントロールカーブのように 40ms 程度で平滑化 (NaN は無視)
    out = {}
    for k, a in (("DYN", dyn), ("BRE", bre), ("BRI", bri)):
        m = np.isfinite(a)
        num = gaussian_filter1d(np.where(m, a, 0.0), 4.0, mode="nearest")
        den = gaussian_filter1d(m.astype(np.float64), 4.0, mode="nearest")
        with np.errstate(all="ignore"):
            s = num / den
        if k != "DYN":
            s[~voiced] = np.nan
        out[k] = s
    return out


# ------------------------------------------------------------------------------------ 全体傾向
def _stat(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None}
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v))}


def summarize_harmony(fs: list[dict], harmony: dict | None) -> dict:
    """キー・テンポ・コードの使用状況と、メロディ (歌) とコードの関係."""
    H: dict = {}
    if not harmony or not harmony.get("chords"):
        return H
    chords = [c for c in harmony["chords"] if c.get("root") is not None]
    H["tempo"] = harmony.get("tempo")
    H["key"] = (harmony.get("key") or {}).get("label")
    H["key_name"] = (harmony.get("key") or {}).get("name")
    H["n_chords"] = len(chords)
    dur: dict[str, float] = {}
    for c in chords:
        dur[c["label"]] = dur.get(c["label"], 0.0) + (c["end"] - c["start"])
    H["top_chords"] = [k for k, _v in sorted(dur.items(), key=lambda kv: -kv[1])[:6]]
    H["progressions"] = [(p["name"], p["count"]) for p in harmony.get("progressions", [])]
    rel = [f for f in fs if f.get("chord_tone") is not None]
    if rel:
        H["chord_tone_ratio"] = sum(bool(f["chord_tone"]) for f in rel) / len(rel)
        roles: dict[str, int] = {}
        for f in rel:
            if not f["chord_tone"]:
                roles[f["chord_role"]] = roles.get(f["chord_role"], 0) + 1
        H["tensions"] = sorted(((k, v / len(rel)) for k, v in roles.items()), key=lambda kv: -kv[1])[:3]
        # ロングトーン (400ms 以上) でどの構成音を伸ばすか
        longs = [f for f in rel if f["dur_ms"] >= 400]
        lr: dict[str, int] = {}
        for f in longs:
            lr[f["chord_role"]] = lr.get(f["chord_role"], 0) + 1
        H["long_roles"] = sorted(((k, v / max(1, len(longs))) for k, v in lr.items()), key=lambda kv: -kv[1])[:3]
    return H


def summarize(notes: list[Note], tuning_offset: float, duration_s: float, harmony: dict | None = None) -> dict:
    fs = [n.features for n in notes]
    S: dict = {"n_notes": len(fs), "duration_s": duration_s}
    S["harmony"] = summarize_harmony(fs, harmony)
    if not fs:
        S["description"] = describe(S)
        return S
    S["tuning_cents"] = tuning_offset * 100.0
    S["a4_hz"] = 440.0 * 2 ** (tuning_offset / 12.0)
    mids = [f["midi"] for f in fs]
    S["range_low"], S["range_high"] = midi_to_name(min(mids)), midi_to_name(max(mids))
    S["range_semitones"] = int(max(mids) - min(mids))
    S["intonation"] = _stat([f["intonation_cents"] for f in fs])
    S["abs_intonation"] = _stat([abs(f["intonation_cents"]) for f in fs])
    S["stability"] = _stat([f["stability_cents"] for f in fs])

    S["approach_counts"] = {k: sum(f["approach"] == k for f in fs) for k in APPROACH_LABELS}
    sc = [f for f in fs if f["approach"] == "scoop"]
    S["scoop_ratio"] = len(sc) / len(fs)
    S["scoop_depth"] = _stat([f["attack_depth_cents"] for f in sc])
    S["scoop_ms"] = _stat([f["attack_ms"] for f in sc])
    fi = [f for f in fs if f["approach"] == "fall_in"]
    S["fall_in_ratio"] = len(fi) / len(fs)
    S["attack_ms_all"] = _stat([f["attack_ms"] for f in fs])
    ov = [f["overshoot_cents"] for f in fs if f["overshoot_cents"] > 0]
    S["overshoot_ratio"] = len(ov) / len(fs)
    S["overshoot"] = _stat(ov)

    pt = [f for f in fs if f["legato_in"] and np.isfinite(f["transition_ms"]) and f["transition_interval"] != 0]
    S["portamento_ms"] = _stat([f["transition_ms"] for f in pt])
    S["portamento_up_ms"] = _stat([f["transition_ms"] for f in pt if f["transition_interval"] > 0])
    S["portamento_down_ms"] = _stat([f["transition_ms"] for f in pt if f["transition_interval"] < 0])
    S["legato_ratio"] = sum(f["legato_in"] for f in fs[1:]) / max(1, len(fs) - 1)

    ends = [f for f in fs if not f["legato_out"]]
    S["release_counts"] = {k: sum(f["release"] == k for f in fs) for k in RELEASE_LABELS}
    fl = [f for f in ends if f["release"] == "fall"]
    S["fall_ratio"] = len(fl) / max(1, len(ends))
    S["fall_depth"] = _stat([f["release_depth_cents"] for f in fl])
    S["fall_ms"] = _stat([f["release_ms"] for f in fl])
    S["rise_ratio"] = sum(f["release"] == "rise" for f in ends) / max(1, len(ends))
    S["preparation_ratio"] = sum(f["release"] == "preparation" for f in fs) / len(fs)

    longs = [f for f in fs if f["dur_ms"] >= 400]
    vb = [f for f in longs if f["vib"]]
    S["n_long_notes"] = len(longs)
    S["vib_ratio"] = len(vb) / max(1, len(longs))
    S["vib_rate"] = _stat([f["vib_rate_hz"] for f in vb])
    S["vib_extent"] = _stat([f["vib_extent_cents"] for f in vb])
    S["vib_onset_ms"] = _stat([f["vib_onset_ms"] for f in vb])
    S["vib_len_ratio"] = _stat([f["vib_len_ratio"] for f in vb])
    S["vib_depth_slope"] = _stat([f["vib_depth_slope"] for f in vb])
    S["vib_rate_slope"] = _stat([f["vib_rate_slope"] for f in vb])
    S["tremolo_db"] = _stat([f["tremolo_db"] for f in vb])

    ev: dict[str, int] = {k: 0 for k in EVENT_LABELS}
    for f in fs:
        for e in f["events"]:
            ev[e["type"]] += 1
    S["event_counts"] = ev
    S["events_per_min"] = {k: v / max(duration_s / 60.0, 1e-6) for k, v in ev.items()}

    S["attack_rise_db"] = _stat([f["attack_rise_db"] for f in fs])
    S["crescendo_db_s"] = _stat([f["crescendo_db_s"] for f in fs if f["dur_ms"] >= 300])
    S["release_decay_db"] = _stat([f["release_decay_db"] for f in ends])
    S["hnr_db"] = _stat([f["hnr_db"] for f in fs])
    S["centroid_hz"] = _stat([f["centroid_hz"] for f in fs])
    S["h1h2_db"] = _stat([f["h1h2_db"] for f in fs])
    S["tilt_db_oct"] = _stat([f["tilt_db_oct"] for f in fs])
    S["description"] = describe(S)
    return S


def _fmt(st: dict, key: str = "mean", digits: int = 0) -> str:
    v = st.get(key) if st else None
    return "-" if v is None else f"{v:.{digits}f}"


def describe(S: dict) -> list[str]:
    """数値サマリーから日本語の歌い方の傾向と VOCALOID 再現のヒントを生成."""
    L: list[str] = []
    H = S.get("harmony") or {}
    if H:
        line = f"曲: キー {H['key']}"
        if H.get("tempo"):
            line += f"・テンポ 約 {H['tempo']:.0f} BPM"
        line += f"。よく使われるコード: {', '.join(H['top_chords'])}。"
        L.append(line)
        if H.get("progressions"):
            L.append("コード進行: " + "、".join(f"{n} ×{c}" for n, c in H["progressions"]) + " を検出。")
    if S.get("n_notes", 0) == 0:
        return L + ["ノートが検出できませんでした。区間・分離方法・音域設定を確認してください。"]
    if H.get("chord_tone_ratio") is not None:
        t = "、".join(f"{k} {v * 100:.0f}%" for k, v in H.get("tensions", []))
        lr = "、".join(f"{k} {v * 100:.0f}%" for k, v in H.get("long_roles", []))
        L.append(f"メロディとコード: 歌の音の {H['chord_tone_ratio'] * 100:.0f}% がコードトーン。"
                 + (f"テンションは {t}。" if t else "") + (f" ロングトーンで伸ばす音: {lr}。" if lr else ""))
    L.append(f"音域: {S['range_low']} 〜 {S['range_high']} ({S['range_semitones']} 半音), "
             f"検出ノート数 {S['n_notes']}。チューニング A4 = {S['a4_hz']:.1f} Hz ({S['tuning_cents']:+.0f} cent)。")
    io = S["intonation"]
    if io["mean"] is not None:
        tend = "やや高め (シャープ)" if io["mean"] > 8 else "やや低め (フラット)" if io["mean"] < -8 else "ほぼ正確"
        L.append(f"音程: 中心音高はバーに対して平均 {io['mean']:+.0f} cent で{tend}。"
                 f"平均絶対ずれ {_fmt(S['abs_intonation'])} cent。")
    # しゃくり
    if S["scoop_depth"]["n"]:
        L.append(f"しゃくり: 全ノートの {S['scoop_ratio'] * 100:.0f}% で下からしゃくり上げ。"
                 f"深さ平均 {_fmt(S['scoop_depth'])} cent / 所要 {_fmt(S['scoop_ms'])} ms。")
    else:
        L.append("しゃくり: ほとんど見られず、ストレートに音に入るタイプ。")
    if S["overshoot"]["n"]:
        L.append(f"オーバーシュート: {S['overshoot_ratio'] * 100:.0f}% のノートで目標音を平均 "
                 f"{_fmt(S['overshoot'])} cent 行き過ぎてから戻る。")
    if S["portamento_ms"]["n"]:
        L.append(f"ポルタメント: レガートな音移りは平均 {_fmt(S['portamento_ms'])} ms "
                 f"(上行 {_fmt(S['portamento_up_ms'])} ms / 下行 {_fmt(S['portamento_down_ms'])} ms)。")
    # 語尾
    if S["fall_depth"]["n"]:
        L.append(f"語尾: フレーズ・音節の終わりの {S['fall_ratio'] * 100:.0f}% でピッチを下げる (フォール)。"
                 f"深さ平均 {_fmt(S['fall_depth'])} cent / {_fmt(S['fall_ms'])} ms。")
    if S["rise_ratio"] > 0.1:
        L.append(f"語尾上げ: 終わりの {S['rise_ratio'] * 100:.0f}% で語尾を持ち上げる。")
    if S["preparation_ratio"] > 0.05:
        L.append(f"予備動作: {S['preparation_ratio'] * 100:.0f}% のノートで次の音と逆方向に一度振ってから移る。")
    # ビブラート
    if S["vib_rate"]["n"]:
        dly = S["vib_onset_ms"]["mean"]
        kind = "後掛け (音を伸ばしてから揺らす)" if dly and dly > 250 else "早掛け (すぐ揺らす)"
        L.append(f"ビブラート: 400ms 以上のロングトーンの {S['vib_ratio'] * 100:.0f}% で使用。"
                 f"速さ {_fmt(S['vib_rate'], digits=1)} Hz, 深さ ±{_fmt(S['vib_extent'])} cent, "
                 f"開始 {_fmt(S['vib_onset_ms'])} ms 後の{kind}。")
        ds = S["vib_depth_slope"]["mean"]
        if ds is not None and abs(ds) > 15:
            L.append(f"  ビブラートは後半にかけて{'深くなる' if ds > 0 else '浅くなる'}傾向 ({ds:+.0f} cent/s)。")
    elif S["n_long_notes"]:
        L.append("ビブラート: ロングトーンでもほとんど揺らさない (ノンビブラート寄り)。")
    ev = S["events_per_min"]
    up = ev["spike_up"] + ev["kobushi"]
    if up > 0.5:
        L.append(f"上ずり・こぶし: 1分あたり約 {up:.1f} 回、一瞬ピッチが跳ね上がる。")
    if S["event_counts"]["flip"]:
        L.append(f"ひっくり返り (大きなピッチ跳躍): {S['event_counts']['flip']} 回 (誤検出の可能性あり)。")
    # 音量・音色
    rd = S["release_decay_db"]["mean"]
    if rd is not None:
        L.append(f"音量: 語尾で平均 {rd:+.1f} dB {'抜く' if rd < -3 else '保つ'}。"
                 f"アタックの立ち上がり {_fmt(S['attack_rise_db'], digits=1)} dB。")
    h = S["hnr_db"]["mean"]
    if h is not None:
        br = "息成分が多い (ブレシー)" if h < 10 else "クリア" if h > 18 else "標準的"
        L.append(f"音色: HNR {h:.1f} dB で{br}。H1-H2 {_fmt(S['h1h2_db'], digits=1)} dB, "
                 f"スペクトル重心 {_fmt(S['centroid_hz'])} Hz。")
    # VOCALOID ヒント
    hints = ["【VOCALOID 等で再現するヒント】"]
    if S["vib_rate"]["n"]:
        lr = S["vib_len_ratio"]["mean"]
        hints.append(f"・ビブラート: Rate ≈ {_fmt(S['vib_rate'], digits=1)} Hz, Depth ≈ ±{_fmt(S['vib_extent'])} cent, "
                     f"ノート長の後ろ {lr * 100 if lr else 0:.0f}% に付ける。")
    if S["scoop_depth"]["n"]:
        hints.append(f"・入り: PIT で {_fmt(S['scoop_depth'])} cent から {_fmt(S['scoop_ms'])} ms かけて目標へ "
                     f"(全体の約 {S['scoop_ratio'] * 100:.0f}% のノート)。")
    if S["portamento_ms"]["n"]:
        hints.append(f"・ポルタメント長 ≈ {_fmt(S['portamento_ms'])} ms。")
    if S["fall_depth"]["n"]:
        hints.append(f"・語尾: 最後の {_fmt(S['fall_ms'])} ms で {_fmt(S['fall_depth'])} cent 下げ、"
                     f"DYN を {rd if rd is not None else 0:+.0f} dB 程度抜く。")
    if len(hints) > 1:
        L.extend(hints)
    return L
