"""コード (和音) 認識・キー推定・テンポ/拍.

手順
1. 拍 (ビート) を検出してテンポを求める (ドラムパートがあればそれを使う)
2. 伴奏 (ベース/ギター/ピアノ/その他) から CQT クロマ (12 音名ごとの強さ) を計算し、拍ごとに集約
3. 各拍のクロマを和音テンプレート (倍音を考慮) と照合し、ベースパートの最低音で根音・転回形を補強
4. 隠れマルコフモデル風の Viterbi で、頻繁にコードが入れ替わらないよう平滑化
5. Krumhansl のキープロファイルでキーを推定し、度数 (IVM7 など) と有名なコード進行を求める
"""
from __future__ import annotations

from typing import Callable

import numpy as np

ProgressCB = Callable[[str, float], None]

SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
DEGREES = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]

# 和音の種類: (構成音 [半音], 事前ペナルティ). 単純な三和音ほど選ばれやすい
QUALITIES: dict[str, tuple[tuple[int, ...], float]] = {
    "": ((0, 4, 7), 0.0),
    "m": ((0, 3, 7), 0.0),
    "7": ((0, 4, 7, 10), 0.012),
    "M7": ((0, 4, 7, 11), 0.012),
    "m7": ((0, 3, 7, 10), 0.012),
    "6": ((0, 4, 7, 9), 0.05),
    "add9": ((0, 2, 4, 7), 0.05),
    "sus4": ((0, 5, 7), 0.07),
    "sus2": ((0, 2, 7), 0.08),
    "dim": ((0, 3, 6), 0.07),
    "m7(b5)": ((0, 3, 6, 10), 0.05),
    "aug": ((0, 4, 8), 0.09),
}
SWITCH_PENALTY = 0.07     # コードを切り替えるコスト (大きいほど安定、小さいほど細かく追従)
BASS_ROOT_BONUS = 0.05    # ベースの最低音が根音のときのボーナス
BASS_TONE_BONUS = 0.03    # ベースの最低音が構成音のときのボーナス
MINOR_LIKE = {"m", "m7", "dim", "m7(b5)"}

INTERVAL_ROLES = {0: "根音 (R)", 1: "♭9th", 2: "9th", 3: "短3度", 4: "長3度", 5: "11th (4度)", 6: "♯11th / ♭5",
                  7: "5度", 8: "♭13th / ♯5", 9: "13th (6度)", 10: "短7度", 11: "長7度"}

# Krumhansl-Kessler のキープロファイル
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# 有名なコード進行 (相対長調の度数, 短調系なら True / None=どちらでも)
PROGRESSIONS = [
    ("王道進行 (IV-V-IIIm-VIm)", [(5, False), (7, False), (4, None), (9, True)]),
    ("小室進行 (VIm-IV-V-I)", [(9, True), (5, False), (7, False), (0, False)]),
    ("カノン進行 (I-V-VIm-IIIm)", [(0, False), (7, False), (9, True), (4, True)]),
    ("丸サ進行 (IVM7-III7-VIm7)", [(5, False), (4, False), (9, True)]),
    ("Axis 進行 (I-V-VIm-IV)", [(0, False), (7, False), (9, True), (5, False)]),
    ("50年代進行 (I-VIm-IV-V)", [(0, False), (9, True), (5, False), (7, False)]),
    ("ツーファイブワン (IIm-V-I)", [(2, True), (7, False), (0, False)]),
]


def _templates() -> tuple[np.ndarray, list[tuple[int, str]]]:
    """倍音を含めた和音テンプレート (n_chords, 12) と (root, quality) の対応表."""
    rows, keys = [], []
    harm = [(0, 1.0), (7, 0.3), (4, 0.12)]       # 第 3 倍音 = 5度上, 第 5 倍音 = 長3度上
    for root in range(12):
        for q, (ivs, _pen) in QUALITIES.items():
            v = np.zeros(12)
            for k, iv in enumerate(ivs):
                w = 1.15 if k == 0 else 1.0
                for h, hw in harm:
                    v[(root + iv + h) % 12] += w * hw
            rows.append(v / np.linalg.norm(v))
            keys.append((root, q))
    return np.array(rows), keys


def note_name(pc: int, flats: bool) -> str:
    return (FLAT if flats else SHARP)[pc % 12]


def chord_label(root: int | None, quality: str, bass: int | None, flats: bool) -> str:
    if root is None:
        return "N.C."
    s = note_name(root, flats) + quality
    if bass is not None and bass != root:
        s += "/" + note_name(bass, flats)
    return s


def degree_label(root: int | None, quality: str, tonic_major: int) -> str:
    if root is None:
        return "-"
    return DEGREES[(root - tonic_major) % 12] + quality


def estimate_key(chroma_sum: np.ndarray) -> dict:
    x = chroma_sum / (np.linalg.norm(chroma_sum) + 1e-9)
    best = (-2.0, 0, "major")
    scores = []
    for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        for tonic in range(12):
            p = np.roll(prof, tonic)
            r = float(np.corrcoef(x, p)[0, 1])
            scores.append(r)
            if r > best[0]:
                best = (r, tonic, mode)
    r, tonic, mode = best
    rel_major = tonic if mode == "major" else (tonic + 3) % 12
    flats = rel_major in (5, 10, 3, 8, 1, 6)        # F Bb Eb Ab Db Gb
    srt = sorted(scores, reverse=True)
    name = note_name(tonic, flats) + ("" if mode == "major" else "m")
    return {"tonic": tonic, "mode": mode, "name": name,
            "label": f"{note_name(tonic, flats)} {'メジャー (長調)' if mode == 'major' else 'マイナー (短調)'}",
            "relative_major": rel_major, "flats": flats, "confidence": r, "margin": srt[0] - srt[1]}


def _sync(feat: np.ndarray, bounds: list[int]) -> np.ndarray:
    out = np.zeros((feat.shape[0], len(bounds) - 1))
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if b > a:
            out[:, i] = np.median(feat[:, a:b], axis=1)
    return out


def analyze_harmony(harmonic: np.ndarray, sr: int, bass: np.ndarray | None = None,
                    rhythm: np.ndarray | None = None, progress: ProgressCB | None = None,
                    hop: int = 512) -> dict:
    """伴奏 (モノラル) からテンポ・拍・キー・コード列を求める.

    harmonic: 音程のある伴奏 (ベース+ギター+ピアノ+その他 など)
    bass:     ベースパート (無ければ harmonic の低域を使う)
    rhythm:   拍検出用の信号 (ドラム or ミックス)
    """
    import librosa

    dur = len(harmonic) / sr
    if progress:
        progress("テンポと拍を検出中...", 0.05)
    src = rhythm if rhythm is not None else harmonic
    onset = librosa.onset.onset_strength(y=src, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, trim=False)
    tempo = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    beat_frames = np.asarray(beat_frames, dtype=int)

    if progress:
        progress("クロマ (音名ごとの強さ) を計算中...", 0.25)
    h = librosa.effects.harmonic(harmonic, margin=3.0)
    # 和音の中身は C3 以上で見る (ベースの基音が強すぎてクロマを支配するのを防ぐ)。
    # 対数圧縮した CQT を 12 音名に畳み込む
    bpo, n_oct = 36, 4
    fmin = librosa.note_to_hz("C3")
    tuning = librosa.estimate_tuning(y=h, sr=sr, bins_per_octave=bpo)
    cq = np.abs(librosa.cqt(h, sr=sr, hop_length=hop, fmin=fmin, n_bins=bpo * n_oct, bins_per_octave=bpo,
                            tuning=tuning))
    cq = np.log1p(100.0 * cq / (cq.max() + 1e-9))
    fold = librosa.filters.cq_to_chroma(n_input=bpo * n_oct, bins_per_octave=bpo, n_chroma=12,
                                        fmin=fmin * 2.0 ** (tuning / 12.0))
    chroma = fold @ cq
    n_frames = chroma.shape[1]
    energy = librosa.feature.rms(y=h, hop_length=hop)[0][:n_frames]
    if progress:
        progress("ベース音を解析中...", 0.55)
    bsrc = bass if bass is not None else harmonic
    bchroma = librosa.feature.chroma_cqt(y=bsrc, sr=sr, hop_length=hop, bins_per_octave=36, tuning=tuning,
                                         fmin=librosa.note_to_hz("E1"), n_octaves=3)[:, :n_frames]
    benergy = librosa.feature.rms(y=bsrc, hop_length=hop)[0][:n_frames]

    # 拍ごとに集約 (拍が取れない場合は 0.5 秒刻み)
    if len(beat_frames) >= 4:
        bounds = sorted(set([0, *[int(b) for b in beat_frames if 0 < b < n_frames], n_frames]))
    else:
        step = max(1, int(0.5 * sr / hop))
        bounds = list(range(0, n_frames, step)) + [n_frames]
    C = _sync(chroma, bounds)                                # (12, nb)
    B = _sync(bchroma, bounds)
    E = np.array([energy[a:b].mean() if b > a else 0 for a, b in zip(bounds[:-1], bounds[1:])])
    BE = np.array([benergy[a:b].mean() if b > a else 0 for a, b in zip(bounds[:-1], bounds[1:])])
    nb = C.shape[1]

    if progress:
        progress("コードを推定中...", 0.75)
    T, keys = _templates()
    pen = np.array([QUALITIES[q][1] for _r, q in keys])
    Cn = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)
    S = T @ Cn - pen[:, None]                               # (n_chords, nb) 類似度
    # ベースの最低音が根音ならボーナス, 構成音なら小さなボーナス
    Bn = B / (B.max(axis=0, keepdims=True) + 1e-9)
    bass_ok = BE > (np.median(BE[BE > 0]) * 0.25 if np.any(BE > 0) else 0)
    for i, (root, q) in enumerate(keys):
        tones = [(root + iv) % 12 for iv in QUALITIES[q][0]]
        S[i] += np.where(bass_ok, BASS_ROOT_BONUS * Bn[root] + BASS_TONE_BONUS * Bn[tones].max(axis=0), 0.0)
    # 無音・ドラムだけの区間は N.C. (コード無し)
    ref_e = np.percentile(E[E > 0], 75) if np.any(E > 0) else 1.0
    quiet = E < ref_e * 0.08
    n_states = len(keys) + 1
    score = np.vstack([S, np.where(quiet, 2.0, 0.35)[None, :]])   # 最後の行が N.C.
    score[:-1, quiet] -= 1.0

    # Viterbi (コードの切り替えにペナルティ)
    lam = SWITCH_PENALTY
    D = score[:, 0].copy()
    back = np.zeros((nb, n_states), dtype=np.int32)
    for t in range(1, nb):
        best = int(np.argmax(D))
        switch = D[best] - lam
        stay = D
        take_switch = switch > stay
        back[t] = np.where(take_switch, best, np.arange(n_states))
        D = np.where(take_switch, switch, stay) + score[:, t]
    path = np.zeros(nb, dtype=np.int32)
    path[-1] = int(np.argmax(D))
    for t in range(nb - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    key = estimate_key((Cn * np.maximum(E, 0)[None, :]).sum(axis=1))
    flats = key["flats"]
    times = librosa.frames_to_time(np.array(bounds), sr=sr, hop_length=hop)
    times[-1] = dur
    # 拍ごとのベース音 (分数コード用): 根音以外の構成音がベースのときだけ記録
    beat_bass = np.full(nb, -1)
    for t in range(nb):
        st = int(path[t])
        if st < len(keys) and bass_ok[t]:
            root, q = keys[st]
            bp = int(np.argmax(B[:, t]))
            tones = {(root + iv) % 12 for iv in QUALITIES[q][0]}
            if bp != root and bp in tones:
                beat_bass[t] = bp
    chords = []
    i = 0
    while i < nb:
        j = i
        # 同じコードでもベースが変わったら (C → C/E) 区切る
        while j + 1 < nb and path[j + 1] == path[i] and beat_bass[j + 1] == beat_bass[i]:
            j += 1
        st = int(path[i])
        if st == len(keys):
            root, q, bass_pc = None, "", None
        else:
            root, q = keys[st]
            bass_pc = int(beat_bass[i]) if beat_bass[i] >= 0 else None
        conf = float(np.mean(score[st, i: j + 1])) if st < len(keys) else 0.0
        chords.append({
            "start": float(times[i]), "end": float(times[j + 1]), "root": root, "quality": q, "bass": bass_pc,
            "label": chord_label(root, q, bass_pc, flats),
            "degree": degree_label(root, q, key["relative_major"]) + (
                "/" + DEGREES[(bass_pc - key["relative_major"]) % 12] if bass_pc is not None else ""),
            "score": conf,
        })
        i = j + 1
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop).tolist()
    res = {"tempo": tempo, "beats": beats, "key": key, "chords": chords,
           "progressions": find_progressions(chords, key["relative_major"])}
    if progress:
        progress("コード解析完了", 1.0)
    return res


def find_progressions(chords: list[dict], tonic_major: int) -> list[dict]:
    seq = [(c["start"], (c["root"] - tonic_major) % 12, c["quality"] in MINOR_LIKE)
           for c in chords if c["root"] is not None]
    # 同じコードの連続を 1 つにまとめる
    comp = []
    for s in seq:
        if not comp or comp[-1][1:] != s[1:]:
            comp.append(s)
    found = []
    for name, pat in PROGRESSIONS:
        times = []
        for i in range(len(comp) - len(pat) + 1):
            if all(comp[i + k][1] == d and (m is None or comp[i + k][2] == m) for k, (d, m) in enumerate(pat)):
                times.append(comp[i][0])
        if times:
            found.append({"name": name, "count": len(times), "times": times})
    return found


def chord_at(chords: list[dict], t: float) -> dict | None:
    for c in chords:
        if c["start"] <= t < c["end"]:
            return c
    return None


def relate_note(midi: int, chord: dict | None) -> dict:
    """メロディの音がコードに対して何度か (コードトーン / テンション)."""
    if chord is None or chord.get("root") is None:
        return {"chord": "N.C." if chord else "-", "interval": None, "role": "-", "chord_tone": None}
    iv = (int(midi) - chord["root"]) % 12
    tones = {i % 12 for i in QUALITIES[chord["quality"]][0]}
    return {"chord": chord["label"], "interval": iv, "role": INTERVAL_ROLES[iv], "chord_tone": iv in tones}
