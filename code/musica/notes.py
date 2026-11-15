"""ピッチ軌跡からカラオケ風の音程バー (ノート) を切り出す.

手順
1. 全体のチューニングずれ (A=440Hz からのセント) を推定して補正
2. 有声区間ごとに、平滑化したピッチを半音状態の Viterbi で区分一定に量子化
   (状態遷移ペナルティによりビブラートや一瞬のしゃくりではノートが割れない)
3. 短すぎる区間 (遷移部分) を隣のノートへ吸収
4. 同じ音高が続く場合は音量の谷 / 立ち上がり (新しい音節) で分割
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


@dataclass
class Note:
    start: int            # 開始フレーム (含む)
    end: int              # 終了フレーム (含まない)
    midi: int             # 音程バーの音高 (チューニング補正後の整数 MIDI)
    center: float = 0.0   # 実際に歌っている中心音高 (MIDI, 小数)
    features: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start


def estimate_tuning(midi: np.ndarray) -> float:
    """安定区間の半音の端数を円周平均してチューニングずれ (半音単位, -0.5..0.5) を求める."""
    m = midi[np.isfinite(midi)]
    if m.size < 20:
        return 0.0
    g = np.abs(np.gradient(midi))
    stable = np.isfinite(midi) & (np.nan_to_num(g, nan=9) < 0.03)
    m = midi[stable] if stable.sum() > 50 else m
    ang = 2 * np.pi * (m - np.round(m))
    off = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))) / (2 * np.pi)
    return float(off)


def fill_short_gaps(midi: np.ndarray, max_gap: int = 4, max_jump: float = 2.5) -> np.ndarray:
    """短い無声ギャップを線形補間で埋める (子音や pYIN の取りこぼし対策)."""
    out = midi.copy()
    v = np.isfinite(midi)
    n = len(midi)
    i = 0
    while i < n:
        if not v[i]:
            j = i
            while j < n and not v[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap and abs(midi[j] - midi[i - 1]) <= max_jump:
                out[i:j] = np.linspace(midi[i - 1], midi[j], j - i + 2)[1:-1]
            i = j
        else:
            i += 1
    return out


def voiced_runs(midi: np.ndarray, min_len: int = 8) -> list[tuple[int, int]]:
    v = np.isfinite(midi)
    runs = []
    n = len(v)
    i = 0
    while i < n:
        if v[i]:
            j = i
            while j < n and v[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def viterbi_quantize(x: np.ndarray, lam: float = 22.0, sigma: float = 0.35, clip: float = 9.0,
                     jump_cost: float = 0.4) -> np.ndarray:
    """x (半音) を整数半音の区分一定列に量子化する."""
    lo = int(np.floor(np.min(x))) - 1
    hi = int(np.ceil(np.max(x))) + 1
    states = np.arange(lo, hi + 1)
    S = len(states)
    T = len(x)
    E = np.minimum(((x[:, None] - states[None, :]) / sigma) ** 2, clip)
    dist = np.abs(states[:, None] - states[None, :])
    trans = np.where(dist > 0, lam + jump_cost * dist, 0.0)  # [prev, cur]
    D = E[0].copy()
    back = np.zeros((T, S), dtype=np.int32)
    for t in range(1, T):
        tot = D[:, None] + trans
        bp = np.argmin(tot, axis=0)
        D = tot[bp, np.arange(S)] + E[t]
        back[t] = bp
    path = np.empty(T, dtype=np.int32)
    path[-1] = int(np.argmin(D))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return states[path]


def _segments_from_path(path: np.ndarray, offset: int) -> list[list[int]]:
    segs = []
    s = 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[s]:
            segs.append([offset + s, offset + i, int(path[s])])
            s = i
    return segs


def _merge_short(segs: list[list[int]], x_full: np.ndarray, min_len: int) -> list[list[int]]:
    segs = [list(s) for s in segs]
    while len(segs) > 1:
        lens = [s[1] - s[0] for s in segs]
        i = int(np.argmin(lens))
        if lens[i] >= min_len:
            break
        mean_p = float(np.nanmean(x_full[segs[i][0]:segs[i][1]]))
        cand = []
        if i > 0:
            cand.append((abs(segs[i - 1][2] - mean_p), i - 1))
        if i < len(segs) - 1:
            cand.append((abs(segs[i + 1][2] - mean_p), i + 1))
        _, j = min(cand)
        if j < i:
            segs[j][1] = segs[i][1]
        else:
            segs[j][0] = segs[i][0]
        del segs[i]
        # 吸収の結果、同じ音高のノートが隣接したら結合
        k = 0
        while k < len(segs) - 1:
            if segs[k][2] == segs[k + 1][2] and segs[k][1] == segs[k + 1][0]:
                segs[k][1] = segs[k + 1][1]
                del segs[k + 1]
            else:
                k += 1
    return segs


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _is_transition(x: np.ndarray, plateau_slope: float = 0.03, min_plateau: int = 5) -> bool:
    """しゃくり / フォール / 経過音のように、平坦部が無く一方向に動くだけの区間か."""
    if len(x) < 3:
        return True
    d = np.diff(x)
    net = abs(x[-1] - x[0])
    tv = np.abs(d).sum() + 1e-9
    plateau = _longest_run(np.abs(np.gradient(x)) < plateau_slope)
    return net >= 0.5 and net / tv >= 0.7 and plateau < min_plateau


def _absorb_transitions(segs: list[list[int]], xm: np.ndarray, off: int, max_len: int = 30) -> list[list[int]]:
    """遷移だけの短い区間を隣のノートの入り (アタック) / 語尾 (リリース) に吸収する."""
    segs = [list(s) for s in segs]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (a, b, _st) in enumerate(segs):
            if b - a > max_len or not _is_transition(xm[a - off: b - off]):
                continue
            if i == 0:
                segs[1][0] = a                       # 入りのしゃくり → 次のノートへ
            elif i == len(segs) - 1:
                segs[i - 1][1] = b                   # 語尾のフォール → 前のノートへ
            else:                                    # 経過部分 → 2 音の中間点で分ける
                mid = 0.5 * (segs[i - 1][2] + segs[i + 1][2])
                cut = a + int(np.argmin(np.abs(xm[a - off: b - off] - mid)))
                segs[i - 1][1] = cut
                segs[i + 1][0] = cut
            del segs[i]
            changed = True
            break
        k = 0
        while k < len(segs) - 1:
            if segs[k][2] == segs[k + 1][2] and segs[k][1] == segs[k + 1][0]:
                segs[k][1] = segs[k + 1][1]
                del segs[k + 1]
            else:
                k += 1
    return segs


def _energy_splits(seg: list[int], rms_db: np.ndarray, onset_env: np.ndarray | None,
                   min_part: int) -> list[int]:
    """同音高で続く音節の境界 (音量の谷 or 強い立ち上がり) を探す."""
    s, e, _ = seg
    if e - s < 2 * min_part:
        return []
    r = median_filter(rms_db[s:e], size=3, mode="nearest")
    cuts: list[int] = []
    W = 20
    for i in range(min_part, (e - s) - min_part):
        if r[i] > r[i - 1] or r[i] > r[i + 1]:
            continue
        left = r[max(0, i - W): i].max()
        right = r[i + 1: i + 1 + W].max()
        if left - r[i] >= 5.0 and right - r[i] >= 5.0:
            if not cuts or (s + i) - cuts[-1] >= min_part:
                cuts.append(s + i)
    if onset_env is not None and len(onset_env) >= e:
        env = onset_env[s:e]
        thr = np.median(onset_env[onset_env > 0]) * 4.0 if np.any(onset_env > 0) else np.inf
        for i in range(min_part, (e - s) - min_part):
            if env[i] >= thr and env[i] >= env[i - 1] and env[i] >= env[i + 1]:
                rise = r[i: i + 5].mean() - r[max(0, i - 5): i].mean()
                if rise >= 3.0 and all(abs((s + i) - c) >= min_part for c in cuts):
                    cuts.append(s + i)
    return sorted(cuts)


def segment_notes(midi_adj: np.ndarray, rms_db: np.ndarray, onset_env: np.ndarray | None = None,
                  min_note: int = 9) -> tuple[list[Note], np.ndarray]:
    """チューニング補正済み MIDI 列からノートを抽出. (notes, 補間後の MIDI 列) を返す."""
    filled = fill_short_gaps(midi_adj)
    notes: list[Note] = []
    for s, e in voiced_runs(filled, min_len=min_note):
        x = filled[s:e]
        xm = median_filter(x, size=5, mode="nearest")
        xs = gaussian_filter1d(xm, sigma=4.0, mode="nearest")
        path = viterbi_quantize(xs)
        segs = _merge_short(_segments_from_path(path, s), filled, min_note)
        segs = _absorb_transitions(segs, xm, s)
        for seg in segs:
            cuts = _energy_splits(seg, rms_db, onset_env, min_part=max(min_note, 12))
            bounds = [seg[0], *cuts, seg[1]]
            for a, b in zip(bounds[:-1], bounds[1:]):
                notes.append(Note(start=a, end=b, midi=seg[2]))
    # 各ノートの中心音高を計算し、バー音高を中心に合わせ直す
    for n in notes:
        core = _core_slice(n)
        n.center = float(np.nanmedian(filled[core]))
        if np.isfinite(n.center):
            n.midi = int(np.round(n.center))
    return notes, filled


def _core_slice(n: Note) -> slice:
    L = n.length
    a = n.start + int(L * 0.2)
    b = n.end - int(L * 0.2)
    if b - a < 3:
        a, b = n.start, n.end
    return slice(a, b)
