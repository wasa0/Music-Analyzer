"""解析パイプライン: 区間切り出し → 音源分離 (全パート) → f0/倍音 → ノート → 特徴量 → コード → 保存."""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import config
from .audio_io import load_wav, resample, save_wav, to_mono
from .chords import analyze_harmony, chord_at, relate_note
from .features import control_lanes, note_features, summarize
from .notes import Note, estimate_tuning, segment_notes
from .pitch import clean_f0_edges, harmonic_analysis, hz_to_midi, track_f0
from .separation import HARMONIC_STEMS, separate, sort_stems

ProgressCB = Callable[[str, float], None]

CURVE_KEYS = ("rms_db", "hnr_db", "centroid_hz", "h1h2_db", "tilt_db_oct", "harm_db")
SEPARATION_VERSION = 2


@dataclass
class AnalysisResult:
    meta: dict
    f0_hz: np.ndarray            # 精密化 f0 (Hz, 無声 NaN)
    midi_raw: np.ndarray         # チューニング補正済み MIDI (無声 NaN)
    midi: np.ndarray             # 上記の短ギャップ補間版 (特徴量計算に使用)
    vprob: np.ndarray
    curves: dict[str, np.ndarray]
    lanes: dict[str, np.ndarray]
    notes: list[Note]
    summary: dict
    folder: Path
    harmony: dict = field(default_factory=dict)   # tempo / beats / key / chords / progressions
    _vocal_cache: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ 便利プロパティ
    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.f0_hz)) * config.HOP_SEC

    @property
    def stems(self) -> list[str]:
        names = self.meta.get("stems")
        if names:
            return sort_stems(n for n in names if self.stem_path(n).exists())
        return ["vocals"] if self.vocal_path.exists() else []

    def stem_path(self, name: str) -> Path:
        p = self.folder / "stems" / f"{name}.wav"
        if name == "vocals" and not p.exists():
            return self.folder / "vocals.wav"          # 旧バージョンの結果
        return p

    @property
    def vocal_path(self) -> Path:
        return self.stem_path("vocals")

    @property
    def is_legacy(self) -> bool:
        """楽器分離に対応する前のバージョン (0.1) で作った結果か (ボーカルとミックスだけがある)."""
        return not self.meta.get("stems")

    def playback_parts(self) -> dict[str, Path]:
        """プレーヤー / ミキサー / ビジュアライザーで使うパート.

        旧形式の結果は伴奏が分離されていないので「ミックス」と「ボーカル」の 2 つにする。
        """
        if self.is_legacy:
            parts = {}
            if self.mix_path.exists():
                parts["mix"] = self.mix_path
            if self.vocal_path.exists():
                parts["vocals"] = self.vocal_path
            return parts
        return {s: self.stem_path(s) for s in self.stems} or {"mix": self.mix_path}

    @property
    def mix_path(self) -> Path:
        return self.folder / "mix.wav"

    @property
    def chords(self) -> list[dict]:
        return self.harmony.get("chords", []) if self.harmony else []

    @property
    def offset_s(self) -> float:
        """元音源での区間開始時刻."""
        return float(self.meta.get("start", 0.0))

    def vocal_analysis_signal(self) -> np.ndarray:
        """解析用 (22.05kHz モノラル) のボーカル信号."""
        if self._vocal_cache is None:
            y, sr = load_wav(self.vocal_path)
            self._vocal_cache = resample(to_mono(y), sr, config.ANALYSIS_SR)
        return self._vocal_cache

    # ------------------------------------------------------------------ 保存 / 読み込み
    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.folder / "curves.npz",
            f0_hz=self.f0_hz, midi_raw=self.midi_raw, midi=self.midi, vprob=self.vprob,
            **{f"curve_{k}": v for k, v in self.curves.items()},
            **{f"lane_{k}": v for k, v in self.lanes.items()},
        )
        doc = {
            "meta": self.meta,
            "summary": self.summary,
            "harmony": self.harmony,
            "notes": [{"start": n.start, "end": n.end, "midi": n.midi, "center": n.center,
                       "features": n.features} for n in self.notes],
        }
        (self.folder / "analysis.json").write_text(
            json.dumps(_jsonable(doc), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: str | Path) -> "AnalysisResult":
        folder = Path(folder)
        doc = json.loads((folder / "analysis.json").read_text(encoding="utf-8"))
        z = np.load(folder / "curves.npz")
        curves = {k[6:]: z[k] for k in z.files if k.startswith("curve_")}
        lanes = {k[5:]: z[k] for k in z.files if k.startswith("lane_")}
        keep_none = ("events", "harmonic_profile_db", "chord", "chord_role", "chord_tone", "chord_degree",
                     "chord_interval")
        notes = []
        for d in doc["notes"]:
            feats = {k: (np.nan if v is None and k not in keep_none else v) for k, v in d["features"].items()}
            notes.append(Note(start=d["start"], end=d["end"], midi=d["midi"], center=d["center"], features=feats))
        return cls(meta=doc["meta"], f0_hz=z["f0_hz"], midi_raw=z["midi_raw"], midi=z["midi"], vprob=z["vprob"],
                   curves=curves, lanes=lanes, notes=notes, summary=doc["summary"], folder=folder,
                   harmony=doc.get("harmony") or {})

    def export_notes_csv(self, path: str | Path) -> Path:
        cols = ["index", "start_s", "end_s", "dur_ms", "name", "midi", "intonation_cents",
                "approach_label", "attack_ms", "attack_depth_cents", "overshoot_cents",
                "release_label", "release_ms", "release_depth_cents", "transition_ms",
                "vib", "vib_rate_hz", "vib_extent_cents", "vib_onset_ms", "vib_len_ratio", "vib_regularity",
                "stability_cents", "dyn_mean_db", "attack_rise_db", "crescendo_db_s", "release_decay_db",
                "hnr_db", "centroid_hz", "h1h2_db", "tilt_db_oct", "chord", "chord_degree", "chord_role",
                "chord_tone", "events"]
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for n in self.notes:
                f = n.features
                row = []
                for c in cols:
                    v = f.get(c)
                    if c == "start_s" or c == "end_s":
                        v = v + self.offset_s
                    if c == "events":
                        v = " / ".join(f"{e['label']}@{e['time_s'] + self.offset_s:.2f}s" for e in v or [])
                    elif isinstance(v, float):
                        v = "" if not math.isfinite(v) else round(v, 3)
                    row.append("" if v is None else v)
                w.writerow(row)
        return path

    def export_chords_csv(self, path: str | Path) -> Path:
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            key = self.harmony.get("key", {})
            w.writerow(["# key", key.get("name", ""), "tempo_bpm", round(self.harmony.get("tempo", 0.0), 1)])
            w.writerow(["start_s", "end_s", "chord", "degree", "score"])
            for c in self.chords:
                w.writerow([round(c["start"] + self.offset_s, 3), round(c["end"] + self.offset_s, 3), c["label"],
                            c["degree"], round(c.get("score") or 0.0, 3)])
        return path

    def export_pitch_csv(self, path: str | Path) -> Path:
        path = Path(path)
        t = self.times + self.offset_s
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time_s", "f0_hz", "midi_tuned", "rms_db", "hnr_db", "centroid_hz",
                        "DYN", "BRE", "BRI"])
            for i in range(len(t)):
                w.writerow([f"{t[i]:.3f}"] + [
                    "" if not np.isfinite(v) else f"{v:.3f}" for v in (
                        self.f0_hz[i], self.midi_raw[i], self.curves["rms_db"][i], self.curves["hnr_db"][i],
                        self.curves["centroid_hz"][i], self.lanes["DYN"][i], self.lanes["BRE"][i],
                        self.lanes["BRI"][i])])
        return path


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, Path):
        return str(o)
    return o


def result_folder(video_id: str, start: float, end: float, method: str) -> Path:
    return config.RESULT_DIR / f"{video_id}_{int(round(start * 100))}-{int(round(end * 100))}_{method}"


def _load_or_separate(folder: Path, mix: np.ndarray, sr: int, key: dict, method: str, device: str,
                      cb: ProgressCB) -> dict[str, np.ndarray]:
    sep_info = folder / "separation.json"
    stems_dir = folder / "stems"
    if sep_info.exists():
        try:
            info = json.loads(sep_info.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            info = {}
        names = info.get("stems") or []
        if info.get("key") == key and names and all((stems_dir / f"{s}.wav").exists() for s in names):
            cb("分離済みのパートを再利用します", 1.0)
            return {s: load_wav(stems_dir / f"{s}.wav")[0] for s in names}
    stems = separate(mix, sr, method=method, device=device, progress=cb)
    stems_dir.mkdir(parents=True, exist_ok=True)
    for name, data in stems.items():
        save_wav(stems_dir / f"{name}.wav", data, sr)
    sep_info.write_text(json.dumps({"key": key, "stems": sort_stems(stems)}, ensure_ascii=False), encoding="utf-8")
    return stems


def run_analysis(wav_path: str | Path, source: dict, start: float, end: float, method: str = "demucs_htdemucs",
                 device: str = "auto", fmin: float = 65.0, fmax: float = 1100.0,
                 progress: ProgressCB | None = None) -> AnalysisResult:
    """1 区間を解析して AnalysisResult を返す (結果は cache/results/ に保存)."""
    config.ensure_dirs()
    t0 = time.time()

    def stage(lo: float, hi: float) -> ProgressCB:
        def cb(msg: str, frac: float) -> None:
            if progress:
                progress(msg, lo + (hi - lo) * frac if frac >= 0 else -1)
        return cb

    folder = result_folder(source.get("video_id", "audio"), start, end, method)
    folder.mkdir(parents=True, exist_ok=True)
    stage(0, 0.02)("区間を切り出し中...", 0.0)
    mix, sr = load_wav(wav_path, start, end)
    if mix.shape[1] < sr * 0.5:
        raise ValueError("区間が短すぎます (0.5 秒以上を指定してください)")

    # ---- 音源分離 (同じ区間・方式ならキャッシュを再利用)
    key = {"start": start, "end": end, "method": method, "src": str(wav_path), "v": SEPARATION_VERSION}
    stems = _load_or_separate(folder, mix, sr, key, method, device, stage(0.02, 0.4))
    vocals = stems["vocals"]
    save_wav(folder / "mix.wav", mix, sr)

    # ---- f0 + FFT 倍音解析
    v = resample(to_mono(vocals), sr, config.ANALYSIS_SR)
    asr, hop = config.ANALYSIS_SR, config.HOP
    f0, vprob = track_f0(v, asr, fmin, fmax, hop, progress=stage(0.4, 0.62))
    curves = harmonic_analysis(v, asr, f0, hop, progress=stage(0.62, 0.74))
    f0_ref = curves.pop("f0")
    # 分離の残響・伴奏の漏れなど、極端に小さい音のピッチは捨てる (曲内の大きい声から -35dB)
    rms = curves["rms_db"]
    vmask = np.isfinite(f0_ref)
    if vmask.any():
        ref_db = np.percentile(rms[vmask], 95)
        f0_ref[vmask & (rms < ref_db - 35.0)] = np.nan
    f0_ref = clean_f0_edges(f0_ref, rms)

    # ---- コード・キー・テンポ (伴奏パートから)
    harm_names = [s for s in HARMONIC_STEMS if s in stems]
    if harm_names:
        harm_sig = sum(to_mono(stems[s]) for s in harm_names)
    else:                                   # 分離なし: 元音源全体から
        harm_sig = to_mono(mix)
    harm22 = resample(np.asarray(harm_sig, dtype=np.float32), sr, asr)
    bass22 = resample(to_mono(stems["bass"]), sr, asr) if "bass" in stems else None
    mix22 = resample(to_mono(mix), sr, asr)
    try:
        harmony = analyze_harmony(harm22, asr, bass=bass22, rhythm=mix22, progress=stage(0.74, 0.92))
    except Exception as exc:                # コード解析の失敗で全体を止めない
        harmony = {"error": f"{type(exc).__name__}: {exc}", "chords": [], "beats": [], "tempo": 0.0}

    stage(0.92, 1.0)("ノート分割と特徴量計算中...", 0.0)
    midi = hz_to_midi(f0_ref)
    tuning = estimate_tuning(midi)
    midi_adj = midi - tuning

    import librosa

    onset = librosa.onset.onset_strength(y=v, sr=asr, hop_length=hop, lag=2, max_size=3)
    n = len(f0_ref)
    onset = np.pad(onset, (0, max(0, n - len(onset))))[:n]
    rms = curves["rms_db"]
    notes, filled = segment_notes(midi_adj, rms, onset)
    note_features(notes, filled, curves)
    # 各ノートとその時のコードの関係 (コードトーンか、テンションか)
    chords = harmony.get("chords", [])
    key_info = harmony.get("key") or {}
    for nt in notes:
        f = nt.features
        c = chord_at(chords, 0.5 * (f["start_s"] + f["end_s"]))
        rel = relate_note(nt.midi, c)
        f["chord"] = rel["chord"]
        f["chord_interval"] = rel["interval"]
        f["chord_role"] = rel["role"]
        f["chord_tone"] = rel["chord_tone"]
        f["chord_degree"] = c["degree"] if c else "-"
        if key_info.get("relative_major") is not None:
            f["scale_degree"] = (nt.midi - key_info["relative_major"]) % 12
    voiced = np.isfinite(midi_adj)
    lanes = control_lanes(curves, voiced)
    duration = (end - start)
    summary = summarize(notes, tuning, duration, harmony)

    meta = {
        "title": source.get("title", ""), "channel": source.get("channel", ""), "url": source.get("url", ""),
        "video_id": source.get("video_id", ""), "source_wav": str(wav_path),
        "start": float(start), "end": float(end), "method": method, "device": device,
        "fmin": fmin, "fmax": fmax, "analysis_sr": asr, "hop_sec": config.HOP_SEC, "io_sr": sr,
        "tuning_offset": tuning, "elapsed_s": time.time() - t0, "stems": sort_stems(stems),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    res = AnalysisResult(meta=meta, f0_hz=f0_ref, midi_raw=midi_adj, midi=filled, vprob=vprob,
                         curves=curves, lanes=lanes, notes=notes, summary=summary, folder=folder,
                         harmony=harmony, _vocal_cache=v)
    res.save()
    stage(0.92, 1.0)(f"解析完了 ({time.time() - t0:.1f} 秒)", 1.0)
    return res
