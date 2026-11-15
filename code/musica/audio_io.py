"""ffmpeg を使った音声のデコード / 保存ユーティリティ."""
from __future__ import annotations

import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf


def run_quiet(cmd: list[str]) -> subprocess.CompletedProcess:
    """外部コマンドを実行 (Windows の GUI アプリでもコンソール窓を出さない)."""
    kw: dict = {"capture_output": True, "check": False, "stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return subprocess.run(cmd, **kw)


@lru_cache(maxsize=1)
def find_ffmpeg() -> str:
    """システムの ffmpeg、無ければ imageio-ffmpeg 同梱バイナリを返す."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - 環境依存
        raise RuntimeError("ffmpeg が見つかりません。`pip install imageio-ffmpeg` を実行してください。") from exc


def decode_audio(
    path: str | Path,
    sr: int = 44100,
    channels: int = 2,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    """任意の音声/動画ファイルを float32 配列 (channels, n) にデコードする."""
    cmd = [find_ffmpeg(), "-v", "error", "-nostdin"]
    if start is not None and start > 0:
        cmd += ["-ss", f"{start:.4f}"]
    cmd += ["-i", str(path)]
    if end is not None:
        dur = end - (start or 0.0)
        if dur <= 0:
            raise ValueError("終了時刻は開始時刻より後にしてください")
        cmd += ["-t", f"{dur:.4f}"]
    cmd += ["-vn", "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    proc = run_quiet(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg デコード失敗: {proc.stderr.decode(errors='ignore')[-500:]}")
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if data.size == 0:
        raise RuntimeError("音声データが空です (区間指定を確認してください)")
    return data.reshape(-1, channels).T.copy()


def convert_to_wav(src: str | Path, dst: str | Path, sr: int = 44100, channels: int = 2) -> Path:
    dst = Path(dst)
    cmd = [find_ffmpeg(), "-v", "error", "-nostdin", "-y", "-i", str(src), "-vn",
           "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
    proc = run_quiet(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"WAV 変換失敗: {proc.stderr.decode(errors='ignore')[-500:]}")
    return dst


def load_wav(path: str | Path, start: float | None = None, end: float | None = None) -> tuple[np.ndarray, int]:
    """WAV を (channels, n) float32 で読み込む (区間指定可)."""
    info = sf.info(str(path))
    sr = info.samplerate
    s = int(round((start or 0.0) * sr))
    e = int(round(end * sr)) if end is not None else info.frames
    s = max(0, min(s, info.frames))
    e = max(s, min(e, info.frames))
    data, sr = sf.read(str(path), start=s, stop=e, dtype="float32", always_2d=True)
    return data.T.copy(), sr


def audio_duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def save_wav(path: str | Path, y: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.T  # soundfile は (n, ch)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    sf.write(str(path), arr, sr, subtype="PCM_16")
    return path


def to_mono(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=0) if y.ndim == 2 else y


def resample(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return y.astype(np.float32, copy=False)
    import soxr

    if y.ndim == 2:
        return soxr.resample(y.T, sr_from, sr_to).T.astype(np.float32)
    return soxr.resample(y, sr_from, sr_to).astype(np.float32)
