"""音源分離 (ボーカル + 楽器パート).

- Demucs htdemucs / htdemucs_ft : ボーカル・ドラム・ベース・その他 の 4 パート
- Demucs htdemucs_6s            : 上記 + ギター・ピアノ の 6 パート
- 簡易分離 : センター定位 + REPET-SIM でボーカル、残りを HPSS と帯域分割でドラム/ベース/その他に分ける (軽量)
- 分離しない : アカペラ音源向け (ボーカル = 元音源)
"""
from __future__ import annotations

import os
import threading
from typing import Callable

import numpy as np

ProgressCB = Callable[[str, float], None]

METHODS: dict[str, str] = {
    "demucs_htdemucs": "Demucs htdemucs (4パート: ボーカル/ドラム/ベース/その他・推奨)",
    "demucs_htdemucs_6s": "Demucs htdemucs_6s (6パート: +ギター/ピアノ)",
    "demucs_htdemucs_ft": "Demucs htdemucs_ft (4パート・最高品質・約4倍遅い)",
    "simple": "簡易分離 (4パート・軽量・品質は低め)",
    "none": "分離しない (アカペラ音源向け)",
}

# 表示順・日本語名・色 (GUI 共通)
STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]
STEM_LABELS = {"vocals": "ボーカル", "drums": "ドラム", "bass": "ベース", "guitar": "ギター",
               "piano": "ピアノ", "other": "その他 (シンセ等)", "mix": "ミックス"}
STEM_COLORS = {"vocals": "#ff6b81", "drums": "#feca57", "bass": "#48dbfb", "guitar": "#ff9f43",
               "piano": "#a29bfe", "other": "#1dd1a1", "mix": "#dfe4ea"}
# コード認識に使う (音程のある伴奏) パート
HARMONIC_STEMS = ("bass", "guitar", "piano", "other")

_model_cache: dict[str, object] = {}
_lock = threading.Lock()


def demucs_available() -> bool:
    """Demucs と PyTorch が入っているか (起動を速くするため import はしない)."""
    from importlib.util import find_spec

    try:
        return find_spec("torch") is not None and find_spec("demucs") is not None
    except (ImportError, ValueError):
        return False


def cuda_usable() -> tuple[bool, str]:
    """NVIDIA GPU が使えるか. 使えない場合は理由も返す.

    インストールした PyTorch がその GPU の世代 (compute capability) 向けにビルドされていないと
    計算時に失敗するので、torch.cuda.get_arch_list() と照合する。
    """
    try:
        import torch
    except Exception:
        return False, "PyTorch がインストールされていません"
    try:
        if not torch.cuda.is_available():
            return False, "CUDA 対応の GPU / ドライバ / PyTorch が見つかりません"
        major, minor = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        cap = major * 10 + minor
        sms = [int(a.split("_")[1]) for a in archs if a.startswith("sm_") and a.split("_")[1].isdigit()]
        ptx = [int(a.split("_")[1]) for a in archs if a.startswith("compute_") and a.split("_")[1].isdigit()]
        # 同じメジャー世代でマイナー以下の sm があれば動く (例: sm_86 の GPU で sm_80 のバイナリ)。
        # PTX (compute_XX) があれば上位世代でも JIT コンパイルで動く
        ok = any(s // 10 == major and s <= cap for s in sms) or any(p <= cap for p in ptx)
        if not ok:
            name = torch.cuda.get_device_name(0)
            return False, (f"{name} (sm_{cap}) はこの PyTorch ({', '.join(archs)}) では使えません。"
                           "セットアップをやり直すと GPU に合った版が入ります")
        return True, torch.cuda.get_device_name(0)
    except Exception as exc:
        return False, f"GPU の確認に失敗しました ({type(exc).__name__}: {exc})"


def cpu_threads() -> int:
    """CPU で Demucs を動かすときのスレッド数.

    PyTorch の既定 (論理 CPU 全部) は、ハイパースレッディングや高効率コア (E コア) のある CPU では
    遅いコアや同期待ちが足を引っ張り、かえって数倍遅くなる (Core Ultra 9 285K の実測: 24 スレッド 37.5 秒 /
    12 スレッド 9.5 秒)。論理 CPU の半分 (最大 16) を使う。環境変数 MUSICA_CPU_THREADS で上書きできる。
    """
    env = os.environ.get("MUSICA_CPU_THREADS", "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    try:
        n = len(os.sched_getaffinity(0))          # Linux: コンテナ等の制限も反映
    except (AttributeError, OSError):
        n = os.cpu_count() or 4
    return n if n <= 6 else min(16, max(4, n // 2))


def resolve_device(device: str = "auto", progress: ProgressCB | None = None) -> str:
    """auto → NVIDIA GPU (cuda) / Apple Silicon (mps) / CPU の順で選ぶ. 指定したものが使えなければ CPU."""
    want = device
    if device in ("auto", "cuda"):
        ok, why = cuda_usable()
        if ok:
            return "cuda"
        if device == "cuda" and progress:
            progress(f"GPU を使えないため CPU で実行します: {why}", -1)
    if device in ("auto", "mps"):
        try:
            import torch

            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
        except Exception:
            pass
        if want == "mps" and progress:
            progress("Apple Silicon の GPU (MPS) を使えないため CPU で実行します", -1)
    return "cpu"


def available_methods() -> dict[str, str]:
    if demucs_available():
        return dict(METHODS)
    return {k: v for k, v in METHODS.items() if not k.startswith("demucs")}


def sort_stems(names) -> list[str]:
    return sorted(names, key=lambda s: STEM_ORDER.index(s) if s in STEM_ORDER else len(STEM_ORDER))


def separate(y: np.ndarray, sr: int, method: str = "demucs_htdemucs", device: str = "auto",
             progress: ProgressCB | None = None) -> dict[str, np.ndarray]:
    """ステレオ (2, n) 入力を {パート名: (2, n) float32} に分離する. 必ず 'vocals' を含む."""
    if y.ndim == 1:
        y = np.stack([y, y])
    y = y.astype(np.float32)
    if method == "none":
        return {"vocals": y}
    if method.startswith("demucs"):
        return _separate_demucs(y, sr, method.split("_", 1)[1], device, progress)
    if method == "simple":
        return _separate_simple(y, sr, progress)
    raise ValueError(f"未知の分離方法: {method}")


def separate_vocals(y: np.ndarray, sr: int, method: str = "demucs_htdemucs", device: str = "auto",
                    progress: ProgressCB | None = None) -> np.ndarray:
    """互換用: ボーカルだけを返す."""
    return separate(y, sr, method, device, progress)["vocals"]


# ----------------------------------------------------------------------------------------- Demucs
def _separate_demucs(y: np.ndarray, sr: int, name: str, device: str,
                     progress: ProgressCB | None) -> dict[str, np.ndarray]:
    import demucs.apply as dapply
    import torch
    from demucs.pretrained import get_model

    from .audio_io import resample

    device = resolve_device(device, progress)
    with _lock:
        if name not in _model_cache:
            if progress:
                progress(f"Demucs モデル '{name}' を読み込み中 (初回はダウンロード)...", -1)
            m = get_model(name)
            m.eval()
            _model_cache[name] = m
        model = _model_cache[name]

    msr = int(model.samplerate)
    x = resample(y, sr, msr) if sr != msr else y
    wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    # demucs 内部の tqdm を差し替えて進捗を GUI に流す
    class _TqdmShim:
        @staticmethod
        def tqdm(iterable, **kw):
            items = list(iterable)
            n = max(1, len(items))
            for i, it in enumerate(items):
                if progress:
                    progress(f"音源分離中 ({device}, {len(model.sources)} パート)...", 0.05 + 0.9 * i / n)
                yield it

    orig_tqdm = dapply.tqdm
    dapply.tqdm = _TqdmShim
    prev_threads = torch.get_num_threads()
    if device == "cpu":
        torch.set_num_threads(cpu_threads())
    try:
        with torch.no_grad():
            try:
                out = dapply.apply_model(model, wav[None], device=device, shifts=1, split=True,
                                         overlap=0.25, progress=True, num_workers=0)[0]
            except Exception as exc:     # CUDA の RuntimeError / AssertionError (CUDA 無し版) など
                if device == "cpu":
                    raise
                # GPU のメモリ不足・未対応の世代・未対応演算 (mps) などは CPU でやり直す
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                if progress:
                    progress(f"GPU で失敗したため CPU で再実行します ({exc.__class__.__name__}: {exc})"[:200], -1)
                model.to("cpu")
                device = "cpu"
                torch.set_num_threads(cpu_threads())
                out = dapply.apply_model(model, wav[None], device="cpu", shifts=1, split=True,
                                         overlap=0.25, progress=True, num_workers=0)[0]
    finally:
        dapply.tqdm = orig_tqdm
        torch.set_num_threads(prev_threads)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    out = (out * std + mean).cpu().numpy()
    n = y.shape[1]
    stems: dict[str, np.ndarray] = {}
    for i, src in enumerate(model.sources):
        s = out[i]
        if msr != sr:
            s = resample(s, msr, sr)
        if s.shape[1] < n:
            s = np.pad(s, ((0, 0), (0, n - s.shape[1])))
        stems[src] = s[:, :n].astype(np.float32)
    if progress:
        progress("音源分離完了", 1.0)
    return stems


# ----------------------------------------------------------------------------------------- 簡易分離
def _separate_simple(y: np.ndarray, sr: int, progress: ProgressCB | None) -> dict[str, np.ndarray]:
    import librosa
    from scipy.signal import butter, sosfiltfilt

    n_fft, hop = 2048, 512
    n = y.shape[1]
    if progress:
        progress("簡易分離: STFT...", 0.05)
    L = librosa.stft(y[0], n_fft=n_fft, hop_length=hop)
    R = librosa.stft(y[1], n_fft=n_fft, hop_length=hop)
    M = 0.5 * (L + R)
    # センター定位マスク: 左右が似ている成分ほど 1 (ボーカルは通常センター)
    center = 1.0 - np.abs(np.abs(L) - np.abs(R)) / (np.abs(L) + np.abs(R) + 1e-9)
    center = center ** 4
    if progress:
        progress("簡易分離: 繰り返し成分 (伴奏) を推定中...", 0.2)
    S = np.abs(M)
    width = int(librosa.time_to_frames(2.0, sr=sr, hop_length=hop))
    width = max(3, min(width, S.shape[1] // 3)) if S.shape[1] > 9 else 1
    if width > 1:
        S_rep = librosa.decompose.nn_filter(S, aggregate=np.median, metric="cosine", width=width)
        S_rep = np.minimum(S, S_rep)
        mask_v = librosa.util.softmask(S - S_rep, 2.0 * S_rep, power=2)
    else:
        mask_v = np.ones_like(S)
    # ボーカル帯域外 (80Hz 未満 / 10kHz 超) は伴奏側へ
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = ((freqs >= 80) & (freqs <= 10000)).astype(np.float32)[:, None]
    mask = mask_v * center * band
    if progress:
        progress("簡易分離: ボーカルを復元中...", 0.45)
    vocal_l = librosa.istft(L * mask, hop_length=hop, length=n)
    vocal_r = librosa.istft(R * mask, hop_length=hop, length=n)
    vocals = np.stack([vocal_l, vocal_r]).astype(np.float32)
    acc = (y - vocals).astype(np.float32)

    if progress:
        progress("簡易分離: ドラム (打楽器) と音程のある楽器を分離中...", 0.6)
    drums, harm = [], []
    for ch in range(2):
        A = librosa.stft(acc[ch], n_fft=n_fft, hop_length=hop)
        H, P = librosa.decompose.hpss(A, margin=2.0)
        drums.append(librosa.istft(P, hop_length=hop, length=n))
        harm.append(librosa.istft(H, hop_length=hop, length=n))
    drums = np.stack(drums).astype(np.float32)
    harm = np.stack(harm).astype(np.float32)
    # ベース = 音程のある成分の 200Hz 以下, その他 = 残り
    sos = butter(4, 200.0, btype="low", fs=sr, output="sos")
    bass = sosfiltfilt(sos, harm, axis=1).astype(np.float32)
    other = (harm - bass).astype(np.float32)
    if progress:
        progress("簡易分離完了", 1.0)
    return {"vocals": vocals, "drums": drums, "bass": bass, "other": other}
