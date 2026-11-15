"""yt-dlp を使った YouTube 検索・音声取得.

取得した音源は cache/downloads/<video_id>.wav (44.1kHz stereo) にキャッシュされる。
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from . import config
from .audio_io import convert_to_wav, find_ffmpeg

ProgressCB = Callable[[str, float], None]  # (メッセージ, 0..1 または -1=不定)


@dataclass
class VideoInfo:
    video_id: str
    title: str
    channel: str = ""
    duration: float | None = None
    view_count: int | None = None
    url: str = ""

    @property
    def duration_text(self) -> str:
        if not self.duration:
            return "-"
        m, s = divmod(int(self.duration), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@lru_cache(maxsize=1)
def _deno_path() -> str | None:
    """YouTube の JavaScript チャレンジを解くための deno (JS ランタイム) の場所."""
    import sys

    name = "deno.exe" if sys.platform == "win32" else "deno"
    # 1) 実行ファイル版 (PyInstaller) に同梱したもの
    if getattr(sys, "frozen", False):
        for base in (getattr(sys, "_MEIPASS", ""), str(Path(sys.executable).parent)):
            cand = Path(base) / "bin" / name
            if base and cand.is_file():
                return str(cand)
    # 2) 仮想環境に pip で入れたもの (deno パッケージ)
    try:
        import deno

        p = deno.find_deno_bin()
        if p and Path(p).is_file():
            return p
    except Exception:
        pass
    # 3) PATH 上のもの
    return shutil.which("deno")


def _base_opts() -> dict:
    # noprogress: 進捗は progress_hooks で GUI に出す (コンソールの無い Windows の GUI アプリで
    # 標準出力へ書き込もうとして落ちるのも防ぐ)
    opts: dict = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True,
                  "ffmpeg_location": find_ffmpeg()}
    deno = _deno_path()
    if deno:
        opts["js_runtimes"] = {"deno": {"path": deno}}
    return opts


_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_CLIENT_FALLBACKS: list[list[str] | None] = [None, None, ["tv", "web_safari"], ["android_vr", "mweb"]]


def parse_video_id(url: str) -> str | None:
    url = url.strip()
    if _ID_RE.match(url):
        return url
    try:
        u = urlparse(url)
    except ValueError:
        return None
    host = (u.hostname or "").lower()
    if host.endswith("youtu.be"):
        vid = u.path.lstrip("/").split("/")[0]
        return vid if _ID_RE.match(vid) else None
    if "youtube" in host:
        q = parse_qs(u.query)
        if "v" in q and _ID_RE.match(q["v"][0]):
            return q["v"][0]
        m = re.match(r"^/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})", u.path)
        if m:
            return m.group(1)
    return None


def parse_start_time(url: str) -> float | None:
    """URL の t= / start= パラメータ (例: 90, 1m30s) を秒に変換."""
    try:
        q = parse_qs(urlparse(url.strip()).query)
    except ValueError:
        return None
    val = (q.get("t") or q.get("start") or [None])[0]
    if not val:
        return None
    if val.isdigit():
        return float(val)
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", val)
    if m and any(m.groups()):
        h, mi, s = (int(g) if g else 0 for g in m.groups())
        return float(h * 3600 + mi * 60 + s)
    return None


def is_url(text: str) -> bool:
    return parse_video_id(text) is not None or text.strip().startswith(("http://", "https://"))


def search(query: str, max_results: int = 15) -> list[VideoInfo]:
    import yt_dlp

    opts = _base_opts() | {"skip_download": True, "extract_flat": "in_playlist"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        res = ydl.extract_info(f"ytsearch{int(max_results)}:{query}", download=False)
    out: list[VideoInfo] = []
    for e in res.get("entries") or []:
        if not e or not e.get("id"):
            continue
        out.append(VideoInfo(
            video_id=e["id"],
            title=e.get("title") or e["id"],
            channel=e.get("channel") or e.get("uploader") or "",
            duration=e.get("duration"),
            view_count=e.get("view_count"),
            url=f"https://www.youtube.com/watch?v={e['id']}",
        ))
    return out


def cached_audio(video_id: str) -> tuple[Path, VideoInfo] | None:
    wav = config.DOWNLOAD_DIR / f"{video_id}.wav"
    meta = config.DOWNLOAD_DIR / f"{video_id}.json"
    if wav.exists() and meta.exists():
        try:
            return wav, VideoInfo(**json.loads(meta.read_text(encoding="utf-8")))
        except (TypeError, json.JSONDecodeError):
            return None
    return None


def download_audio(url: str, progress: ProgressCB | None = None) -> tuple[Path, VideoInfo]:
    """YouTube から音声を取得して WAV に変換 (キャッシュ有り)."""
    import yt_dlp

    config.ensure_dirs()
    vid = parse_video_id(url)
    if vid:
        hit = cached_audio(vid)
        if hit:
            if progress:
                progress("キャッシュ済みの音源を使用します", 1.0)
            return hit
        url = f"https://www.youtube.com/watch?v={vid}"

    def hook(d: dict) -> None:
        if not progress:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            frac = (d.get("downloaded_bytes", 0) / total) if total else -1
            progress("ダウンロード中...", min(frac, 1.0) * 0.9 if frac >= 0 else -1)
        elif d.get("status") == "finished":
            progress("ダウンロード完了。WAV に変換中...", 0.92)

    with tempfile.TemporaryDirectory(prefix="musica_dl_") as tmp:
        info = None
        last_err: Exception | None = None
        # YouTube 側の仕様変更 (SABR / PO Token) で一時的に 403 になることがあるので、
        # 取得クライアントを変えながら数回リトライする
        for attempt, clients in enumerate(_CLIENT_FALLBACKS):
            opts = _base_opts() | {
                "format": "bestaudio/best",
                "outtmpl": str(Path(tmp) / "%(id)s.%(ext)s"),
                "progress_hooks": [hook],
            }
            if clients:
                opts["extractor_args"] = {"youtube": {"player_client": clients}}
            if progress:
                progress("動画情報を取得中..." if attempt == 0 else f"再試行中 ({attempt + 1}/{len(_CLIENT_FALLBACKS)})...", -1)
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                break
            except yt_dlp.utils.DownloadError as exc:
                last_err = exc
                msg = str(exc)
                if not any(s in msg for s in ("403", "Forbidden", "fragment", "Requested format")):
                    raise
                for f in Path(tmp).glob("*"):
                    f.unlink(missing_ok=True)
        if info is None:
            raise RuntimeError(f"YouTube から取得できませんでした (yt-dlp を更新すると直る場合があります: "
                               f"pip install -U 'yt-dlp[default]'): {last_err}")
        if info.get("_type") == "playlist":
            raise RuntimeError("プレイリストではなく単一の動画 URL を指定してください")
        files = [Path(r["filepath"]) for r in info.get("requested_downloads") or [] if r.get("filepath")]
        if not files:
            files = sorted(Path(tmp).glob(f"{info['id']}.*"))
        if not files:
            raise RuntimeError("音声ファイルを取得できませんでした")
        meta = VideoInfo(
            video_id=info["id"],
            title=info.get("title") or info["id"],
            channel=info.get("channel") or info.get("uploader") or "",
            duration=info.get("duration"),
            view_count=info.get("view_count"),
            url=info.get("webpage_url") or url,
        )
        wav = config.DOWNLOAD_DIR / f"{meta.video_id}.wav"
        convert_to_wav(files[0], wav, sr=config.IO_SR, channels=2)
    (config.DOWNLOAD_DIR / f"{meta.video_id}.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=1), encoding="utf-8")
    if progress:
        progress("音源の準備完了", 1.0)
    return wav, meta


def import_local_file(path: str | Path, progress: ProgressCB | None = None) -> tuple[Path, VideoInfo]:
    """ローカルの音声/動画ファイルを WAV に変換して同じ形式で扱う."""
    import hashlib

    config.ensure_dirs()
    path = Path(path)
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:12]
    vid = f"local_{key}"
    hit = cached_audio(vid)
    if hit:
        return hit
    if progress:
        progress("ファイルを WAV に変換中...", -1)
    wav = config.DOWNLOAD_DIR / f"{vid}.wav"
    convert_to_wav(path, wav, sr=config.IO_SR, channels=2)
    from .audio_io import audio_duration

    meta = VideoInfo(video_id=vid, title=path.name, channel="ローカルファイル",
                     duration=audio_duration(wav), url=str(path))
    (config.DOWNLOAD_DIR / f"{vid}.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=1), encoding="utf-8")
    if progress:
        progress("読み込み完了", 1.0)
    return wav, meta
