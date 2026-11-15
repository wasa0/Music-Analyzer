"""音声プレーヤー (パートごとの音量・ミュート・ソロを実時間でミックス).

- 複数パート (ステム) をメモリに読み込み、QAudioSink (pull モード) に渡す直前にミックスする。
  そのため、再生中にミュート/ソロ/音量を変えても途切れずに即反映される。
- 出力処理は専用スレッドで動かすので、GUI の描画が重くても音が途切れにくい。
- 音声出力デバイスが無い環境では、音を出さずに時計だけ進める (画面の動作確認用)。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QIODevice, QObject, QThread, QTimer, Signal, Slot
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices


def _read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.T
    if data.shape[0] == 1:
        data = np.vstack([data, data])
    return np.ascontiguousarray(data[:2]), int(sr)


def _enum_name(e) -> str:
    return getattr(e, "name", str(e).split(".")[-1])


def _bytes_per_frame(fmt) -> int:
    """ステレオ 1 フレームのバイト数."""
    return {QAudioFormat.Float: 8, QAudioFormat.Int32: 8, QAudioFormat.Int16: 4, QAudioFormat.UInt8: 2}.get(fmt, 8)


def _encode(out: np.ndarray, fmt) -> bytes:
    """float32 (frames, 2) → 出力フォーマットのバイト列."""
    if fmt == QAudioFormat.Int16:
        return (out * 32767.0).astype("<i2").tobytes()
    if fmt == QAudioFormat.Int32:
        return (out.astype(np.float64) * 2147483647.0).astype("<i4").tobytes()
    if fmt == QAudioFormat.UInt8:
        return ((out + 1.0) * 127.5).astype(np.uint8).tobytes()
    return out.astype("<f4").tobytes()


class _MixState:
    """GUI スレッドとオーディオスレッドで共有するミックス状態."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stems: dict[str, np.ndarray] = {}    # 名前 → (2, n) float32 (出力サンプルレート)
        self.gains: dict[str, float] = {}
        self.volume = 0.8
        self.sr = 44100
        self.n = 0
        self.pos = 0                               # 次に出力するサンプル位置
        self.end: int | None = None                # 範囲再生の終了サンプル
        self.base = 0.0                            # 再生開始時の位置 (秒)
        self.clock = (0.0, 0.0)                    # (再生済み位置[秒], perf_counter)
        self.fmt: QAudioFormat.SampleFormat = QAudioFormat.Float

    def render(self, frames: int) -> np.ndarray | None:
        with self.lock:
            s = self.pos
            e = min(self.n, s + frames)
            if self.end is not None:
                e = min(e, self.end)
            if e <= s:
                return None
            out = np.zeros((e - s, 2), dtype=np.float32)
            for name, data in self.stems.items():
                g = self.gains.get(name, 1.0)
                if g > 1e-4:
                    out += g * data[:, s:e].T
            out *= self.volume
            self.pos = e
        np.clip(out, -1.0, 1.0, out=out)
        return out

    def mono_window(self, center: int, size: int) -> np.ndarray:
        """可視化用: 現在のミックス (ゲイン適用後) のモノラル波形."""
        a, b = center - size // 2, center - size // 2 + size
        out = np.zeros(size, dtype=np.float64)
        s, e = max(0, a), min(self.n, b)
        if e <= s:
            return out
        with self.lock:
            items = [(self.gains.get(k, 1.0), v) for k, v in self.stems.items()]
        for g, data in items:
            if g > 1e-4:
                out[s - a: e - a] += g * 0.5 * (data[0, s:e] + data[1, s:e])
        return out


class _MixDevice(QIODevice):
    def __init__(self, state: _MixState, worker: "_AudioWorker") -> None:
        super().__init__()
        self.state = state
        self.worker = worker

    def isSequential(self) -> bool:  # noqa: N802
        return True

    def bytesAvailable(self) -> int:  # noqa: N802
        return 1 << 20

    def readData(self, maxlen: int):  # noqa: N802
        st = self.state
        bpf = _bytes_per_frame(st.fmt)
        frames = maxlen // bpf
        self.worker.update_clock()
        if frames <= 0:
            return bytes()
        out = st.render(frames)
        if out is None:
            self.worker.request_finish()
            return bytes()
        return _encode(out, st.fmt)

    def writeData(self, data) -> int:  # noqa: N802
        return 0


class _AudioWorker(QObject):
    """オーディオスレッド側: QAudioSink を所有する."""

    finished = Signal()
    failed = Signal(str)

    def __init__(self, state: _MixState) -> None:
        super().__init__()
        self.state = state
        self.sink: QAudioSink | None = None
        self.dev: _MixDevice | None = None
        self._finishing = False

    @Slot(object, object)
    def start(self, device, fmt) -> None:
        self.stop()
        self._finishing = False
        self.sink = QAudioSink(device, fmt)
        # 120ms: GUI が重い瞬間 (Python の GIL 待ち) があっても途切れにくい長さ。
        # 再生位置は processedUSecs() から求めるので、バッファ長は表示のずれにはならない
        self.sink.setBufferSize(int(self.state.sr * 0.12) * _bytes_per_frame(fmt.sampleFormat()))
        self.dev = _MixDevice(self.state, self)
        self.dev.open(QIODevice.ReadOnly)
        # 注意: PySide6 6.12 では QAudioSink.stateChanged の引数 (QAudio::State) を Python に変換できず
        # 例外になるため接続しない。再生終了は readData でデータが尽きたことから判定する。
        self.state.clock = (self.state.base, time.perf_counter())
        self.sink.start(self.dev)
        # Qt 6.7 以降は QAudio が QtAudio に改名されており enum 同士の比較が一致しないので名前で判定
        if _enum_name(self.sink.error()) != "NoError" or _enum_name(self.sink.state()) == "StoppedState":
            err = _enum_name(self.sink.error())
            self.stop()
            self.failed.emit(err)

    @Slot()
    def stop(self) -> None:
        if self.sink is not None:
            self.sink.stop()
            self.sink.deleteLater()
            self.sink = None
        if self.dev is not None:
            self.dev.close()
            self.dev.deleteLater()
            self.dev = None

    def update_clock(self) -> None:
        if self.sink is not None:
            self.state.clock = (self.state.base + self.sink.processedUSecs() / 1e6, time.perf_counter())

    def request_finish(self) -> None:
        if not self._finishing:
            self._finishing = True
            QTimer.singleShot(150, self._finish)     # バッファに残った分を鳴らしきってから止める

    def _finish(self) -> None:
        self.stop()
        self.finished.emit()

    @Slot()
    def shutdown(self) -> None:
        """出力を止めてからオーディオスレッドを終える (この順で処理されることを保証する)."""
        self.stop()
        QThread.currentThread().quit()


class AudioPlayer(QObject):
    """GUI から使うプレーヤー. load / load_stems → play / pause / seek / set_gains."""

    positionChanged = Signal(float)   # 再生位置 (秒)
    playingChanged = Signal(bool)
    sourceChanged = Signal(str)
    _sig_start = Signal(object, object)
    _sig_stop = Signal()
    _sig_shutdown = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.state = _MixState()
        self._source: str | None = None
        self._playing = False
        self._virtual_t0: float | None = None
        self._device = self._pick_device()
        self._fmt: QAudioFormat | None = None
        # デバイス一覧は起動直後に非同期で埋まることがあるので、変化したら選び直す
        self._media = QMediaDevices(self)
        self._media.audioOutputsChanged.connect(self._outputs_changed)
        self._thread = QThread()
        self._thread.setObjectName("musica-audio")
        self._worker = _AudioWorker(self.state)
        self._worker.moveToThread(self._thread)
        self._sig_start.connect(self._worker.start)
        self._sig_stop.connect(self._worker.stop)
        self._sig_shutdown.connect(self._worker.shutdown)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()
        self._timer = QTimer(self)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ デバイス
    @staticmethod
    def _pick_device():
        dev = QMediaDevices.defaultAudioOutput()
        if dev.isNull():
            outs = QMediaDevices.audioOutputs()
            dev = outs[0] if outs else None
        return dev

    @property
    def has_device(self) -> bool:
        return self._device is not None and not self._device.isNull()

    def _choose_format(self, file_sr: int) -> tuple[QAudioFormat | None, int]:
        if not self.has_device:
            return None, file_sr
        for rate in (file_sr, self._device.preferredFormat().sampleRate()):
            for sf in (QAudioFormat.Float, QAudioFormat.Int16):
                fmt = QAudioFormat()
                fmt.setSampleRate(int(rate))
                fmt.setChannelCount(2)
                fmt.setSampleFormat(sf)
                if self._device.isFormatSupported(fmt):
                    return fmt, int(rate)
        fmt = self._device.preferredFormat()
        fmt.setChannelCount(2)
        return fmt, fmt.sampleRate()

    # ------------------------------------------------------------------ 読み込み
    @property
    def path(self) -> str | None:
        """現在読み込んでいるソースの識別子 (単一ファイルならそのパス)."""
        return self._source

    def load(self, path: str | Path) -> None:
        self.load_stems({"mix": path}, source_id=str(path))

    def load_stems(self, stems: dict[str, str | Path], source_id: str, gains: dict[str, float] | None = None) -> None:
        """複数パートのファイルを読み込む (長さが違う場合は短い方をゼロで埋める)."""
        if source_id == self._source and set(stems) == set(self.state.stems):
            if gains:
                self.set_gains(gains)
            return
        loaded = {}
        sr = None
        for name, p in stems.items():
            data, s = _read_audio(p)
            sr = sr or s
            if s != sr:
                from ..audio_io import resample

                data = resample(data, s, sr)
            loaded[name] = data
        self.load_arrays(loaded, sr or 44100, source_id, gains)

    def load_arrays(self, stems: dict[str, np.ndarray], sr: int, source_id: str,
                    gains: dict[str, float] | None = None) -> None:
        """メモリ上の波形 {名前: (2, n) または (n,)} を読み込む."""
        self.stop()
        loaded = {}
        for name, data in stems.items():
            d = np.asarray(data, dtype=np.float32)
            if d.ndim == 1:
                d = np.vstack([d, d])
            elif d.shape[0] == 1:
                d = np.vstack([d[0], d[0]])
            loaded[name] = np.ascontiguousarray(d[:2])
        n = max((v.shape[1] for v in loaded.values()), default=0)
        loaded = {k: (np.pad(v, ((0, 0), (0, n - v.shape[1]))) if v.shape[1] < n else v) for k, v in loaded.items()}
        with self.state.lock:
            self.state.stems = loaded
            self.state.gains = {k: (gains or {}).get(k, 1.0) for k in loaded}
            self.state.sr = sr
            self.state.n = n
            self.state.pos = 0
            self.state.end = None
            self.state.base = 0.0
            self.state.clock = (0.0, time.perf_counter())
        self._fmt = None                       # 出力フォーマットは再生開始時に決める
        self._source = source_id
        self.sourceChanged.emit(source_id)
        self.positionChanged.emit(0.0)

    def stem_names(self) -> list[str]:
        return list(self.state.stems)

    # ------------------------------------------------------------------ ミキサー
    def set_gains(self, gains: dict[str, float]) -> None:
        with self.state.lock:
            for k, v in gains.items():
                if k in self.state.stems:
                    self.state.gains[k] = float(v)

    def gains(self) -> dict[str, float]:
        return dict(self.state.gains)

    def set_volume(self, v: float) -> None:
        self.state.volume = max(0.0, min(1.0, float(v)))

    def volume(self) -> float:
        return self.state.volume

    def mono_window(self, t: float, size: int) -> np.ndarray:
        return self.state.mono_window(int(t * self.state.sr), size)

    @property
    def sample_rate(self) -> int:
        return self.state.sr

    # ------------------------------------------------------------------ 再生制御
    def duration(self) -> float:
        return self.state.n / float(self.state.sr) if self.state.n else 0.0

    def is_playing(self) -> bool:
        return self._playing

    def play(self, start: float | None = None, end: float | None = None) -> None:
        if not self.state.n:
            return
        pos = self.position() if start is None else max(0.0, float(start))
        if pos >= self.duration() - 0.01:
            pos = 0.0
        self._sig_stop.emit()
        ready = self._prepare_output()
        with self.state.lock:
            self.state.pos = int(pos * self.state.sr)
            self.state.end = int(end * self.state.sr) if end is not None else None
            self.state.base = pos
            self.state.clock = (pos, time.perf_counter())
        if ready:
            self._virtual_t0 = None
            self._sig_start.emit(self._device, self._fmt)
        else:
            self._virtual_t0 = time.perf_counter()
        self._set_playing(True)
        self._timer.start()

    def play_file(self, path: str | Path, start: float | None = None, end: float | None = None) -> None:
        self.load(path)
        self.play(start, end)

    def pause(self) -> None:
        if not self._playing:
            return
        pos = self.position()
        self._sig_stop.emit()
        with self.state.lock:
            self.state.pos = int(pos * self.state.sr)
            self.state.base = pos
            self.state.clock = (pos, time.perf_counter())
        self._virtual_t0 = None
        self._timer.stop()
        self._set_playing(False)
        self.positionChanged.emit(pos)

    def stop(self) -> None:
        self.pause()

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def seek(self, sec: float) -> None:
        sec = max(0.0, min(float(sec), self.duration()))
        if self._playing:
            end = self.state.end / self.state.sr if self.state.end is not None else None
            self.play(sec, end)
        else:
            with self.state.lock:
                self.state.pos = int(sec * self.state.sr)
                self.state.base = sec
                self.state.clock = (sec, time.perf_counter())
        self.positionChanged.emit(sec)

    def position(self) -> float:
        st = self.state
        if not self._playing:
            return st.base
        if self._virtual_t0 is not None:
            pos = st.base + (time.perf_counter() - self._virtual_t0)
        else:
            played, stamp = st.clock
            pos = played + min(0.12, time.perf_counter() - stamp)   # 直近の通知から補間
        if st.end is not None:
            pos = min(pos, st.end / st.sr)
        return min(pos, self.duration())

    # ------------------------------------------------------------------ 内部
    def _prepare_output(self) -> bool:
        """出力デバイスとフォーマットを決め、必要ならパートをデバイスのサンプルレートに変換する."""
        if not self.has_device:
            self._device = self._pick_device()
            if not self.has_device:
                return False
        if self._fmt is not None:
            return True
        fmt, rate = self._choose_format(self.state.sr)
        if fmt is None:
            return False
        if rate != self.state.sr:
            from ..audio_io import resample

            with self.state.lock:
                old = self.state.sr
                self.state.stems = {k: resample(v, old, rate) for k, v in self.state.stems.items()}
                self.state.n = max((v.shape[1] for v in self.state.stems.values()), default=0)
                self.state.sr = rate
        self.state.fmt = fmt.sampleFormat()
        self._fmt = fmt
        return True

    def _outputs_changed(self) -> None:
        if not self._playing:
            self._device = self._pick_device()
            self._fmt = None

    def _set_playing(self, on: bool) -> None:
        if on != self._playing:
            self._playing = on
            self.playingChanged.emit(on)

    def _tick(self) -> None:
        if not self._playing:
            return
        pos = self.position()
        self.positionChanged.emit(pos)
        if self._virtual_t0 is not None:
            end = self.state.end / self.state.sr if self.state.end is not None else self.duration()
            if pos >= end - 1e-3:
                self._on_finished()

    def _on_finished(self) -> None:
        if not self._playing:
            return
        end = self.state.end / self.state.sr if self.state.end is not None else self.duration()
        with self.state.lock:
            self.state.base = end
            self.state.pos = int(end * self.state.sr)
            self.state.end = None
        self._virtual_t0 = None
        self._timer.stop()
        self._set_playing(False)
        self.positionChanged.emit(end)

    def _on_failed(self, msg: str) -> None:
        # 出力デバイスが使えない: 無音の時計モードで続行
        print(f"[AudioPlayer] 音声出力を開けませんでした ({msg})。無音で再生位置だけ進めます。")
        self._device = None
        self._fmt = None
        if self._playing:
            self._virtual_t0 = time.perf_counter()
            with self.state.lock:
                self.state.base = self.state.clock[0]

    def shutdown(self) -> None:
        """ウィンドウを閉じるとき用. 出力を止めてオーディオスレッドを終わらせる."""
        self._timer.stop()
        self._playing = False
        if self._thread.isRunning():
            self._sig_shutdown.emit()            # 出力停止 → スレッド終了 の順でオーディオスレッドが処理
            if not self._thread.wait(2000):
                self._thread.quit()
                self._thread.wait(1000)

    def __del__(self):  # pragma: no cover - 終了時の後始末
        try:
            if self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(500)
        except Exception:
            pass
