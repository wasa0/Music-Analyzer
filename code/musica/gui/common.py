"""GUI 共通: テーマ, バックグラウンドワーカー, 時刻入力ボックス, 配色."""
from __future__ import annotations

import os
import re
import sys
import traceback
from typing import Callable

from PySide6.QtCore import QCoreApplication, QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette, QValidator
from PySide6.QtWidgets import QApplication, QDoubleSpinBox

from .. import config

# ----------------------------------------------------------------------------------- 配色
APPROACH_COLORS = {
    "scoop": "#ff9f43",       # しゃくり
    "fall_in": "#a29bfe",     # 上から
    "portamento": "#48dbfb",  # ポルタメント
    "straight": "#1dd1a1",    # ストレート
}
RELEASE_COLORS = {
    "fall": "#ff6b6b",
    "rise": "#feca57",
    "preparation": "#f368e0",
    "portamento": "#48dbfb",
    "flat": "#8395a7",
}
EVENT_COLORS = {
    "spike_up": "#ff4757",
    "spike_down": "#5352ed",
    "kobushi": "#ffa502",
    "dip": "#3742fa",
    "flip": "#ff6b81",
}
LANE_COLORS = {"DYN": "#2ed573", "BRE": "#70a1ff", "BRI": "#ffa502", "PIT": "#eccc68"}
BG = "#15171d"
FG = "#c8ccd4"


def setup_environment(app_id: str = "MUSICA.EX") -> None:
    """OS ごとの Qt の初期設定 (QApplication を作る前に呼ぶ)."""
    if sys.platform.startswith("linux"):
        # Wayland セッションでは xcb プラグイン (libxcb-cursor) 依存を避けて wayland を優先
        if "QT_QPA_PLATFORM" not in os.environ and os.environ.get("WAYLAND_DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "wayland;xcb"
        # Qt 標準の PipeWire バックエンドは一部の出力 (リモートデスクトップ転送用の Dummy Output 等) を
        # 無視するため、PipeWire/PulseAudio のどちらでも動く PulseAudio 互換バックエンドを使う
        os.environ.setdefault("QT_AUDIO_BACKEND", "pulseaudio")
    if sys.platform == "win32":
        # タスクバーで Python ではなくこのアプリのアイコンを出す
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass
    os.environ.setdefault("QT_LOGGING_RULES", "qt.multimedia.*=false;qt.qpa.*=false")


def apply_theme(app: QApplication) -> None:
    import pyqtgraph as pg

    app.setStyle("Fusion")
    pal = QPalette()
    base = QColor("#1b1e25")
    pal.setColor(QPalette.Window, QColor("#1f222a"))
    pal.setColor(QPalette.WindowText, QColor(FG))
    pal.setColor(QPalette.Base, base)
    pal.setColor(QPalette.AlternateBase, QColor("#23262e"))
    pal.setColor(QPalette.ToolTipBase, QColor("#2b2f39"))
    pal.setColor(QPalette.ToolTipText, QColor("#eeeeee"))
    pal.setColor(QPalette.Text, QColor(FG))
    pal.setColor(QPalette.Button, QColor("#2a2e38"))
    pal.setColor(QPalette.ButtonText, QColor(FG))
    pal.setColor(QPalette.Highlight, QColor("#3d7eff"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.PlaceholderText, QColor("#6c7280"))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#5c6270"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#5c6270"))
    app.setPalette(pal)
    families = set(QFontDatabase.families())
    for fam in config.APP_FONT_FAMILIES:
        if fam in families:
            app.setFont(QFont(fam, 10))
            break
    app.setStyleSheet("""
        QGroupBox { border: 1px solid #343846; border-radius: 6px; margin-top: 10px; padding-top: 6px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #9aa4b8; }
        QPushButton { padding: 5px 10px; border-radius: 4px; border: 1px solid #3a3f4d; background: #2a2e38; }
        QPushButton:hover { background: #343947; }
        QPushButton:pressed { background: #3d7eff; }
        QPushButton#primary { background: #3d7eff; color: white; border: none; font-weight: bold; }
        QPushButton#primary:hover { background: #5a92ff; }
        QPushButton:disabled { color: #5c6270; background: #23262e; }
        QTabBar::tab { padding: 6px 14px; background: #23262e; border: 1px solid #343846; border-bottom: none;
                       border-top-left-radius: 4px; border-top-right-radius: 4px; color: #9aa4b8; }
        QTabBar::tab:selected { background: #2f3440; color: #ffffff; }
        QHeaderView::section { background: #262a33; color: #9aa4b8; border: none; padding: 4px; }
        QTableWidget { gridline-color: #2c303a; }
        QProgressBar { border: 1px solid #343846; border-radius: 4px; text-align: center; height: 16px; }
        QProgressBar::chunk { background: #3d7eff; border-radius: 3px; }
        QToolTip { border: 1px solid #4a5060; }
    """)
    pg.setConfigOptions(antialias=True, background=BG, foreground=FG)


def app_icon():
    from .icon import app_icon as _ai

    return _ai()


def fmt_time(sec: float, digits: int = 1) -> str:
    if sec is None or sec != sec:
        return "-"
    neg = sec < 0
    sec = abs(sec)
    m, s = divmod(sec, 60.0)
    txt = f"{int(m)}:{s:0{3 + digits}.{digits}f}" if digits else f"{int(m)}:{int(s):02d}"
    return "-" + txt if neg else txt


def parse_time(text: str) -> float | None:
    """'1:23.5' / '83.5' / '1:02:03' を秒に変換."""
    text = text.strip().replace("：", ":")
    if not text:
        return None
    if re.fullmatch(r"\d+(\.\d*)?", text):
        return float(text)
    parts = text.split(":")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    sec = 0.0
    for v in vals:
        sec = sec * 60 + v
    return sec


class TimeSpinBox(QDoubleSpinBox):
    """m:ss.ss 形式で表示し、'1:23.5' でも '83.5' でも入力できる秒数ボックス."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDecimals(2)
        self.setSingleStep(0.5)
        self.setRange(0.0, 36000.0)
        self.setKeyboardTracking(False)
        self.setAlignment(Qt.AlignRight)
        self.setMinimumWidth(96)

    def textFromValue(self, v: float) -> str:  # noqa: N802 (Qt API)
        return fmt_time(v, 2)

    def valueFromText(self, text: str) -> float:  # noqa: N802
        v = parse_time(text)
        return self.value() if v is None else v

    def validate(self, text: str, pos: int):  # noqa: D401
        if re.fullmatch(r"[\d:：.]*", text.strip()):
            return QValidator.Acceptable if parse_time(text) is not None else QValidator.Intermediate, text, pos
        return QValidator.Invalid, text, pos


# ------------------------------------------------------------------------------- ワーカー
class Worker(QObject):
    progress = Signal(str, float)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable, *args, use_progress: bool = True, **kwargs) -> None:
        super().__init__()
        self.fn, self.args, self.kwargs, self.use_progress = fn, args, kwargs, use_progress

    @Slot()
    def run(self) -> None:
        try:
            kw = dict(self.kwargs)
            if self.use_progress:
                kw["progress"] = lambda msg, frac: self.progress.emit(str(msg), float(frac))
            res = self.fn(*self.args, **kw)
        except Exception as exc:  # GUI に表示する
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            ok, payload = False, f"{type(exc).__name__}: {exc}"
        else:
            ok, payload = True, res
        # 終わったスレッドに置いたままにせず GUI スレッドへ戻す (後で安全に破棄できるように)
        app = QCoreApplication.instance()
        if app is not None:
            self.moveToThread(app.thread())
        if ok:
            self.finished.emit(payload)
        else:
            self.failed.emit(payload)


class _Relay(QObject):
    """ワーカーの結果を GUI スレッドで受け取る中継役.

    PySide6 では lambda やクロージャをシグナルに直接つなぐと「送信側のスレッド」で実行されるため、
    GUI スレッドに置いた QObject のスロット経由にして必ず GUI スレッドで呼ばれるようにする。
    """

    def __init__(self, on_done, on_error, on_progress) -> None:
        super().__init__()
        self._done, self._error, self._progress = on_done, on_error, on_progress
        self.finished = False

    @Slot(object)
    def done(self, res) -> None:
        if self._done:
            self._done(res)

    @Slot(str)
    def error(self, msg: str) -> None:
        if self._error:
            self._error(msg)

    @Slot(str, float)
    def progress(self, msg: str, frac: float) -> None:
        if self._progress:
            self._progress(msg, frac)

    @Slot()
    def mark_finished(self) -> None:
        self.finished = True


# 実行中・実行済みのジョブ (QThread, Worker, Relay)。参照を持っておかないと Python 側で破棄されてしまう。
# 破棄は次に start_worker が呼ばれたときにまとめて行う (コールバック実行中に消さないため)。
_active: set = set()


def _purge_finished() -> None:
    for job in list(_active):
        th, _w, relay = job
        if relay.finished and th.isFinished():
            _active.discard(job)


def start_worker(fn: Callable, *args, on_done: Callable | None = None, on_error: Callable | None = None,
                 on_progress: Callable | None = None, use_progress: bool = True, **kwargs) -> Worker:
    """fn を別スレッドで実行し、結果を GUI スレッドのコールバックで受け取る."""
    _purge_finished()
    app = QCoreApplication.instance()
    th = QThread()
    w = Worker(fn, *args, use_progress=use_progress, **kwargs)
    w.moveToThread(th)
    relay = _Relay(on_done, on_error, on_progress)
    if app is not None and th.thread() is not app.thread():
        # ワーカーの中から呼ばれた場合: 制御用オブジェクトも GUI スレッドへ移す
        # (呼び出し元のスレッドが先に終わっても quit/後片付けが届くように)
        th.moveToThread(app.thread())
        relay.moveToThread(app.thread())
    th.started.connect(w.run)
    # 接続順 = GUI スレッドでの実行順 (結果のコールバック → スレッド終了)
    w.progress.connect(relay.progress)
    w.finished.connect(relay.done)
    w.failed.connect(relay.error)
    w.finished.connect(th.quit)
    w.failed.connect(th.quit)
    th.finished.connect(relay.mark_finished)
    _active.add((th, w, relay))
    th.start()
    return w


def running_workers() -> int:
    return sum(1 for th, _w, _r in list(_active) if th.isRunning())


def wait_all_workers(timeout_ms: int = 3000) -> bool:
    """終了時用: 実行中のワーカーの終了を待つ. 全部終わったら True."""
    for th, _w, _r in list(_active):
        if th.isRunning():
            th.quit()
            th.wait(timeout_ms)
    return running_workers() == 0


def confirm_quit_if_busy(parent) -> bool:
    """解析やダウンロードの途中で閉じようとしたら確認する. 閉じてよければ True."""
    if running_workers() == 0:
        return True
    from PySide6.QtWidgets import QMessageBox

    ans = QMessageBox.question(parent, "処理中です",
                               "解析・ダウンロードなどの処理が実行中です。\n中断して終了しますか？",
                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    return ans == QMessageBox.Yes
