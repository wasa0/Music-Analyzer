"""アプリのアイコン (コードで描画: 倍音の花びらリング + 中心の波形).

    python -m musica.gui.icon assets/musica.ico   # Windows のショートカット / exe 用に .ico を書き出す
"""
from __future__ import annotations

import math
import sys

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient

SIZES = (16, 24, 32, 48, 64, 128, 256)


def render_icon(size: int = 256) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    s = size / 256.0
    c = QPointF(size / 2, size / 2)
    # 背景: 角丸の濃紺 + 中心のグロー
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(14, 16, 24))
    p.drawRoundedRect(QRectF(4 * s, 4 * s, 248 * s, 248 * s), 52 * s, 52 * s)
    g = QRadialGradient(c, 120 * s)
    g.setColorAt(0.0, QColor(255, 80, 160, 120))
    g.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setBrush(g)
    p.drawEllipse(c, 120 * s, 120 * s)
    # 倍音リング: k 倍音 = k 枚の花びら, 色 = 倍音の音名
    hues = {1: 0.92, 2: 0.92, 3: 0.50, 4: 0.92, 5: 0.25}
    for k in range(5, 0, -1):
        r0 = (34 + 15 * k) * s
        path = QPainterPath()
        for i in range(181):
            th = 2 * math.pi * i / 180
            r = r0 + 6 * s * math.cos(k * th)
            pt = QPointF(c.x() + r * math.cos(th), c.y() + r * math.sin(th))
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        col = QColor.fromHsvF(hues[k], 0.7, 1.0)
        col.setAlphaF(0.95 - 0.1 * k)
        p.setPen(QPen(col, max(1.0, 5.5 * s - 0.6 * k * s)))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
    # 中心: 倍音を足した 1 周期の波形 (白)
    path = QPainterPath()
    for i in range(121):
        th = 2 * math.pi * i / 120
        x = math.cos(th) + 0.5 * math.cos(2 * th + 0.7) + 0.3 * math.cos(3 * th - 1.2)
        r = (26 + 9 * x) * s
        pt = QPointF(c.x() + r * math.cos(th - math.pi / 2), c.y() + r * math.sin(th - math.pi / 2))
        path.moveTo(pt) if i == 0 else path.lineTo(pt)
    p.setPen(QPen(QColor(255, 255, 255), max(1.0, 5 * s)))
    p.drawPath(path)
    p.end()
    return img


_icon: QIcon | None = None


def app_icon() -> QIcon:
    global _icon
    if _icon is None:
        _icon = QIcon()
        for sz in SIZES:
            _icon.addPixmap(QPixmap.fromImage(render_icon(sz)))
    return _icon


def _png_bytes(img: QImage) -> bytes:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice

    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba.data())


def save_icon_files(ico_path: str) -> None:
    """Windows 用 .ico (16〜256px の複数解像度, PNG 圧縮) と、同じ場所に .png (256px) を書き出す."""
    import struct
    from pathlib import Path

    out = Path(ico_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    render_icon(256).save(str(out.with_suffix(".png")))
    images = [(sz, _png_bytes(render_icon(sz))) for sz in SIZES]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for sz, data in images:
        dim = 0 if sz >= 256 else sz          # 256px は 0 で表す決まり
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    out.write_bytes(header + entries + blobs)


if __name__ == "__main__":
    from PySide6.QtGui import QGuiApplication

    app = QGuiApplication(sys.argv)
    save_icon_files(sys.argv[1] if len(sys.argv) > 1 else "assets/musica.ico")
    print("saved")
