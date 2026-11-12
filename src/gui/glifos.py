"""
glifos.py
=========
Dibuja cada letra con SUS TRAZOS (no con una fuente tipográfica), tomados de la
guía de dibujo del proyecto Glifo (assets/guia_letras.json, crédito: Daniel Santos).
Se usa para los íconos de referencia del panel y para la letra ganadora.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

from src.gui import styles

_RUTA = Path(__file__).resolve().parents[2] / "assets" / "guia_letras.json"
# Caja común a todas las letras (en el espacio 0..1000 de la guía): ascendente 180 → descendente 900
_CAJA = QRectF(320, 160, 380, 760)


@lru_cache(maxsize=1)
def trazos_guia() -> dict[str, list[list[list[float]]]]:
    try:
        return json.loads(_RUTA.read_text(encoding="utf-8"))["letras"]
    except (OSError, KeyError, ValueError):
        return {}


def dibujar_letra(g: QPainter, letra: str, rect: QRectF, color: str = styles.TINTA,
                  grosor: float | None = None, numeros: bool = False, linea_base: bool = False) -> None:
    """Dibuja `letra` dentro de `rect` con sus trazos de la guía (o con texto si no hay guía)."""
    trazos = trazos_guia().get(letra)
    escala = min(rect.width() / _CAJA.width(), rect.height() / _CAJA.height())
    ox = rect.center().x() - _CAJA.center().x() * escala
    oy = rect.center().y() - _CAJA.center().y() * escala
    a_px = lambda x, y: QPointF(ox + x * escala, oy + y * escala)
    grosor = grosor or max(1.6, rect.height() / 16)

    if linea_base:
        g.setPen(QPen(QColor(styles.RENGLON_FUERTE), 1))
        g.drawLine(a_px(200, 700), a_px(820, 700))

    if not trazos:
        f = QFont(g.font())
        f.setPixelSize(int(rect.height() * 0.6))
        g.setFont(f)
        g.setPen(QColor(color))
        g.drawText(rect, Qt.AlignCenter, letra)
        return

    g.setPen(QPen(QColor(color), grosor, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    g.setBrush(Qt.NoBrush)
    for t in trazos:
        camino = QPainterPath(a_px(*t[0]))
        for x, y in t[1:]:
            camino.lineTo(a_px(x, y))
        if len(t) == 1:
            camino.lineTo(a_px(t[0][0] + 1, t[0][1]))
        g.drawPath(camino)

    if numeros and len(trazos) > 1:
        f = QFont(g.font())
        f.setPixelSize(max(9, int(grosor * 2.4)))
        f.setWeight(QFont.DemiBold)
        g.setFont(f)
        r = grosor * 1.9
        for i, t in enumerate(trazos, start=1):
            c = a_px(*t[0])
            g.setPen(Qt.NoPen)
            g.setBrush(QColor(styles.LAPIZ_ROJO))
            g.drawEllipse(c, r, r)
            g.setPen(QColor(styles.HOJA))
            g.drawText(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), Qt.AlignCenter, str(i))
