"""
StateChain.py
=============
La cadena de Markov de la letra ganadora, en vivo.

Cada círculo es una fase (estado oculto) del modelo izquierda-derecha.
    - el relleno de cada círculo es α̂_t(i): qué tan probable es estar en esa fase ahora
    - la fase más probable (s = argmax α̂_t) se marca con lápiz rojo
    - los arcos sobre cada círculo son los auto-lazos a_ii (quedarse en la fase)
    - las flechas entre círculos son las transiciones a_i,i+1 (avanzar)
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from src.gui import styles


class StateChain(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.estado: dict | None = None
        self.setMinimumHeight(86)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_estado(self, estado: dict | None) -> None:
        self.estado = estado
        self.update()

    def paintEvent(self, _) -> None:
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if not self.estado:
            g.setPen(QColor(styles.GRAFITO))
            g.drawText(QRectF(0, 0, w, h), Qt.AlignCenter,
                       "Aquí aparecerán las fases ocultas de la letra")
            g.end()
            return

        N = self.estado["N"]
        fase = self.estado["fase"]
        alpha = self.estado["alpha"]
        margen = 14
        paso = (w - 2 * margen) / max(N, 1)
        r = min(paso * 0.32, 13.0)
        cy = h * 0.56

        for i in range(N):
            cx = margen + paso * (i + 0.5)
            # transición a la siguiente fase
            if i < N - 1:
                x0, x1 = cx + r + 2, cx + paso - r - 4
                g.setPen(QPen(QColor(styles.RENGLON_FUERTE), 1.4))
                g.drawLine(QPointF(x0, cy), QPointF(x1, cy))
                cabeza = QPainterPath(QPointF(x1 + 3, cy))
                cabeza.lineTo(x1 - 3, cy - 3.5)
                cabeza.lineTo(x1 - 3, cy + 3.5)
                cabeza.closeSubpath()
                g.setPen(Qt.NoPen)
                g.setBrush(QColor(styles.RENGLON_FUERTE))
                g.drawPath(cabeza)
            # auto-lazo (quedarse en la fase)
            actual = i == fase
            color_lazo = QColor(styles.LAPIZ_ROJO if actual else styles.RENGLON_FUERTE)
            g.setPen(QPen(color_lazo, 1.4))
            g.setBrush(Qt.NoBrush)
            g.drawArc(QRectF(cx - r * 0.7, cy - r * 2.25, r * 1.4, r * 1.7), -40 * 16, 260 * 16)
            # círculo: relleno según α̂_t(i)
            fondo = QColor(styles.TINTA)
            fondo.setAlphaF(float(np.clip(0.08 + 0.85 * alpha[i], 0.08, 0.93)))
            g.setPen(QPen(QColor(styles.LAPIZ_ROJO if actual else styles.TINTA), 2.2 if actual else 1.0))
            g.setBrush(fondo)
            g.drawEllipse(QPointF(cx, cy), r, r)
            f = QFont(self.font())
            f.setPixelSize(int(r * 0.95))
            f.setWeight(QFont.DemiBold if actual else QFont.Normal)
            g.setFont(f)
            g.setPen(QColor(styles.HOJA if alpha[i] > 0.45 else styles.TINTA))
            g.drawText(QRectF(cx - r, cy - r, 2 * r, 2 * r), Qt.AlignCenter, str(i + 1))
        g.end()
