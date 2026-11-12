"""
ObservationStrip.py
===================
La secuencia observada O_1 ... O_t que recibe el HMM, dibujada como flechas.
Cada flecha es un símbolo: su dirección (16 posibles). Las de "lápiz arriba"
(saltos entre trazos) se dibujan huecas y punteadas. La última se resalta.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from src.features import FeatureExtractor as FE
from src.gui import styles


class ObservationStrip(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.simbolos: list[int] = []
        self.setFixedHeight(40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_simbolos(self, simbolos: list[int]) -> None:
        self.simbolos = list(simbolos)
        self.update()

    def paintEvent(self, _) -> None:
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        etiqueta = f"O₁…O{self._sub(len(self.simbolos))}" if self.simbolos else "Observaciones"
        g.setPen(QColor(styles.GRAFITO))
        g.drawText(QRectF(0, 0, 96, h), Qt.AlignVCenter | Qt.AlignLeft, etiqueta)

        celda = 18.0
        x0 = 100.0
        cabe = int((w - x0) // celda)
        visibles = self.simbolos[-cabe:] if cabe > 0 else []
        cy = h / 2
        for i, s in enumerate(visibles):
            cx = x0 + celda * (i + 0.5)
            ultimo = i == len(visibles) - 1
            arriba = s >= FE.PEN_UP
            ux, uy = FE.UNIT[s % FE.K]
            largo = celda * 0.36
            p0 = QPointF(cx - ux * largo, cy - uy * largo)
            p1 = QPointF(cx + ux * largo, cy + uy * largo)
            color = QColor(styles.LAPIZ_ROJO if ultimo else styles.TINTA)
            if not ultimo:
                color.setAlphaF(0.35 + 0.65 * (i + 1) / max(len(visibles), 1))
            g.setPen(QPen(color, 1.6, Qt.DotLine if arriba else Qt.SolidLine, Qt.RoundCap))
            g.drawLine(p0, p1)
            ang = np.arctan2(uy, ux)
            cabeza = QPainterPath(p1)
            for d in (2.6, -2.6):
                cabeza.moveTo(p1)
                cabeza.lineTo(p1.x() - 4.5 * np.cos(ang + d / 2.6 * 0.6),
                              p1.y() - 4.5 * np.sin(ang + d / 2.6 * 0.6))
            g.setPen(QPen(color, 1.6, Qt.SolidLine, Qt.RoundCap))
            g.drawPath(cabeza)
        g.end()

    @staticmethod
    def _sub(n: int) -> str:
        return "".join("₀₁₂₃₄₅₆₇₈₉"[int(d)] for d in str(n))
