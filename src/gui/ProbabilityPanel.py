"""
ProbabilityPanel.py
===================
Lista de las 27 letras ordenadas de mayor a menor P(c | O_1:t).
Cada fila muestra: ícono de referencia, letra, barra y porcentaje.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget

from src.gui import styles

ALTO_FILA = 44


class ProbabilityPanel(QWidget):
    def __init__(self, letras: list[str], parent=None) -> None:
        super().__init__(parent)
        self.datos = [(c, 1.0 / len(letras)) for c in letras]
        self.setMinimumWidth(300)
        self.setFixedHeight(ALTO_FILA * len(letras) + 8)

    def set_datos(self, datos: list[tuple[str, float]]) -> None:
        self.datos = datos
        self.update()

    def _icono(self, g: QPainter, r: QRectF, letra: str, destacada: bool) -> None:
        """Ícono estático de referencia: la letra dentro de una tarjeta redondeada."""
        g.setPen(Qt.NoPen)
        g.setBrush(QColor(styles.ACENTO if destacada else styles.SUPERFICIE_2))
        g.drawRoundedRect(r, 8, 8)
        g.setPen(QColor(styles.FONDO if destacada else styles.TEXTO))
        f = QFont(self.font())
        f.setPixelSize(int(r.height() * 0.62))
        g.setFont(f)
        g.drawText(r.adjusted(0, -2, 0, 0), Qt.AlignCenter, letra)

    def paintEvent(self, _) -> None:
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        ancho = self.width()
        for i, (letra, p) in enumerate(self.datos):
            y = 4 + i * ALTO_FILA
            top = i == 0 and p > 1.5 / len(self.datos)
            self._icono(g, QRectF(6, y + 4, 36, 36), letra, top)

            # barra de probabilidad
            x0, x1 = 54, ancho - 70
            fondo = QRectF(x0, y + 17, x1 - x0, 10)
            g.setPen(Qt.NoPen)
            g.setBrush(QColor(styles.SUPERFICIE_2))
            g.drawRoundedRect(fondo, 5, 5)
            g.setBrush(QColor(styles.ACENTO if top else "#5f7d96"))
            g.drawRoundedRect(QRectF(x0, y + 17, max(4.0, (x1 - x0) * p), 10), 5, 5)

            # porcentaje
            g.setPen(QColor(styles.TEXTO if top else styles.TEXTO_SUAVE))
            f = QFont(self.font())
            f.setPixelSize(14)
            f.setBold(top)
            g.setFont(f)
            g.drawText(QRectF(ancho - 66, y, 60, ALTO_FILA), Qt.AlignVCenter | Qt.AlignRight,
                       f"{p * 100:5.1f} %")
        g.end()
