"""
ProbabilityPanel.py
===================
Lista de las 27 letras ordenadas de mayor a menor P(c | O_1:t).
Cada fila muestra: ícono de referencia (la letra dibujada con sus trazos de la
guía), la letra, una barra de probabilidad y el porcentaje.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget

from src.gui import glifos, styles

ALTO_FILA = 46


class ProbabilityPanel(QWidget):
    def __init__(self, letras: list[str], parent=None) -> None:
        super().__init__(parent)
        self.datos = [(c, 1.0 / len(letras)) for c in letras]
        self.activo = False
        self.setMinimumWidth(280)
        self.setFixedHeight(ALTO_FILA * len(letras) + 6)

    def set_datos(self, datos: list[tuple[str, float]], activo: bool = True) -> None:
        self.datos = datos
        self.activo = activo
        self.update()

    def paintEvent(self, _) -> None:
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        ancho = self.width()
        for i, (letra, p) in enumerate(self.datos):
            y = 3 + i * ALTO_FILA
            primera = self.activo and i == 0
            if primera:                                         # resaltador detrás de la ganadora
                g.setPen(Qt.NoPen)
                g.setBrush(QColor(styles.RESALTADOR))
                g.drawRoundedRect(QRectF(2, y + 2, ancho - 4, ALTO_FILA - 4), 10, 10)
            elif i % 2 == 1:
                g.setPen(Qt.NoPen)
                g.setBrush(QColor(styles.PAPEL))
                g.drawRoundedRect(QRectF(2, y + 2, ancho - 4, ALTO_FILA - 4), 10, 10)

            # ícono: la letra con sus trazos, sobre su línea base
            caja = QRectF(10, y + 5, 30, ALTO_FILA - 10)
            glifos.dibujar_letra(g, letra, caja, styles.TINTA, grosor=1.8, linea_base=True)

            # nombre de la letra
            f = QFont(self.font())
            f.setPixelSize(15)
            f.setWeight(QFont.DemiBold if primera else QFont.Normal)
            g.setFont(f)
            g.setPen(QColor(styles.TINTA))
            g.drawText(QRectF(50, y, 24, ALTO_FILA), Qt.AlignVCenter | Qt.AlignLeft, letra)

            # barra
            x0, x1 = 78, ancho - 72
            g.setPen(Qt.NoPen)
            g.setBrush(QColor(styles.HOJA if primera else styles.NIEBLA))
            g.drawRoundedRect(QRectF(x0, y + ALTO_FILA / 2 - 4, x1 - x0, 8), 4, 4)
            if self.activo:
                g.setBrush(QColor(styles.TINTA if primera else styles.GRAFITO))
                g.drawRoundedRect(QRectF(x0, y + ALTO_FILA / 2 - 4, max(3.0, (x1 - x0) * p), 8), 4, 4)

            # porcentaje
            f.setPixelSize(13)
            g.setFont(f)
            g.setPen(QColor(styles.TINTA if primera else styles.GRAFITO))
            texto = f"{p * 100:.1f} %" if self.activo else "–"
            g.drawText(QRectF(ancho - 70, y, 62, ALTO_FILA), Qt.AlignVCenter | Qt.AlignRight, texto)
        g.end()
