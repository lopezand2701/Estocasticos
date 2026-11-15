"""
ObservationStrip.py
===================
La secuencia observada O_1 ... O_T que recibe el HMM, dibujada como flechas.
Cada flecha es un símbolo: su dirección (16 posibles). Las de "lápiz arriba"
(saltos entre trazos) se dibujan punteadas. La última se marca en rojo.

Interacción
    - Deslizar: arrastrar con el mouse, rueda del mouse o la barra de abajo.
    - Seleccionar: clic sobre una flecha (o flechas ← → del teclado).
      Se emite `seleccion(t)` para que el lienzo muestre ese tramo del trazo.
    - Esc o clic en la misma flecha: quitar la selección.
Mientras se escribe, la tira sigue sola al final (a menos que el usuario la haya movido).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from src.features import FeatureExtractor as FE
from src.gui import styles

CELDA = 22.0          # ancho de cada flecha
X0 = 108.0            # espacio para la etiqueta "O₁…O_T"
ALTO_BARRA = 6


def _sub(n: int) -> str:
    return "".join("₀₁₂₃₄₅₆₇₈₉"[int(d)] for d in str(n))


class ObservationStrip(QWidget):
    seleccion = Signal(object)        # índice t (0..T-1) o None

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.simbolos: list[int] = []
        self.sel: int | None = None
        self.offset = 0.0             # desplazamiento horizontal en píxeles
        self.seguir = True            # seguir al último símbolo mientras se escribe
        self._arrastre: tuple[float, float] | None = None
        self._movio = False
        self.setFixedHeight(58)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)

    # ---------------- datos ----------------
    def set_simbolos(self, simbolos: list[int]) -> None:
        nuevos = list(simbolos)
        if self.sel is not None and (self.sel >= len(nuevos) or nuevos[:self.sel + 1] != self.simbolos[:self.sel + 1]):
            self._seleccionar(None)                      # la flecha seleccionada ya no existe o cambió
        self.simbolos = nuevos
        if not nuevos:
            self.offset, self.seguir = 0.0, True
        if self.seguir:
            self.offset = self._max_offset()
        self._limitar()
        self.update()

    # ---------------- geometría ----------------
    def _ancho_visible(self) -> float:
        return max(self.width() - X0 - 6, 1.0)

    def _max_offset(self) -> float:
        return max(0.0, len(self.simbolos) * CELDA - self._ancho_visible())

    def _limitar(self) -> None:
        self.offset = float(np.clip(self.offset, 0.0, self._max_offset()))

    def _indice_en(self, x: float) -> int | None:
        if x < X0:
            return None
        t = int((x - X0 + self.offset) // CELDA)
        return t if 0 <= t < len(self.simbolos) else None

    def _seleccionar(self, t: int | None) -> None:
        self.sel = t
        self.seleccion.emit(t)
        if t is not None:                                 # que la flecha seleccionada quede a la vista
            x = t * CELDA
            if x < self.offset:
                self.offset = x
            elif x + CELDA > self.offset + self._ancho_visible():
                self.offset = x + CELDA - self._ancho_visible()
            self.seguir = False
        self._limitar()
        self.update()

    # ---------------- eventos ----------------
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._arrastre = (e.position().x(), self.offset)
            self._movio = False
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e) -> None:
        if self._arrastre is not None:
            x0, off0 = self._arrastre
            dx = e.position().x() - x0
            if abs(dx) > 3:
                self._movio = True
            if self._movio:
                self.offset = off0 - dx
                self.seguir = self.offset >= self._max_offset() - 1
                self._limitar()
                self.update()
        else:
            t = self._indice_en(e.position().x())
            self.setToolTip(self._texto(t) if t is not None else "")

    def mouseReleaseEvent(self, e) -> None:
        if self._arrastre is not None and not self._movio:          # fue un clic, no un arrastre
            t = self._indice_en(e.position().x())
            self._seleccionar(None if t == self.sel else t)
        self._arrastre = None
        self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, e) -> None:
        delta = e.angleDelta().y() or e.angleDelta().x()
        self.offset -= delta / 120 * CELDA * 3
        self.seguir = self.offset >= self._max_offset() - 1
        self._limitar()
        self.update()

    def keyPressEvent(self, e) -> None:
        if not self.simbolos:
            return
        if e.key() == Qt.Key_Escape:
            self._seleccionar(None)
        elif e.key() in (Qt.Key_Left, Qt.Key_Right):
            paso = -1 if e.key() == Qt.Key_Left else 1
            base = self.sel if self.sel is not None else len(self.simbolos)
            self._seleccionar(int(np.clip(base + paso, 0, len(self.simbolos) - 1)))
        else:
            super().keyPressEvent(e)

    def resizeEvent(self, e) -> None:
        if self.seguir:
            self.offset = self._max_offset()
        self._limitar()
        super().resizeEvent(e)

    # ---------------- dibujo ----------------
    def _texto(self, t: int | None) -> str:
        if t is None:
            return ""
        s = self.simbolos[t]
        tipo = "lápiz arriba" if s >= FE.PEN_UP else "lápiz abajo"
        return f"O{_sub(t + 1)} = {FE.FLECHAS[s % FE.K]}  (símbolo {s}, {tipo})"

    def paintEvent(self, _) -> None:
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cy = (h - ALTO_BARRA - 4) / 2

        f = QFont(self.font())
        f.setPixelSize(12)
        g.setFont(f)
        g.setPen(QColor(styles.GRAFITO))
        if self.sel is not None:
            etiqueta = f"O{_sub(self.sel + 1)} de {len(self.simbolos)}"
        elif self.simbolos:
            etiqueta = f"O₁…O{_sub(len(self.simbolos))}"
        else:
            etiqueta = "Observaciones"
        g.drawText(QRectF(0, 0, X0 - 8, h - ALTO_BARRA), Qt.AlignVCenter | Qt.AlignLeft, etiqueta)

        g.save()
        g.setClipRect(QRectF(X0, 0, w - X0, h))
        primero = int(self.offset // CELDA)
        ultimo_vis = min(len(self.simbolos), int((self.offset + self._ancho_visible()) // CELDA) + 2)
        n = len(self.simbolos)
        for t in range(primero, ultimo_vis):
            s = self.simbolos[t]
            cx = X0 + t * CELDA - self.offset + CELDA / 2
            elegido = t == self.sel
            if elegido:
                g.setPen(Qt.NoPen)
                g.setBrush(QColor(styles.RESALTADOR))
                g.drawRoundedRect(QRectF(cx - CELDA / 2 + 1, cy - 14, CELDA - 2, 28), 6, 6)
            arriba = s >= FE.PEN_UP
            ux, uy = FE.UNIT[s % FE.K]
            largo = CELDA * 0.32
            p0 = QPointF(cx - ux * largo, cy - uy * largo)
            p1 = QPointF(cx + ux * largo, cy + uy * largo)
            if elegido or t == n - 1:
                color = QColor(styles.LAPIZ_ROJO)
            else:
                color = QColor(styles.TINTA)
                color.setAlphaF(0.55)
            grosor = 2.2 if elegido else 1.6
            g.setPen(QPen(color, grosor, Qt.DotLine if arriba else Qt.SolidLine, Qt.RoundCap))
            g.drawLine(p0, p1)
            ang = np.arctan2(uy, ux)
            cabeza = QPainterPath()
            for d in (0.6, -0.6):
                cabeza.moveTo(p1)
                cabeza.lineTo(p1.x() - 5 * np.cos(ang + d), p1.y() - 5 * np.sin(ang + d))
            g.setPen(QPen(color, grosor, Qt.SolidLine, Qt.RoundCap))
            g.drawPath(cabeza)
        g.restore()

        # barra de desplazamiento (solo si no caben todas)
        total = n * CELDA
        vis = self._ancho_visible()
        if total > vis:
            pista = QRectF(X0, h - ALTO_BARRA - 1, vis, ALTO_BARRA)
            g.setPen(Qt.NoPen)
            g.setBrush(QColor(styles.NIEBLA))
            g.drawRoundedRect(pista, 3, 3)
            ancho = max(24.0, vis * vis / total)
            x = X0 + (vis - ancho) * (self.offset / max(self._max_offset(), 1e-9))
            g.setBrush(QColor(styles.RENGLON_FUERTE))
            g.drawRoundedRect(QRectF(x, h - ALTO_BARRA - 1, ancho, ALTO_BARRA), 3, 3)
        g.end()
