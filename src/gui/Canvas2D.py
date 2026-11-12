"""
Canvas2D.py
===========
Lienzo cuadrado donde el usuario dibuja con el mouse (o un lápiz/táctil), con
aspecto de hoja de cuaderno de caligrafía. Trabaja internamente en el mismo
espacio lógico que el dataset Glifo: x, y entre 0 y 1000, con la y creciendo
hacia abajo, y las mismas líneas guía.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from src.features.FeatureExtractor import ESPACIO
from src.gui import styles

# Mismas líneas guía que Glifo (en coordenadas 0..1000)
GUIAS = {"ascendente": 180, "altura x": 420, "línea base": 700, "descendente": 900}


class Canvas2D(QWidget):
    trazos_cambiaron = Signal()        # se emite cada vez que el dibujo cambia

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.trazos: list[list[tuple[float, float]]] = []
        self.proyeccion: list = []                 # segmentos (p0, p1, lapiz_abajo)
        self._dibujando = False
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)

    # ---------------- conversión pantalla <-> espacio lógico ----------------
    def _area(self) -> QRectF:
        """Cuadrado más grande que cabe en el widget, centrado."""
        lado = min(self.width(), self.height()) - 4
        return QRectF((self.width() - lado) / 2, (self.height() - lado) / 2, lado, lado)

    def _a_logico(self, pos: QPointF) -> tuple[float, float]:
        a = self._area()
        x = (pos.x() - a.left()) / a.width() * ESPACIO
        y = (pos.y() - a.top()) / a.height() * ESPACIO
        return (min(max(x, 0.0), ESPACIO), min(max(y, 0.0), ESPACIO))

    def _a_pantalla(self, x: float, y: float) -> QPointF:
        a = self._area()
        return QPointF(a.left() + x / ESPACIO * a.width(), a.top() + y / ESPACIO * a.height())

    # ---------------- API pública ----------------
    def limpiar(self) -> None:
        self.trazos.clear()
        self.proyeccion = []
        self.update()
        self.trazos_cambiaron.emit()

    def deshacer(self) -> None:
        if self.trazos:
            self.trazos.pop()
            self.update()
            self.trazos_cambiaron.emit()

    def set_proyeccion(self, segmentos: list) -> None:
        self.proyeccion = segmentos
        self.update()

    def ultimo_punto(self):
        return self.trazos[-1][-1] if self.trazos and self.trazos[-1] else None

    @property
    def dibujando(self) -> bool:
        """True mientras el botón del mouse está presionado (lápiz abajo)."""
        return self._dibujando

    # ---------------- eventos del mouse ----------------
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._dibujando = True
            self.trazos.append([self._a_logico(e.position())])
            self.update()
            self.trazos_cambiaron.emit()

    def mouseMoveEvent(self, e) -> None:
        if not self._dibujando:
            return
        p = self._a_logico(e.position())
        u = self.trazos[-1][-1]
        if (p[0] - u[0]) ** 2 + (p[1] - u[1]) ** 2 >= 1.5 ** 2:     # igual que Glifo
            self.trazos[-1].append(p)
            self.update()
            self.trazos_cambiaron.emit()

    def mouseReleaseEvent(self, e) -> None:
        if self._dibujando:
            self._dibujando = False
            self.trazos_cambiaron.emit()           # al levantar el lápiz se reevalúa como letra completa

    # ---------------- dibujo ----------------
    def paintEvent(self, _) -> None:
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        a = self._area()

        # la hoja
        g.setPen(QPen(QColor(styles.RENGLON), 1))
        g.setBrush(QColor(styles.HOJA))
        g.drawRoundedRect(a, 14, 14)

        # cuadrícula muy tenue de fondo, como papel de caligrafía
        tenue = QColor(styles.RENGLON)
        tenue.setAlpha(70)
        g.setPen(QPen(tenue, 1))
        for k in range(50, 1000, 50):
            g.drawLine(self._a_pantalla(k, 10), self._a_pantalla(k, 990))

        # margen vertical en lápiz rojo
        margen = QColor(styles.LAPIZ_ROJO)
        margen.setAlpha(60)
        g.setPen(QPen(margen, 1.2))
        g.drawLine(self._a_pantalla(90, 10), self._a_pantalla(90, 990))

        # renglones (las mismas guías de Glifo); la franja de la altura x apenas sombreada
        franja = QColor(styles.NIEBLA)
        franja.setAlpha(110)
        g.setPen(Qt.NoPen)
        g.setBrush(franja)
        g.drawRect(QRectF(self._a_pantalla(90, GUIAS["altura x"]), self._a_pantalla(990, GUIAS["línea base"])))
        for nombre, y in GUIAS.items():
            base = nombre == "línea base"
            g.setPen(QPen(QColor(styles.RENGLON_FUERTE if base else styles.RENGLON), 1.6 if base else 1.0,
                          Qt.SolidLine if base else Qt.DashLine))
            g.drawLine(self._a_pantalla(90, y), self._a_pantalla(990, y))
            g.setPen(QColor(styles.GRAFITO))
            f = g.font()
            f.setPixelSize(11)
            g.setFont(f)
            g.drawText(QRectF(self._a_pantalla(100, y - 40), self._a_pantalla(600, y - 4)), Qt.AlignLeft | Qt.AlignBottom, nombre)

        grosor = max(3.0, a.width() / 130)

        # proyección: lápiz rojo punteado
        rojo = QColor(styles.LAPIZ_ROJO)
        pen = QPen(rojo, grosor * 0.75, Qt.CustomDashLine, Qt.RoundCap)
        pen.setDashPattern([0.1, 2.2])
        g.setPen(pen)
        for p0, p1, abajo in self.proyeccion:
            if abajo:
                g.drawLine(self._a_pantalla(*p0), self._a_pantalla(*p1))
        rojo.setAlpha(80)
        g.setPen(QPen(rojo, grosor * 0.35, Qt.DotLine, Qt.RoundCap))
        for p0, p1, abajo in self.proyeccion:
            if not abajo:
                g.drawLine(self._a_pantalla(*p0), self._a_pantalla(*p1))

        # trazos del usuario en tinta
        g.setPen(QPen(QColor(styles.TINTA), grosor, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        g.setBrush(Qt.NoBrush)
        for trazo in self.trazos:
            camino = QPainterPath(self._a_pantalla(*trazo[0]))
            for x, y in trazo[1:]:
                camino.lineTo(self._a_pantalla(x, y))
            if len(trazo) == 1:
                camino.lineTo(self._a_pantalla(trazo[0][0] + 0.5, trazo[0][1]))
            g.drawPath(camino)

        # inicio de cada trazo con su número de orden
        f = g.font()
        f.setPixelSize(int(grosor * 2.2))
        f.setBold(True)
        g.setFont(f)
        r = grosor * 1.6
        for i, trazo in enumerate(self.trazos, start=1):
            c = self._a_pantalla(*trazo[0])
            g.setPen(Qt.NoPen)
            g.setBrush(QColor(styles.TINTA))
            g.drawEllipse(c, r, r)
            g.setPen(QColor(styles.HOJA))
            g.drawText(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), Qt.AlignCenter, str(i))

        if not self.trazos:
            g.setPen(QColor(styles.GRAFITO))
            f.setPixelSize(15)
            f.setBold(False)
            g.setFont(f)
            g.drawText(QRectF(self._a_pantalla(90, 720), self._a_pantalla(990, 880)), Qt.AlignCenter,
                       "Escribe una letra minúscula sobre los renglones")
        g.end()
