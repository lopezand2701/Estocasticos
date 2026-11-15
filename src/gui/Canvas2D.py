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
        self.resaltado = None                      # (p0, p1, texto, lapiz_arriba) de la flecha elegida
        self.plantilla: list | None = None         # trazos de una letra modelo para calcar (modo práctica)
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

    def set_resaltado(self, segmento, texto: str = "", arriba: bool = False) -> None:
        """Marca en la hoja el tramo que produjo la flecha seleccionada en la tira O₁…O_T."""
        self.resaltado = None if segmento is None else (segmento[0], segmento[1], texto, arriba)
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

        # letra modelo "en calco" (sección Estandarización del alfabeto)
        if self.plantilla:
            calco = QColor(styles.RENGLON_FUERTE)
            calco.setAlpha(150)
            g.setPen(QPen(calco, grosor * 3.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            g.setBrush(Qt.NoBrush)
            for t in self.plantilla:
                camino = QPainterPath(self._a_pantalla(*t[0]))
                for x, y in t[1:]:
                    camino.lineTo(self._a_pantalla(x, y))
                if len(t) == 1:
                    camino.lineTo(self._a_pantalla(t[0][0] + 0.5, t[0][1]))
                g.drawPath(camino)

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

        if self.resaltado is not None:
            self._dibujar_resaltado(g, grosor)

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
            self.resaltado = None
            g.setPen(QColor(styles.GRAFITO))
            f.setPixelSize(15)
            f.setBold(False)
            g.setFont(f)
            g.drawText(QRectF(self._a_pantalla(90, 720), self._a_pantalla(990, 880)), Qt.AlignCenter,
                       "Escribe una letra minúscula sobre los renglones")
        g.end()

    def _dibujar_resaltado(self, g: QPainter, grosor: float) -> None:
        """Resaltador amarillo sobre el tramo, flecha roja encima y una etiqueta con O_t."""
        p0, p1, texto, arriba = self.resaltado
        a, b = self._a_pantalla(*p0), self._a_pantalla(*p1)
        amarillo = QColor(styles.RESALTADOR)
        amarillo.setAlpha(210)
        g.setPen(QPen(amarillo, grosor * 4.2, Qt.SolidLine, Qt.RoundCap))
        g.drawLine(a, b)
        rojo = QColor(styles.LAPIZ_ROJO)
        g.setPen(QPen(rojo, grosor * 0.9, Qt.DashLine if arriba else Qt.SolidLine, Qt.RoundCap))
        g.drawLine(a, b)
        # punta de flecha en p1
        d = b - a
        largo = max((d.x() ** 2 + d.y() ** 2) ** 0.5, 1e-6)
        ux, uy = d.x() / largo, d.y() / largo
        tam = grosor * 2.6
        for sx in (1, -1):
            g.drawLine(b, QPointF(b.x() - tam * (ux * 0.87 - sx * uy * 0.5), b.y() - tam * (uy * 0.87 + sx * ux * 0.5)))
        # etiqueta
        if texto:
            f = g.font()
            f.setPixelSize(12)
            f.setBold(True)
            g.setFont(f)
            ancho = g.fontMetrics().horizontalAdvance(texto) + 14
            caja = QRectF(b.x() + 10, b.y() - 30, ancho, 22)
            if caja.right() > self.width() - 4:
                caja.moveRight(a.x() - 10)
            g.setPen(Qt.NoPen)
            g.setBrush(QColor(styles.TINTA))
            g.drawRoundedRect(caja, 6, 6)
            g.setPen(QColor(styles.HOJA))
            g.drawText(caja, Qt.AlignCenter, texto)
