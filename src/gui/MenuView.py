"""
MenuView.py
===========
Ventana A: menú principal de inicio. Permite elegir modo 2D (plano) o 3D (esfera).
Cada modo es una tarjeta grande con una ilustración dibujada a mano con QPainter.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from src.gui import glifos, styles


class TarjetaModo(QPushButton):
    """Botón-tarjeta: ilustración arriba, título y descripción abajo."""

    def __init__(self, modo: str, titulo: str, descripcion: str, parent=None) -> None:
        super().__init__(parent)
        self.modo = modo
        self.titulo = titulo
        self.descripcion = descripcion
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(250, 330)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("QPushButton { background: transparent; border: none; padding: 0; }")

    def paintEvent(self, _) -> None:
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        activa = self.isEnabled()
        encima = self.underMouse() and activa
        foco = self.hasFocus()

        # la hoja de la tarjeta
        g.setPen(QPen(QColor(styles.TINTA if (encima or foco) else styles.RENGLON), 1.6 if (encima or foco) else 1))
        g.setBrush(QColor(styles.HOJA))
        g.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 18, 18)

        ilus = QRectF(18, 18, w - 36, h * 0.62)
        g.setPen(Qt.NoPen)
        g.setBrush(QColor(styles.PAPEL))
        g.drawRoundedRect(ilus, 12, 12)
        if self.modo == "2d":
            self._ilustracion_plano(g, ilus)
        else:
            self._ilustracion_esfera(g, ilus)

        f = QFont(self.font())
        f.setPixelSize(22)
        f.setWeight(QFont.DemiBold)
        g.setFont(f)
        g.setPen(QColor(styles.TINTA if activa else styles.RENGLON_FUERTE))
        g.drawText(QRectF(24, ilus.bottom() + 16, w - 48, 30), Qt.AlignLeft | Qt.AlignVCenter, self.titulo)
        f.setPixelSize(14)
        f.setWeight(QFont.Normal)
        g.setFont(f)
        g.setPen(QColor(styles.GRAFITO if activa else styles.RENGLON_FUERTE))
        texto = self.descripcion if activa else "Entrena primero los modelos 3D: python -m training.train --modo 3d"
        g.drawText(QRectF(24, ilus.bottom() + 48, w - 48, h - ilus.bottom() - 56),
                   Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, texto)
        g.end()

    def _ilustracion_plano(self, g: QPainter, r: QRectF) -> None:
        # renglones de cuaderno (ascendente, altura x, base, descendente)
        caja = QRectF(r.center().x() - r.height() * 0.45, r.top() + r.height() * 0.03,
                      r.height() * 0.9, r.height() * 0.94)
        esc = caja.height() / 760
        y_de = lambda y: caja.top() + (y - 160) * esc
        for y, fuerte in ((180, False), (420, False), (700, True), (900, False)):
            g.setPen(QPen(QColor(styles.RENGLON_FUERTE if fuerte else styles.RENGLON), 1.4 if fuerte else 1,
                          Qt.SolidLine if fuerte else Qt.DashLine))
            g.drawLine(QPointF(r.left() + 16, y_de(y)), QPointF(r.right() - 16, y_de(y)))
        # la "a": la barriga ya escrita en tinta y el palito que propone el modelo en lápiz rojo
        trazos = glifos.trazos_guia().get("a", [])
        ox = caja.center().x() - 510 * esc
        a_px = lambda x, y: QPointF(ox + x * esc, y_de(y))
        for i, t in enumerate(trazos):
            if i == 0:
                g.setPen(QPen(QColor(styles.TINTA), 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            else:
                pen = QPen(QColor(styles.LAPIZ_ROJO), 5, Qt.CustomDashLine, Qt.RoundCap)
                pen.setDashPattern([0.1, 2.2])
                g.setPen(pen)
            for (x0, y0), (x1, y1) in zip(t, t[1:]):
                g.drawLine(a_px(x0, y0), a_px(x1, y1))

    def _ilustracion_esfera(self, g: QPainter, r: QRectF) -> None:
        cx, cy = r.center().x(), r.center().y()
        rad = min(r.width(), r.height()) * 0.40
        grad = QRadialGradient(QPointF(cx - rad * 0.4, cy - rad * 0.45), rad * 1.6)
        grad.setColorAt(0.0, QColor("#FFFFFF"))
        grad.setColorAt(0.55, QColor("#EEF2F7"))
        grad.setColorAt(1.0, QColor("#C9D5E2"))
        g.setPen(QPen(QColor(styles.RENGLON_FUERTE), 1.2))
        g.setBrush(grad)
        g.drawEllipse(QPointF(cx, cy), rad, rad)
        # meridianos y paralelos inclinados
        tinta = QColor(styles.TINTA)
        tinta.setAlpha(45)
        g.setPen(QPen(tinta, 1))
        g.setBrush(Qt.NoBrush)
        inc = np.radians(20)
        for lon in np.radians(np.arange(-60, 61, 30)):
            pts = []
            for t in np.linspace(-np.pi / 2, np.pi / 2, 40):
                x, y, z = np.cos(t) * np.sin(lon), np.sin(t), np.cos(t) * np.cos(lon)
                y2, z2 = y * np.cos(inc) - z * np.sin(inc), y * np.sin(inc) + z * np.cos(inc)
                pts.append(QPointF(cx + rad * x, cy - rad * y2))
            for a, b in zip(pts, pts[1:]):
                g.drawLine(a, b)
        for lat in np.radians([-30, 0, 30]):
            ry = rad * np.cos(lat) * np.sin(inc)
            g.drawEllipse(QPointF(cx, cy - rad * np.sin(lat) * np.cos(inc)), rad * np.cos(lat), abs(ry) + 0.5)
        # letra escrita sobre la bola
        glifos.dibujar_letra(g, "k", QRectF(cx - rad * 0.42, cy - rad * 0.62, rad * 0.84, rad * 1.24),
                             styles.TINTA, grosor=4.5)
        # flecha curva de giro
        g.setPen(QPen(QColor(styles.LAPIZ_ROJO), 2, Qt.SolidLine, Qt.RoundCap))
        arco = QRectF(cx - rad * 1.18, cy - rad * 0.35, rad * 2.36, rad * 0.7)
        g.drawArc(arco, 200 * 16, 140 * 16)
        punta = QPointF(cx + rad * 1.1, cy + rad * 0.14)
        g.drawLine(punta, punta + QPointF(-9, -1))
        g.drawLine(punta, punta + QPointF(-3, -8))


class TarjetaGuia(QPushButton):
    """Tarjeta ancha: Estandarización del alfabeto (cómo se escribe cada letra)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(118)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet("QPushButton { background: transparent; border: none; padding: 0; }")

    def paintEvent(self, _) -> None:
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        encima = self.underMouse() or self.hasFocus()
        g.setPen(QPen(QColor(styles.TINTA if encima else styles.RENGLON), 1.6 if encima else 1))
        g.setBrush(QColor(styles.HOJA))
        g.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 18, 18)
        ilus = QRectF(14, 12, 230, h - 24)
        g.setPen(Qt.NoPen)
        g.setBrush(QColor(styles.PAPEL))
        g.drawRoundedRect(ilus, 12, 12)
        for i, letra in enumerate("akt"):
            caja = QRectF(ilus.left() + 14 + i * 70, ilus.top() + 4, 62, ilus.height() - 8)
            glifos.dibujar_letra(g, letra, caja, styles.TINTA, grosor=3.2, numeros=True, linea_base=True)
        f = QFont(self.font())
        f.setPixelSize(20)
        f.setWeight(QFont.DemiBold)
        g.setFont(f)
        g.setPen(QColor(styles.TINTA))
        x = ilus.right() + 22
        g.drawText(QRectF(x, 18, w - x - 20, 30), Qt.AlignLeft | Qt.AlignVCenter, "Estandarización del alfabeto")
        f.setPixelSize(14)
        f.setWeight(QFont.Normal)
        g.setFont(f)
        g.setPen(QColor(styles.GRAFITO))
        g.drawText(QRectF(x, 50, w - x - 20, h - 58), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
                   "Mira cómo se traza cada letra, en orden y con sus números, y practica encima del calco.")
        g.end()


class MenuView(QWidget):
    abrir_2d = Signal()
    abrir_3d = Signal()
    abrir_guia = Signal()

    def __init__(self, hay_3d: bool = False, parent=None) -> None:
        super().__init__(parent)
        titulo = QLabel("Escribe una letra. Una cadena oculta de Markov adivina cuál es.")
        titulo.setObjectName("titulo")
        titulo.setWordWrap(True)
        bajada = QLabel("Mientras escribes, 27 modelos ocultos de Markov comparan tu trazo "
                        "con las letras del alfabeto español, ordenan las más probables y "
                        "dibujan en lápiz rojo cómo seguiría la letra.")
        bajada.setObjectName("bajada")
        bajada.setWordWrap(True)
        bajada.setMaximumWidth(460)
        curso = QLabel("Proyecto final de Procesos Estocásticos\nIngeniería de Sistemas, Universidad de los Llanos")
        curso.setObjectName("nota")

        izq = QVBoxLayout()
        izq.addStretch(2)
        izq.addWidget(titulo)
        izq.addSpacing(18)
        izq.addWidget(bajada)
        izq.addStretch(3)
        izq.addWidget(curso)

        t2d = TarjetaModo("2d", "Plano", "Escribe sobre los renglones de un cuaderno, como en papel.")
        t3d = TarjetaModo("3d", "Esfera", "Escribe sobre una bola que puedes girar en cualquier dirección.")
        t3d.setEnabled(hay_3d)
        t2d.clicked.connect(self.abrir_2d.emit)
        t3d.clicked.connect(self.abrir_3d.emit)
        tarjetas = QHBoxLayout()
        tarjetas.setSpacing(18)
        tarjetas.addWidget(t2d)
        tarjetas.addWidget(t3d)

        raiz = QHBoxLayout(self)
        raiz.setContentsMargins(56, 44, 56, 44)
        raiz.setSpacing(56)
        raiz.addLayout(izq, 4)
        guia = TarjetaGuia()
        guia.clicked.connect(self.abrir_guia.emit)
        der = QVBoxLayout()
        der.setSpacing(18)
        der.addStretch(1)
        der.addLayout(tarjetas, 6)
        der.addWidget(guia)
        der.addStretch(1)
        raiz.addLayout(der, 7)
