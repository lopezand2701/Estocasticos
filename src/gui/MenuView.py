"""
MenuView.py
===========
Ventana A: menú principal de inicio. Permite elegir modo 2D o 3D.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget


class BotonModo(QPushButton):
    """
    Botón grande con título y descripción en dos etiquetas separadas.
    Así el alto se calcula bien con cualquier escala de pantalla de Windows
    (125 %, 150 %...) y el texto no se recorta.
    """

    def __init__(self, titulo: str, descripcion: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("modo")
        self.setCursor(Qt.PointingHandCursor)
        t = QLabel(titulo)
        t.setObjectName("modoTitulo")
        d = QLabel(descripcion)
        d.setObjectName("modoDesc")
        d.setWordWrap(True)
        for lbl in (t, d):
            lbl.setAttribute(Qt.WA_TransparentForMouseEvents)   # el clic lo recibe el botón
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(4)
        lay.addWidget(t)
        lay.addWidget(d)

    # El alto lo decide el contenido (ya con las fuentes del estilo aplicadas).
    def sizeHint(self):
        return self.layout().sizeHint()

    def minimumSizeHint(self):
        return self.layout().sizeHint()


class MenuView(QWidget):
    abrir_2d = Signal()
    abrir_3d = Signal()

    def __init__(self, hay_3d: bool = False, parent=None) -> None:
        super().__init__(parent)
        inst = QLabel("UNIVERSIDAD DE LOS LLANOS · PROCESOS ESTOCÁSTICOS")
        inst.setObjectName("seccion")
        titulo = QLabel("Reconocedor de caracteres con HMM")
        titulo.setObjectName("titulo")
        titulo.setWordWrap(True)
        sub = QLabel("Clasificación y autocompletado de trazos en tiempo real\n"
                     "mediante Cadenas Ocultas de Markov implementadas desde cero.")
        sub.setObjectName("subtitulo")

        b2d = BotonModo("Modo 2D", "Trazos en el plano cartesiano")
        b2d.clicked.connect(self.abrir_2d.emit)
        b3d = BotonModo("Modo 3D", "Trazos sobre la superficie de una esfera")
        b3d.setEnabled(hay_3d)
        b3d.clicked.connect(self.abrir_3d.emit)

        tarjeta = QFrame()
        tarjeta.setObjectName("tarjeta")
        tarjeta.setFixedWidth(560)
        lt = QVBoxLayout(tarjeta)
        lt.setContentsMargins(36, 34, 36, 34)
        lt.setSpacing(14)
        lt.addWidget(inst)
        lt.addWidget(titulo)
        lt.addWidget(sub)
        lt.addSpacing(12)
        lt.addWidget(b2d)
        lt.addWidget(b3d)

        raiz = QVBoxLayout(self)
        raiz.addStretch()
        raiz.addWidget(tarjeta, 0, Qt.AlignHCenter)
        raiz.addStretch()
