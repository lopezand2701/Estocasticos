"""
View2D.py
=========
Ventana B: reconocedor y visualizador 2D.
    Izquierda: lienzo de trazado + proyección dinámica.
    Derecha:   letra más probable + lista ordenada de probabilidades.

Conecta la interfaz con la lógica:
    trazos → FeatureExtractor → Recognizer (Forward + Bayes) → panel
                                         └→ proyección de la letra ganadora → lienzo
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from src.features import FeatureExtractor as FE
from src.gui.Canvas2D import Canvas2D
from src.gui.ProbabilityPanel import ProbabilityPanel
from src.hmm.Recognizer import Recognizer

MARGEN_CAMBIO = 0.10   # histéresis: otra letra debe superar por 10 puntos para cambiar la proyección


class View2D(QWidget):
    volver = Signal()

    def __init__(self, recognizer: Recognizer, parent=None) -> None:
        super().__init__(parent)
        self.rec = recognizer
        self.letra_proyectada: str | None = None

        # ---------------- lado izquierdo ----------------
        self.canvas = Canvas2D()
        self.canvas.trazos_cambiaron.connect(self._actualizar)

        b_volver = QPushButton("← Menú")
        b_deshacer = QPushButton("Deshacer trazo")
        b_limpiar = QPushButton("Limpiar lienzo")
        b_reiniciar = QPushButton("Reiniciar captura")
        b_volver.clicked.connect(self.volver.emit)
        b_deshacer.clicked.connect(self.canvas.deshacer)
        b_limpiar.clicked.connect(self.canvas.limpiar)
        b_reiniciar.clicked.connect(self.reiniciar)

        titulo = QLabel("Reconocedor 2D")
        titulo.setObjectName("seccion")
        self.info = QLabel("Dibuja una letra minúscula")
        self.info.setObjectName("info")

        barra = QHBoxLayout()
        barra.addWidget(b_volver)
        barra.addStretch()
        for b in (b_deshacer, b_limpiar, b_reiniciar):
            barra.addWidget(b)

        izq = QFrame()
        izq.setObjectName("tarjeta")
        li = QVBoxLayout(izq)
        li.setContentsMargins(18, 16, 18, 16)
        li.addWidget(titulo)
        li.addWidget(self.canvas, 1)
        li.addWidget(self.info)
        li.addLayout(barra)

        # ---------------- lado derecho ----------------
        cab = QLabel("P(carácter | O₁:ₜ)")
        cab.setObjectName("seccion")
        self.prediccion = QLabel("–")
        self.prediccion.setObjectName("prediccion")
        self.prediccion.setAlignment(Qt.AlignCenter)
        self.detalle = QLabel("")
        self.detalle.setObjectName("info")
        self.detalle.setAlignment(Qt.AlignCenter)

        self.panel = ProbabilityPanel(self.rec.letters)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        der = QFrame()
        der.setObjectName("tarjeta")
        der.setFixedWidth(360)
        ld = QVBoxLayout(der)
        ld.setContentsMargins(18, 16, 18, 16)
        ld.addWidget(cab)
        ld.addWidget(self.prediccion)
        ld.addWidget(self.detalle)
        ld.addWidget(scroll, 1)

        raiz = QHBoxLayout(self)
        raiz.setContentsMargins(20, 20, 20, 20)
        raiz.setSpacing(18)
        raiz.addWidget(izq, 1)
        raiz.addWidget(der)

    # ------------------------------------------------------------------ #
    def reiniciar(self) -> None:
        self.rec.reset()
        self.letra_proyectada = None
        self.canvas.limpiar()

    def _actualizar(self) -> None:
        """Se llama cada vez que cambia el dibujo (cada movimiento del mouse)."""
        simbolos = FE.strokes_to_symbols(self.canvas.trazos, step=self.rec.step)
        self.rec.update(simbolos)
        # Lápiz abajo: "¿es el comienzo de la letra?"  ·  Lápiz arriba: "¿es la letra completa?"
        ranking = self.rec.posteriors(completo=not self.canvas.dibujando)
        self.panel.set_datos(ranking)

        if not simbolos:
            self.letra_proyectada = None
            self.prediccion.setText("–")
            self.detalle.setText("")
            self.info.setText("Dibuja una letra minúscula")
            self.canvas.set_proyeccion([])
            return

        # histéresis: solo cambiar la letra proyectada si otra la supera con margen
        probs = dict(ranking)
        mejor, p_mejor = ranking[0]
        actual = self.letra_proyectada
        if actual is None or p_mejor > probs.get(actual, 0.0) + MARGEN_CAMBIO:
            self.letra_proyectada = mejor
        letra = self.letra_proyectada

        self.prediccion.setText(letra)
        self.detalle.setText(f"{probs[letra] * 100:.1f} %  ·  proyectando «{letra}»")
        self.info.setText(f"T = {len(simbolos)} símbolos   ·   últimos: {FE.describe(simbolos[-8:])}")
        self.canvas.set_proyeccion(self.rec.project(letra, self.canvas.ultimo_punto()))
