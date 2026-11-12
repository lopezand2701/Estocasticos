"""
View3D.py
=========
Ventana C: reconocedor y visualizador 3D (lienzo esférico).
Misma estructura que la Ventana B:
    Izquierda: esfera para dibujar (gira 360°, zoom) + proyección sobre la superficie.
    Derecha:   letra más probable + lista ordenada de probabilidades.

    trazos sobre la esfera → (θ, φ) → FeatureExtractor3D → Recognizer (Forward + Bayes) → panel
                                                     └→ proyección: se "camina" sobre la superficie
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from src.features import FeatureExtractor as FE
from src.features import FeatureExtractor3D as F3
from src.gui.ProbabilityPanel import ProbabilityPanel
from src.gui.Sphere3D import Sphere3D
from src.hmm.Recognizer import Recognizer

MARGEN_CAMBIO = 0.10   # histéresis, igual que en 2D


class View3D(QWidget):
    volver = Signal()

    def __init__(self, recognizer: Recognizer, parent=None) -> None:
        super().__init__(parent)
        self.rec = recognizer
        self.letra_proyectada: str | None = None

        # ---------------- lado izquierdo ----------------
        self.esfera = Sphere3D()
        self.esfera.trazos_cambiaron.connect(self._actualizar)

        b_volver = QPushButton("← Menú")
        b_deshacer = QPushButton("Deshacer trazo")
        b_limpiar = QPushButton("Limpiar lienzo")
        b_reiniciar = QPushButton("Reiniciar captura")
        b_orientar = QPushButton("Reorientar")
        self.b_auto = QPushButton("Auto-giro 360°")
        self.b_auto.setCheckable(True)
        b_volver.clicked.connect(self.volver.emit)
        b_deshacer.clicked.connect(self.esfera.deshacer)
        b_limpiar.clicked.connect(self.esfera.limpiar)
        b_reiniciar.clicked.connect(self.reiniciar)
        b_orientar.clicked.connect(self.esfera.reorientar)
        self.b_auto.toggled.connect(lambda on: self.esfera.auto.start(30) if on else self.esfera.auto.stop())

        titulo = QLabel("Reconocedor 3D · lienzo esférico")
        titulo.setObjectName("seccion")
        self.info = QLabel("Dibuja una letra minúscula sobre la esfera")
        self.info.setObjectName("info")

        barra = QHBoxLayout()
        barra.addWidget(b_volver)
        barra.addStretch()
        for b in (b_orientar, self.b_auto, b_deshacer, b_limpiar, b_reiniciar):
            barra.addWidget(b)

        izq = QFrame()
        izq.setObjectName("tarjeta")
        li = QVBoxLayout(izq)
        li.setContentsMargins(18, 16, 18, 16)
        li.addWidget(titulo)
        li.addWidget(self.esfera, 1)
        li.addWidget(self.info)
        li.addLayout(barra)

        # ---------------- lado derecho ----------------
        cab = QLabel("P(carácter | O₁:ₜ) · 3D")
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
        self.esfera.limpiar()

    def _actualizar(self) -> None:
        """Se llama cada vez que cambia el dibujo sobre la esfera."""
        trazos = self.esfera.trazos_escritura()
        simbolos = F3.strokes_to_symbols(trazos)
        self.rec.update(simbolos)
        ranking = self.rec.posteriors(completo=not self.esfera.dibujando)
        self.panel.set_datos(ranking)

        if not simbolos:
            self.letra_proyectada = None
            self.prediccion.setText("–")
            self.detalle.setText("")
            self.info.setText("Dibuja una letra minúscula sobre la esfera")
            self.esfera.set_proyeccion_escritura([])
            return

        probs = dict(ranking)
        mejor, p_mejor = ranking[0]
        actual = self.letra_proyectada
        if actual is None or p_mejor > probs.get(actual, 0.0) + MARGEN_CAMBIO:
            self.letra_proyectada = mejor
        letra = self.letra_proyectada

        self.prediccion.setText(letra)
        self.detalle.setText(f"{probs[letra] * 100:.1f} %  ·  proyectando «{letra}»")
        ultimo = self.esfera.ultimo_punto_escritura()
        th, ph = F3.a_esfericas(ultimo)
        self.info.setText(f"T = {len(simbolos)} símbolos   ·   último punto θ = {np.degrees(th[0]):.1f}°, "
                          f"φ = {np.degrees(ph[0]):.1f}°   ·   últimos: {FE.describe(simbolos[-6:])}")

        # Proyección: se avanza sobre la superficie paso a paso (mapa exponencial)
        paso = F3.STEP / F3.escala(trazos)
        segmentos, p = [], ultimo
        for vec, abajo in self.rec.project_steps(letra):
            q = F3.mover(p, vec[0] * paso, vec[1] * paso)       # vec = (este, sur) en "pasos"
            segmentos.append((p, q, abajo))
            p = q
        self.esfera.set_proyeccion_escritura(segmentos)
