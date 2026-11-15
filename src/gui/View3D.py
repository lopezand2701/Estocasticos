"""
View3D.py
=========
Ventana C: reconocedor y visualizador 3D (lienzo esférico).
Misma vista común (RecognizerView) que el 2D, con la esfera como lienzo.

    trazos sobre la esfera → (θ, φ) → FeatureExtractor3D → Recognizer (Forward + Bayes) → panel
                                                     └→ proyección: se "camina" sobre la superficie
"""
from __future__ import annotations

from PySide6.QtWidgets import QPushButton

from src.features import FeatureExtractor3D as F3
from src.gui.RecognizerView import RecognizerView
from src.gui.Sphere3D import Sphere3D


class View3D(RecognizerView):
    MODO = "3d"
    TITULO = "Esfera"
    AYUDA = ("Clic izquierdo: escribir. Clic derecho o Shift + clic: girar 360°. Rueda: zoom. "
             "Clic en una flecha de la tira: ver dónde está en la esfera.")

    def crear_lienzo(self) -> Sphere3D:
        return Sphere3D()

    @property
    def esfera(self) -> Sphere3D:
        return self.lienzo

    def acciones_extra(self) -> list[QPushButton]:
        b_orientar = QPushButton("Reorientar")
        b_orientar.clicked.connect(lambda: self.lienzo.reorientar())
        self.b_auto = QPushButton("Girar sola")
        self.b_auto.setCheckable(True)
        self.b_auto.toggled.connect(lambda on: self.lienzo.auto.start(30) if on else self.lienzo.auto.stop())
        return [b_orientar, self.b_auto]

    def simbolos_y_segmentos(self):
        return F3.simbolos_y_segmentos(self.lienzo.trazos_escritura())

    def resaltar(self, segmento, texto: str, arriba: bool) -> None:
        self.lienzo.set_resaltado_escritura(segmento, texto, arriba)

    def proyectar(self, letra: str | None) -> None:
        if letra is None:
            self.lienzo.set_proyeccion_escritura([])
            return
        trazos = self.lienzo.trazos_escritura()
        paso = F3.STEP / F3.escala(trazos)
        segmentos, p = [], self.lienzo.ultimo_punto_escritura()
        for vec, abajo in self.rec.project_steps(letra, completo=not self.lienzo.dibujando):
            q = F3.mover(p, vec[0] * paso, vec[1] * paso)       # vec = (este, sur) en "pasos"
            segmentos.append((p, q, abajo))
            p = q
        self.lienzo.set_proyeccion_escritura(segmentos)
