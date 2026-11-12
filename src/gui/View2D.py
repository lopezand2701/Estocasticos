"""
View2D.py
=========
Ventana B: reconocedor y visualizador 2D (plano cartesiano).
Usa la vista común (RecognizerView) con un lienzo de cuaderno.

    trazos → FeatureExtractor → Recognizer (Forward + Bayes) → panel
                                         └→ proyección de la letra ganadora → lienzo
"""
from __future__ import annotations

from src.features import FeatureExtractor as FE
from src.gui.Canvas2D import Canvas2D
from src.gui.RecognizerView import RecognizerView


class View2D(RecognizerView):
    MODO = "2d"
    TITULO = "Plano"
    AYUDA = "Escribe con el clic izquierdo. Al soltar, la letra se evalúa completa."

    def crear_lienzo(self) -> Canvas2D:
        return Canvas2D()

    @property
    def canvas(self) -> Canvas2D:            # nombre usado en versiones anteriores
        return self.lienzo

    def simbolos(self) -> list[int]:
        return FE.strokes_to_symbols(self.lienzo.trazos, step=self.rec.step)

    def proyectar(self, letra: str | None) -> None:
        if letra is None:
            self.lienzo.set_proyeccion([])
            return
        escala = FE.escala(self.lienzo.trazos)          # la proyección usa el mismo tamaño normalizado
        self.lienzo.set_proyeccion(self.rec.project(letra, self.lienzo.ultimo_punto(), escala))
