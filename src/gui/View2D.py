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
    AYUDA = ("Escribe con el clic izquierdo. Al soltar, la letra se evalúa completa. "
             "Haz clic en una flecha de la tira para ver de qué parte del trazo salió.")

    def crear_lienzo(self) -> Canvas2D:
        return Canvas2D()

    @property
    def canvas(self) -> Canvas2D:            # nombre usado en versiones anteriores
        return self.lienzo

    def simbolos_y_segmentos(self):
        return FE.simbolos_y_segmentos(self.lienzo.trazos, step=self.rec.step)

    def resaltar(self, segmento, texto: str, arriba: bool) -> None:
        self.lienzo.set_resaltado(segmento, texto, arriba)

    def proyectar(self, letra: str | None) -> None:
        if letra is None:
            self.lienzo.set_proyeccion([])
            return
        escala = FE.escala(self.lienzo.trazos)          # la proyección usa el mismo tamaño normalizado
        completo = not self.lienzo.dibujando
        self.lienzo.set_proyeccion(self.rec.project(letra, self.lienzo.ultimo_punto(), escala, completo))
