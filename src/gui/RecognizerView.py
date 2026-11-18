"""
RecognizerView.py
=================
Vista común de las Ventanas B (2D) y C (3D). Solo cambia el lienzo:

    ┌ barra: ← Menú · título · selector Plano/Esfera ───────────────────────┐
    │ ┌ hoja ──────────────────────────────┐  ┌ lectura ──────────────────┐ │
    │ │ lienzo (cuaderno o esfera)         │  │ letra ganadora + %        │ │
    │ │                                    │  │ cadena de estados en vivo │ │
    │ │ O₁…Oₜ  ↓ ↓ ↘ → →                   │  │ todas las letras (lista)  │ │
    │ │ acciones                           │  │                           │ │
    │ └────────────────────────────────────┘  └───────────────────────────┘ │
    └────────────────────────────────────────────────────────────────────────┘

Flujo: trazos → símbolos → Recognizer (Forward + Bayes) → lista y letra ganadora
                                      └→ fase actual (cadena) y proyección (lápiz rojo)
Las subclases definen el lienzo, cómo se obtienen los símbolos y cómo se proyecta.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

from src.features import FeatureExtractor as FE
from src.gui import glifos, styles
from src.gui.ObservationStrip import ObservationStrip
from src.gui.ProbabilityPanel import ProbabilityPanel
from src.gui.StateChain import StateChain
from src.hmm.Recognizer import Recognizer

MARGEN_CAMBIO = 0.10   # histéresis: otra letra debe superar por 10 puntos para cambiar la proyección


class LetraGanadora(QWidget):
    """La letra más probable, dibujada con sus trazos numerados sobre un toque de resaltador."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.letra: str | None = None
        self.setFixedSize(120, 132)

    def set_letra(self, letra: str | None) -> None:
        self.letra = letra
        self.update()

    def paintEvent(self, _) -> None:
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if self.letra:
            # trazo de resaltador irregular, a la altura de la x
            mancha = QPainterPath()
            mancha.moveTo(14, h * 0.50)
            mancha.cubicTo(w * 0.35, h * 0.42, w * 0.7, h * 0.47, w - 10, h * 0.44)
            mancha.lineTo(w - 6, h * 0.76)
            mancha.cubicTo(w * 0.65, h * 0.80, w * 0.3, h * 0.74, 10, h * 0.79)
            mancha.closeSubpath()
            g.setPen(Qt.NoPen)
            g.setBrush(QColor(styles.RESALTADOR))
            g.drawPath(mancha)
            glifos.dibujar_letra(g, self.letra, QRectF(14, 16, w - 28, h - 26), styles.TINTA,
                                 grosor=4.5, numeros=True, linea_base=True)
        else:
            g.setPen(QColor(styles.RENGLON_FUERTE))
            g.setBrush(Qt.NoBrush)
            g.drawRoundedRect(QRectF(16, 14, w - 32, h - 28), 12, 12)
            g.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "?")
        g.end()


class RecognizerView(QWidget):
    volver = Signal()
    cambiar_modo = Signal(str)            # "2d" o "3d"

    MODO = "2d"
    TITULO = ""
    AYUDA = ""

    def __init__(self, recognizer: Recognizer, hay_3d: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.rec = recognizer
        self.letra_proyectada: str | None = None
        self.lienzo = self.crear_lienzo()
        self.lienzo.trazos_cambiaron.connect(self._actualizar)

        # ---------------- barra superior ----------------
        b_menu = QPushButton("← Menú")
        b_menu.clicked.connect(self.volver.emit)
        titulo = QLabel(self.TITULO)
        titulo.setObjectName("marca")
        selector = QFrame()
        selector.setObjectName("selector")
        ls = QHBoxLayout(selector)
        ls.setContentsMargins(3, 3, 3, 3)
        ls.setSpacing(2)
        for modo, texto in (("2d", "Plano"), ("3d", "Esfera")):
            b = QPushButton(texto)
            b.setObjectName("segmento")
            b.setCheckable(True)
            b.setChecked(modo == self.MODO)
            b.setEnabled(modo == "2d" or hay_3d)
            b.clicked.connect(lambda _=False, m=modo: self._pedir_modo(m))
            ls.addWidget(b)
        barra = QHBoxLayout()
        barra.addWidget(b_menu)
        barra.addSpacing(14)
        barra.addWidget(titulo)
        barra.addStretch()
        barra.addWidget(selector)

        # ---------------- hoja izquierda ----------------
        self.observaciones = ObservationStrip()
        self.observaciones.seleccion.connect(self._mostrar_simbolo)
        self._segmentos: list = []
        ayuda = QLabel(self.AYUDA)
        ayuda.setObjectName("nota")
        acciones = QHBoxLayout()
        acciones.addStretch()
        for b in self.acciones_extra():
            acciones.addWidget(b)
        for texto, fn in (("Deshacer trazo", self.lienzo.deshacer),
                          ("Limpiar", self.lienzo.limpiar),
                          ("Reiniciar captura", self.reiniciar)):
            b = QPushButton(texto)
            b.clicked.connect(fn)
            acciones.addWidget(b)

        hoja = QFrame()
        hoja.setObjectName("hoja")
        lh = QVBoxLayout(hoja)
        lh.setContentsMargins(18, 18, 18, 14)
        lh.setSpacing(10)
        lh.addWidget(self.lienzo, 1)
        lh.addWidget(self.observaciones)
        lh.addWidget(ayuda)
        lh.addLayout(acciones)

        # ---------------- lectura (derecha) ----------------
        cab = QLabel("Lectura")
        cab.setObjectName("seccion")
        self.ganadora = LetraGanadora()
        self.porcentaje = QLabel("")
        self.porcentaje.setObjectName("porcentaje")
        self.explicacion = QLabel("Empieza a escribir y aquí verás qué letra cree el modelo que es.")
        self.explicacion.setObjectName("nota")
        self.explicacion.setWordWrap(True)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addStretch()
        col.addWidget(self.porcentaje)
        col.addWidget(self.explicacion)
        col.addStretch()
        fila = QHBoxLayout()
        fila.setSpacing(14)
        fila.addWidget(self.ganadora)
        fila.addLayout(col, 1)

        cab_cadena = QLabel("Fases ocultas de la letra")
        cab_cadena.setObjectName("seccion")
        self.cadena = StateChain()
        self.nota_cadena = QLabel("")
        self.nota_cadena.setObjectName("nota")

        cab_lista = QLabel("Todas las letras, de más a menos probable")
        cab_lista.setObjectName("seccion")
        self.panel = ProbabilityPanel(self.rec.letters)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        lectura = QFrame()
        lectura.setObjectName("hoja")
        lectura.setFixedWidth(390)
        ll = QVBoxLayout(lectura)
        ll.setContentsMargins(20, 18, 14, 14)
        ll.setSpacing(8)
        ll.addWidget(cab)
        ll.addLayout(fila)
        ll.addSpacing(6)
        ll.addWidget(cab_cadena)
        ll.addWidget(self.cadena)
        ll.addWidget(self.nota_cadena)
        ll.addSpacing(6)
        ll.addWidget(cab_lista)
        ll.addWidget(scroll, 1)

        cuerpo = QHBoxLayout()
        cuerpo.setSpacing(18)
        cuerpo.addWidget(hoja, 1)
        cuerpo.addWidget(lectura)

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(22, 16, 22, 22)
        raiz.setSpacing(14)
        raiz.addLayout(barra)
        raiz.addLayout(cuerpo, 1)

    # ---------------- para las subclases ----------------
    def crear_lienzo(self) -> QWidget:
        raise NotImplementedError

    def simbolos_y_segmentos(self) -> tuple[list[int], list]:
        """Símbolos O_1…O_T y, para cada uno, el tramo del lienzo que lo produjo."""
        raise NotImplementedError

    def resaltar(self, segmento, texto: str, arriba: bool) -> None:
        """Marca en el lienzo el tramo de una flecha (o lo quita si segmento es None)."""
        raise NotImplementedError

    def proyectar(self, letra: str | None) -> None:
        raise NotImplementedError

    def acciones_extra(self) -> list[QPushButton]:
        return []

    # ---------------- lógica común ----------------
    def _pedir_modo(self, modo: str) -> None:
        if modo != self.MODO:
            self.cambiar_modo.emit(modo)
        for b in self.findChildren(QPushButton, "segmento"):
            b.setChecked(b.text() == ("Plano" if self.MODO == "2d" else "Esfera"))

    def _mostrar_simbolo(self, t) -> None:
        """Clic en una flecha de la tira: se resalta su tramo en el plano o en la esfera."""
        if t is None or t >= len(self._segmentos):
            self.resaltar(None, "", False)
            return
        s = self.observaciones.simbolos[t]
        texto = f"O{''.join('₀₁₂₃₄₅₆₇₈₉'[int(d)] for d in str(t + 1))} {FE.FLECHAS[s % FE.K]}"
        self.resaltar(self._segmentos[t], texto, s >= FE.PEN_UP)

    def reiniciar(self) -> None:
        self.rec.reset()
        self.letra_proyectada = None
        self.lienzo.limpiar()

    def _actualizar(self) -> None:
        """Se llama cada vez que cambia el dibujo (cada movimiento del mouse)."""
        simbolos, self._segmentos = self.simbolos_y_segmentos()
        self.rec.update(simbolos)
        completo = not self.lienzo.dibujando
        # Lápiz abajo: "¿es el comienzo de la letra?"  ·  Lápiz arriba: "¿es la letra completa?"
        ranking = self.rec.posteriors(completo=completo)
        self.observaciones.set_simbolos(simbolos)

        if not simbolos:
            self.letra_proyectada = None
            self.panel.set_datos(ranking, activo=False)
            self.ganadora.set_letra(None)
            self.porcentaje.setText("")
            self.explicacion.setText("Empieza a escribir y aquí verás qué letra cree el modelo que es.")
            self.cadena.set_estado(None)
            self.nota_cadena.setText("")
            self.proyectar(None)
            return

        motivo = self.rec.rechazo(ranking[0][0], completo)
        if motivo:                                   # no parece ninguna letra (por ejemplo, un garabato)
            self.letra_proyectada = None
            self.panel.set_datos(ranking, activo=False)
            self.ganadora.set_letra(None)
            self.porcentaje.setText("No es una letra")
            self.explicacion.setText(f"El trazo {motivo}.")
            self.cadena.set_estado(None)
            self.nota_cadena.setText("")
            self.proyectar(None)
            return

        self.panel.set_datos(ranking)
        probs = dict(ranking)
        mejor, p_mejor = ranking[0]
        if self.letra_proyectada is None or p_mejor > probs.get(self.letra_proyectada, 0.0) + MARGEN_CAMBIO:
            self.letra_proyectada = mejor
        letra = self.letra_proyectada

        self.ganadora.set_letra(letra)
        self.porcentaje.setText(f"{probs[letra] * 100:.1f} %")
        if completo:
            self.explicacion.setText(f"Con el trazo completo, lo más probable es una «{letra}».")
        else:
            self.explicacion.setText(f"Mientras escribes parece el comienzo de una «{letra}». "
                                     "La línea roja muestra cómo seguiría.")

        estado = self.rec.estado_actual(letra, completo)
        self.cadena.set_estado(estado)
        if estado:
            variante = (f", forma {estado['variante']} de {estado['variantes']}"
                        if estado["variantes"] > 1 else "")
            self.nota_cadena.setText(f"Fase {estado['fase'] + 1} de {estado['N']} del modelo de «{letra}»{variante}.")
        self.proyectar(letra)
