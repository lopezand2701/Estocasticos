"""
GuideView.py
============
Estandarización del alfabeto: cómo se escribe correctamente cada letra.

    ┌ barra: ← Menú · Estandarización del alfabeto ──────────────────────────┐
    │ ┌ alfabeto ┐ ┌ trazado animado ──────────────┐ ┌ práctica ───────────┐ │
    │ │ a b c d  │ │ letra en gris; cada trazo se  │ │ escribe encima del  │ │
    │ │ e f g h  │ │ dibuja en orden con su número │ │ calco y el programa │ │
    │ │ ...      │ │ Trazo 1: empieza… va hacia…   │ │ revisa letra, número│ │
    │ └──────────┘ │ Repetir · ← Anterior · Sig. → │ │ de trazos e inicios │ │
    │              └───────────────────────────────┘ └─────────────────────┘ │
    └────────────────────────────────────────────────────────────────────────┘

Los trazos modelo vienen de la guía de dibujo del proyecto Glifo
(assets/guia_letras.json, crédito: Daniel Santos): el mismo estándar con el
que el curso recolectó los datos de entrenamiento.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from src.features import FeatureExtractor as FE
from src.gui import glifos, styles
from src.gui.Canvas2D import Canvas2D, GUIAS
from src.hmm.Recognizer import Recognizer

LETRAS = list("abcdefghijklmnñopqrstuvwxyz")
VELOCIDAD = 520.0        # unidades del lienzo por segundo
PAUSA = 0.45             # segundos entre un trazo y el siguiente
TOL_INICIO = 110.0       # qué tan cerca debe empezar cada trazo de su punto en la guía
TOL_ANGULO = 60.0        # diferencia máxima de dirección inicial (grados)

_DIRECCIONES = ["hacia la derecha", "hacia arriba a la derecha", "hacia arriba", "hacia arriba a la izquierda",
                "hacia la izquierda", "hacia abajo a la izquierda", "hacia abajo", "hacia abajo a la derecha"]


# ---------------------------------------------------------------------- #
# Geometría de los trazos modelo
# ---------------------------------------------------------------------- #
def _longitudes(t: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(t, axis=0), axis=1))])


def direccion_inicial(t) -> float | None:
    """Ángulo (grados, 0 = derecha, 90 = arriba) de los primeros ~60 unidades del trazo."""
    t = np.asarray(t, dtype=float)
    L = _longitudes(t)
    if L[-1] < 10:
        return None
    j = int(np.searchsorted(L, min(60.0, L[-1])))
    d = t[min(j, len(t) - 1)] - t[0]
    return float(np.degrees(np.arctan2(-d[1], d[0])) % 360)


def describir_trazo(i: int, t) -> str:
    """Frase en español que dice dónde empieza el trazo y hacia dónde va."""
    t = np.asarray(t, dtype=float)
    x, y = t[0]
    if y < 300:
        zona = "en la línea ascendente"
    elif y < 560:
        zona = "en la altura x"
    elif y < 800:
        zona = "cerca de la línea base"
    else:
        zona = "en la zona descendente"
    lado = "a la izquierda" if x < 470 else ("a la derecha" if x > 560 else "en el centro")
    ang = direccion_inicial(t)
    if ang is None:
        return f"Trazo {i}: un punto {zona}, {lado}."
    hacia = _DIRECCIONES[int(round(ang / 45)) % 8]
    # ¿vuelve por el mismo camino sin levantar el lápiz? (como la n, la h o la m)
    d = np.diff(t, axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-6]
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    vuelve = bool(np.any(np.sum(u[1:] * u[:-1], axis=1) < -0.9))
    extra = " y luego vuelve por el mismo camino sin levantar el lápiz" if vuelve else ""
    return f"Trazo {i}: empieza {zona}, {lado}, y va {hacia}{extra}."


# ---------------------------------------------------------------------- #
# Animación del trazado
# ---------------------------------------------------------------------- #
class TrazadoAnimado(QWidget):
    """Muestra la letra en gris y dibuja encima cada trazo, en orden, con un lápiz rojo."""
    termino = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.trazos: list[np.ndarray] = []
        self.tiempo = 0.0
        self._plan: list[tuple[float, float]] = []     # (inicio, duración) de cada trazo
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._avanzar)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_letra(self, letra: str) -> None:
        self.trazos = [np.asarray(t, dtype=float) for t in glifos.trazos_guia().get(letra, [])]
        self._plan, inicio = [], 0.4
        for t in self.trazos:
            dur = max(_longitudes(t)[-1] / VELOCIDAD, 0.25)
            self._plan.append((inicio, dur))
            inicio += dur + PAUSA
        self.repetir()

    def repetir(self) -> None:
        self.tiempo = 0.0
        self._timer.start(16)
        self.update()

    def _avanzar(self) -> None:
        self.tiempo += 0.016
        if self._plan and self.tiempo > self._plan[-1][0] + self._plan[-1][1] + 0.3:
            self._timer.stop()
            self.termino.emit()
        self.update()

    # espacio de la guía → pantalla (caja 160..940 en y, centrada)
    def _transf(self):
        lado = min(self.width(), self.height() * 0.95)
        esc = lado / 800
        ox = self.width() / 2 - 510 * esc
        oy = self.height() / 2 - 560 * esc
        return lambda x, y: QPointF(ox + x * esc, oy + y * esc), esc

    def paintEvent(self, _) -> None:
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        a_px, esc = self._transf()
        # renglones
        for nombre, y in GUIAS.items():
            base = nombre == "línea base"
            g.setPen(QPen(QColor(styles.RENGLON_FUERTE if base else styles.RENGLON), 1.6 if base else 1,
                          Qt.SolidLine if base else Qt.DashLine))
            g.drawLine(a_px(110, y), a_px(910, y))
            f = g.font()
            f.setPixelSize(11)
            g.setFont(f)
            g.setPen(QColor(styles.GRAFITO))
            g.drawText(QRectF(a_px(110, y - 34), a_px(500, y - 4)), Qt.AlignLeft | Qt.AlignBottom, nombre)
        grosor = max(5.0, 34 * esc)

        def camino(t: np.ndarray, hasta: float | None = None) -> QPainterPath:
            L = _longitudes(t)
            c = QPainterPath(a_px(*t[0]))
            if len(t) == 1 or L[-1] < 1:
                c.lineTo(a_px(t[0][0] + 1, t[0][1]))
                return c
            for k in range(1, len(t)):
                if hasta is not None and L[k] > hasta:
                    u = (hasta - L[k - 1]) / max(L[k] - L[k - 1], 1e-9)
                    p = t[k - 1] + (t[k] - t[k - 1]) * u
                    c.lineTo(a_px(*p))
                    break
                c.lineTo(a_px(*t[k]))
            return c

        # la letra completa en gris claro
        gris = QColor(styles.RENGLON)
        g.setPen(QPen(gris, grosor, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        g.setBrush(Qt.NoBrush)
        for t in self.trazos:
            g.drawPath(camino(t))

        # trazos animados en tinta, uno tras otro
        lapiz = None
        for i, (t, (ini, dur)) in enumerate(zip(self.trazos, self._plan)):
            if self.tiempo < ini:
                continue
            frac = min((self.tiempo - ini) / dur, 1.0)
            L = _longitudes(t)
            g.setPen(QPen(QColor(styles.TINTA), grosor * 0.62, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            g.drawPath(camino(t, L[-1] * frac))
            if frac < 1.0:
                j = int(np.searchsorted(L, L[-1] * frac))
                lapiz = a_px(*t[min(j, len(t) - 1)])
        # números de inicio de los trazos ya empezados
        f = g.font()
        f.setPixelSize(int(grosor * 0.62))
        f.setBold(True)
        g.setFont(f)
        r = grosor * 0.62
        for i, (t, (ini, _)) in enumerate(zip(self.trazos, self._plan), start=1):
            if self.tiempo < ini:
                continue
            c = a_px(*t[0])
            g.setPen(Qt.NoPen)
            g.setBrush(QColor(styles.LAPIZ_ROJO))
            g.drawEllipse(c, r, r)
            g.setPen(QColor(styles.HOJA))
            g.drawText(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), Qt.AlignCenter, str(i))
        # la punta del lápiz mientras dibuja
        if lapiz is not None:
            g.setPen(QPen(QColor(styles.HOJA), 2))
            g.setBrush(QColor(styles.LAPIZ_ROJO))
            g.drawEllipse(lapiz, grosor * 0.38, grosor * 0.38)
        g.end()


# ---------------------------------------------------------------------- #
# Botón de letra en la cuadrícula
# ---------------------------------------------------------------------- #
class BotonLetra(QPushButton):
    def __init__(self, letra: str, parent=None) -> None:
        super().__init__(parent)
        self.letra = letra
        self.setCheckable(True)
        self.setFixedSize(52, 62)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"Ver cómo se traza la «{letra}»")
        self.setStyleSheet("QPushButton { background: transparent; border: none; padding: 0; }")

    def paintEvent(self, _) -> None:
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        r = QRectF(1, 1, self.width() - 2, self.height() - 2)
        elegida = self.isChecked()
        g.setPen(QPen(QColor(styles.TINTA if (elegida or self.hasFocus()) else styles.RENGLON), 1.2))
        g.setBrush(QColor(styles.RESALTADOR if elegida else (styles.HOJA if self.underMouse() else styles.PAPEL)))
        g.drawRoundedRect(r, 10, 10)
        glifos.dibujar_letra(g, self.letra, r.adjusted(8, 6, -8, -6), styles.TINTA, grosor=2.4)
        g.end()


# ---------------------------------------------------------------------- #
# La vista completa
# ---------------------------------------------------------------------- #
class GuideView(QWidget):
    volver = Signal()

    def __init__(self, recognizer: Recognizer, parent=None) -> None:
        super().__init__(parent)
        self.rec = recognizer
        self.letra = "a"

        # ---------------- barra ----------------
        b_menu = QPushButton("← Menú")
        b_menu.clicked.connect(self.volver.emit)
        titulo = QLabel("Estandarización del alfabeto")
        titulo.setObjectName("marca")
        barra = QHBoxLayout()
        barra.addWidget(b_menu)
        barra.addSpacing(14)
        barra.addWidget(titulo)
        barra.addStretch()

        # ---------------- alfabeto ----------------
        cab_alf = QLabel("Elige una letra")
        cab_alf.setObjectName("seccion")
        rejilla = QGridLayout()
        rejilla.setSpacing(6)
        self.botones: dict[str, BotonLetra] = {}
        for i, c in enumerate(LETRAS):
            b = BotonLetra(c)
            b.clicked.connect(lambda _=False, c=c: self.set_letra(c))
            rejilla.addWidget(b, i // 4, i % 4)
            self.botones[c] = b
        alfabeto = QFrame()
        alfabeto.setObjectName("hoja")
        alfabeto.setFixedWidth(270)
        la = QVBoxLayout(alfabeto)
        la.setContentsMargins(16, 16, 16, 16)
        la.addWidget(cab_alf)
        la.addLayout(rejilla)
        la.addStretch()

        # ---------------- trazado animado ----------------
        self.titulo_letra = QLabel("")
        self.titulo_letra.setObjectName("seccion")
        self.resumen = QLabel("")
        self.resumen.setObjectName("nota")
        self.resumen.setWordWrap(True)
        self.animacion = TrazadoAnimado()
        self.pasos = QLabel("")
        self.pasos.setWordWrap(True)
        self.pasos.setStyleSheet(f"color: {styles.TINTA}; font-size: 13px;")
        b_rep = QPushButton("↻ Repetir")
        b_ant = QPushButton("← Anterior")
        b_sig = QPushButton("Siguiente →")
        b_rep.clicked.connect(self.animacion.repetir)
        b_ant.clicked.connect(lambda: self._mover(-1))
        b_sig.clicked.connect(lambda: self._mover(1))
        botones = QHBoxLayout()
        botones.addWidget(b_rep)
        botones.addStretch()
        botones.addWidget(b_ant)
        botones.addWidget(b_sig)
        trazado = QFrame()
        trazado.setObjectName("hoja")
        lt = QVBoxLayout(trazado)
        lt.setContentsMargins(20, 16, 20, 14)
        lt.addWidget(self.titulo_letra)
        lt.addWidget(self.resumen)
        lt.addWidget(self.animacion, 1)
        lt.addWidget(self.pasos)
        lt.addLayout(botones)

        # ---------------- práctica ----------------
        cab_pr = QLabel("Practica encima del calco")
        cab_pr.setObjectName("seccion")
        self.lienzo = Canvas2D()
        self.lienzo.setMinimumSize(280, 280)
        self.lienzo.trazos_cambiaron.connect(self._revisar)
        self.revision = QLabel("Escribe la letra siguiendo el orden de los números. "
                               "Al soltar el mouse se revisa tu trazo.")
        self.revision.setObjectName("nota")
        self.revision.setWordWrap(True)
        self.revision.setTextFormat(Qt.RichText)
        b_limpiar = QPushButton("Limpiar")
        b_limpiar.clicked.connect(self.lienzo.limpiar)
        self.b_calco = QPushButton("Calco")
        self.b_calco.setCheckable(True)
        self.b_calco.setChecked(True)
        self.b_calco.toggled.connect(self._actualizar_calco)
        bp = QHBoxLayout()
        bp.addWidget(self.b_calco)
        bp.addStretch()
        bp.addWidget(b_limpiar)
        practica = QFrame()
        practica.setObjectName("hoja")
        practica.setFixedWidth(400)
        lp = QVBoxLayout(practica)
        lp.setContentsMargins(16, 16, 16, 14)
        lp.addWidget(cab_pr)
        lp.addWidget(self.lienzo, 1)
        lp.addWidget(self.revision)
        lp.addLayout(bp)

        cuerpo = QHBoxLayout()
        cuerpo.setSpacing(16)
        cuerpo.addWidget(alfabeto)
        cuerpo.addWidget(trazado, 1)
        cuerpo.addWidget(practica)

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(22, 16, 22, 22)
        raiz.setSpacing(14)
        raiz.addLayout(barra)
        raiz.addLayout(cuerpo, 1)

        self.set_letra("a")

    # ---------------- navegación ----------------
    def set_letra(self, letra: str) -> None:
        self.letra = letra
        for c, b in self.botones.items():
            b.setChecked(c == letra)
        trazos = glifos.trazos_guia().get(letra, [])
        n = len(trazos)
        self.titulo_letra.setText(f"Así se traza la «{letra}»")
        if n == 1:
            self.resumen.setText("Un solo trazo: empieza en el 1 y no levantes el lápiz hasta terminar.")
        else:
            self.resumen.setText(f"{n} trazos: empieza en el 1 y levanta el lápiz entre uno y otro, en orden.")
        self.pasos.setText("\n".join(describir_trazo(i, t) for i, t in enumerate(trazos, start=1)))
        self.animacion.set_letra(letra)
        self.lienzo.limpiar()
        self._actualizar_calco(self.b_calco.isChecked())

    def _mover(self, paso: int) -> None:
        i = (LETRAS.index(self.letra) + paso) % len(LETRAS)
        self.set_letra(LETRAS[i])

    def _actualizar_calco(self, visible: bool) -> None:
        self.lienzo.plantilla = glifos.trazos_guia().get(self.letra) if visible else None
        self.lienzo.update()

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key_Left, Qt.Key_Right):
            self._mover(-1 if e.key() == Qt.Key_Left else 1)
        elif e.key() == Qt.Key_Space:
            self.animacion.repetir()
        else:
            super().keyPressEvent(e)

    def reiniciar(self) -> None:
        self.lienzo.limpiar()
        self.rec.reset()

    # ---------------- revisión de la práctica ----------------
    def _revisar(self) -> None:
        trazos = [t for t in self.lienzo.trazos if len(t)]
        if not trazos:
            self.revision.setText("Escribe la letra siguiendo el orden de los números. "
                                  "Al soltar el mouse se revisa tu trazo.")
            return
        if self.lienzo.dibujando:
            return
        modelo = glifos.trazos_guia().get(self.letra, [])
        bien = f"<span style='color:#2E7D4F'>✔</span>"
        mal = f"<span style='color:{styles.LAPIZ_ROJO}'>✘</span>"
        lineas = []

        # 1. ¿el HMM la reconoce como la letra que se está practicando?
        simbolos = FE.strokes_to_symbols(trazos, step=self.rec.step)
        self.rec.reset()
        self.rec.update(simbolos)
        ranking = self.rec.posteriors(completo=True)
        mejor, p = ranking[0]
        p_letra = dict(ranking).get(self.letra, 0.0)
        if mejor == self.letra:
            lineas.append(f"{bien} El modelo la reconoce como «{mejor}» ({p * 100:.0f} %).")
        else:
            lineas.append(f"{mal} El modelo la lee como «{mejor}» ({p * 100:.0f} %); "
                          f"«{self.letra}» tiene {p_letra * 100:.0f} %.")

        # 2. número de trazos
        if len(trazos) == len(modelo):
            lineas.append(f"{bien} Usaste {len(trazos)} trazo{'s' if len(trazos) > 1 else ''}, como la guía.")
        else:
            lineas.append(f"{mal} Usaste {len(trazos)} trazo{'s' if len(trazos) > 1 else ''}; "
                          f"la guía usa {len(modelo)}.")

        # 3. cada trazo: dónde empieza y hacia dónde arranca
        for i, (t_user, t_guia) in enumerate(zip(trazos, modelo), start=1):
            d = float(np.hypot(t_user[0][0] - t_guia[0][0], t_user[0][1] - t_guia[0][1]))
            ang_u, ang_g = direccion_inicial(t_user), direccion_inicial(t_guia)
            dif = None if ang_u is None or ang_g is None else abs((ang_u - ang_g + 180) % 360 - 180)
            if d > TOL_INICIO:
                lineas.append(f"{mal} Trazo {i}: empieza lejos del punto {i} de la guía.")
            elif dif is not None and dif > TOL_ANGULO:
                lineas.append(f"{mal} Trazo {i}: empieza bien, pero arranca en otra dirección "
                              f"({_DIRECCIONES[int(round(ang_g / 45)) % 8]}).")
            else:
                lineas.append(f"{bien} Trazo {i}: buen inicio y buena dirección.")
        self.revision.setText("<br>".join(lineas))
