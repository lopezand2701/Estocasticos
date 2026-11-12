"""
Sphere3D.py
===========
Lienzo 3D: una esfera de radio 1 sobre la que se dibuja con el mouse.
Todo el 3D está hecho a mano con NumPy + QPainter (sin OpenGL):

Orientación      R (matriz de rotación 3×3): lleva un punto del marco de la esfera
                 al marco de la cámara.  p_cam = R · p_esfera
Giro 360°        tipo "trackball": arrastrar (dx, dy) gira la esfera alrededor del
                 eje (dy, dx, 0) un ángulo proporcional a la distancia arrastrada.
                 R ← Rot(eje, ángulo) · R   (sin límites: puede dar vueltas completas)
Proyección       cámara en (0, 0, D) mirando al origen (perspectiva):
                 x_pantalla = cx + f · x / (D − z),   y_pantalla = cy − f · y / (D − z)
Visibilidad      un punto de la superficie se ve si  z_cam > 1/D
Raycast          el rayo desde la cámara por el píxel del mouse, o + t·d, toca la
                 esfera donde |o + t·d|² = 1  (ecuación cuadrática en t)

Los trazos se guardan en el marco de la ESFERA (giran con ella). Para
reconocer, se pasan al "marco de escritura": el de la cámara en el momento
en que se empezó la letra, para que se lea derecha tal como se ve.

Controles: clic izquierdo = dibujar · clic derecho (o Shift + clic) = girar ·
rueda = zoom · flechas = girar 15°.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QSizePolicy, QWidget

from src.features import FeatureExtractor3D as F3
from src.gui import styles

D = 3.2                 # distancia de la cámara al centro (radio de la esfera = 1)


def rotacion(eje: np.ndarray, angulo: float) -> np.ndarray:
    """Matriz de rotación de Rodrigues alrededor de un eje unitario."""
    eje = np.asarray(eje, dtype=float)
    eje = eje / max(np.linalg.norm(eje), 1e-12)
    x, y, z = eje
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angulo) * K + (1 - np.cos(angulo)) * (K @ K)


def _malla() -> list[np.ndarray]:
    """Meridianos y paralelos cada 30° (en el marco de la esfera)."""
    lineas = []
    t = np.linspace(0, 2 * np.pi, 97)
    for lon in np.radians(np.arange(0, 360, 30)):
        lineas.append(F3.desde_esfericas(t / 2, np.full_like(t, lon)))        # de polo a polo
    for lat in np.radians([-60, -30, 0, 30, 60]):
        lineas.append(F3.desde_esfericas(np.full_like(t, np.pi / 2 - lat), t))
    return lineas


class Sphere3D(QWidget):
    trazos_cambiaron = Signal()
    vista_cambio = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.R = np.eye(3)                         # orientación de la esfera
        self.zoom = 1.0
        self.trazos: list[list[np.ndarray]] = []   # en el marco de la ESFERA
        self.marco: np.ndarray | None = None       # R en el momento de empezar la letra
        self.proyeccion: list = []                 # (p0, p1, abajo) en el marco de la esfera
        self.cursor_angulos: tuple[float, float] | None = None
        self._dibujando = False
        self._girando = False
        self._ultimo_mouse = None
        self._malla = _malla()
        self.auto = QTimer(self)
        self.auto.timeout.connect(lambda: self.girar(np.array([0.0, 1.0, 0.0]), np.radians(1.2)))
        self.setMinimumSize(380, 380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.CrossCursor)

    # ---------------- geometría de la cámara ----------------
    def _radio_px(self) -> float:
        return 0.40 * min(self.width(), self.height()) * self.zoom

    def _f(self) -> float:
        return self._radio_px() * np.sqrt(D * D - 1)

    def _centro(self) -> tuple[float, float]:
        return self.width() / 2, self.height() / 2

    def _a_pantalla(self, p_cam: np.ndarray) -> np.ndarray:
        """Proyección en perspectiva de puntos en el marco de la cámara (N×3 → N×2)."""
        cx, cy = self._centro()
        f = self._f()
        den = D - p_cam[:, 2]
        return np.stack([cx + f * p_cam[:, 0] / den, cy - f * p_cam[:, 1] / den], axis=1)

    def _raycast(self, x: float, y: float) -> np.ndarray | None:
        """Píxel → punto de la esfera (marco de la cámara), o None si el rayo no la toca."""
        cx, cy = self._centro()
        f = self._f()
        d = np.array([(x - cx) / f, -(y - cy) / f, -1.0])
        d /= np.linalg.norm(d)
        o = np.array([0.0, 0.0, D])
        b = float(o @ d)
        disc = b * b - (D * D - 1)
        if disc < 0:
            return None
        t = -b - np.sqrt(disc)                    # la intersección más cercana a la cámara
        return o + t * d

    # ---------------- API pública ----------------
    def girar(self, eje: np.ndarray, angulo: float) -> None:
        self.R = rotacion(eje, angulo) @ self.R
        self.update()
        self.vista_cambio.emit()

    def reorientar(self) -> None:
        self.R = np.eye(3)
        self.zoom = 1.0
        self.update()

    def limpiar(self) -> None:
        self.trazos.clear()
        self.marco = None
        self.proyeccion = []
        self.update()
        self.trazos_cambiaron.emit()

    def deshacer(self) -> None:
        if self.trazos:
            self.trazos.pop()
            if not self.trazos:
                self.marco = None
            self.update()
            self.trazos_cambiaron.emit()

    @property
    def dibujando(self) -> bool:
        return self._dibujando

    def trazos_escritura(self) -> list[np.ndarray]:
        """Trazos en el marco de escritura (para reconocer): p = R_marco · p_esfera."""
        if self.marco is None:
            return []
        return [np.asarray(t) @ self.marco.T for t in self.trazos if len(t)]

    def ultimo_punto_escritura(self) -> np.ndarray | None:
        if self.marco is None or not self.trazos or not self.trazos[-1]:
            return None
        return self.marco @ self.trazos[-1][-1]

    def set_proyeccion_escritura(self, segmentos) -> None:
        """Recibe la proyección en el marco de escritura y la guarda en el de la esfera."""
        if self.marco is None:
            self.proyeccion = []
        else:
            Mt = self.marco.T
            self.proyeccion = [(Mt @ p, Mt @ q, abajo) for p, q, abajo in segmentos]
        self.update()

    # ---------------- eventos ----------------
    def mousePressEvent(self, e) -> None:
        pos = e.position()
        rotar = e.button() == Qt.RightButton or (e.button() == Qt.LeftButton and e.modifiers() & Qt.ShiftModifier)
        if rotar:
            self._girando = True
            self._ultimo_mouse = (pos.x(), pos.y())
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() == Qt.LeftButton:
            hit = self._raycast(pos.x(), pos.y())
            if hit is None:
                return
            if self.marco is None:                              # empieza una letra nueva
                self.marco = self.R.copy()
            self._dibujando = True
            self.trazos.append([self.R.T @ hit])
            self.update()
            self.trazos_cambiaron.emit()

    def mouseMoveEvent(self, e) -> None:
        pos = e.position()
        hit = self._raycast(pos.x(), pos.y())
        if hit is not None:
            th, ph = F3.a_esfericas(self.R.T @ hit)
            self.cursor_angulos = (float(np.degrees(th[0])), float(np.degrees(ph[0])))
        else:
            self.cursor_angulos = None
        if self._girando:
            x0, y0 = self._ultimo_mouse
            dx, dy = pos.x() - x0, pos.y() - y0
            self._ultimo_mouse = (pos.x(), pos.y())
            dist = np.hypot(dx, dy)
            if dist > 0:
                self.girar(np.array([dy, dx, 0.0]), dist / self._radio_px())
            return
        if self._dibujando and hit is not None:
            p = self.R.T @ hit
            if F3._angulo(self.trazos[-1][-1], p) >= np.radians(0.15):
                self.trazos[-1].append(p)
                self.update()
                self.trazos_cambiaron.emit()
        else:
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        if self._girando:
            self._girando = False
            self.setCursor(Qt.CrossCursor)
        if self._dibujando:
            self._dibujando = False
            self.trazos_cambiaron.emit()           # al levantar el lápiz se evalúa como letra completa

    def wheelEvent(self, e) -> None:
        self.zoom = float(np.clip(self.zoom * 1.1 ** (e.angleDelta().y() / 120), 0.5, 2.4))
        self.update()

    def keyPressEvent(self, e) -> None:
        paso = np.radians(15)
        ejes = {Qt.Key_Left: ([0, 1, 0], -paso), Qt.Key_Right: ([0, 1, 0], paso),
                Qt.Key_Up: ([1, 0, 0], -paso), Qt.Key_Down: ([1, 0, 0], paso)}
        if e.key() in ejes:
            eje, ang = ejes[e.key()]
            self.girar(np.array(eje, dtype=float), ang)
        else:
            super().keyPressEvent(e)

    # ---------------- dibujo ----------------
    def _polilinea(self, g: QPainter, pts_esfera: np.ndarray, pen_frente: QPen, pen_atras: QPen | None) -> None:
        """Dibuja una línea sobre la esfera: el lado visible con pen_frente, el oculto con pen_atras."""
        if len(pts_esfera) < 2:
            return
        cam = pts_esfera @ self.R.T
        scr = self._a_pantalla(cam)
        visible = cam[:, 2] > 1.0 / D
        for i in range(len(cam) - 1):
            frente = visible[i] and visible[i + 1]
            if not frente and pen_atras is None:
                continue
            g.setPen(pen_frente if frente else pen_atras)
            g.drawLine(QPointF(*scr[i]), QPointF(*scr[i + 1]))

    def paintEvent(self, _) -> None:
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        cx, cy = self._centro()
        r = self._radio_px()

        # sombra suave sobre la hoja
        sombra = QRadialGradient(QPointF(cx, cy + r * 0.92), r * 0.9)
        sombra.setColorAt(0.0, QColor(28, 47, 77, 40))
        sombra.setColorAt(1.0, QColor(28, 47, 77, 0))
        g.setPen(Qt.NoPen)
        g.setBrush(sombra)
        g.drawEllipse(QPointF(cx, cy + r * 0.92), r * 0.9, r * 0.22)

        # globo de papel con luz arriba a la izquierda
        grad = QRadialGradient(QPointF(cx - r * 0.38, cy - r * 0.42), r * 1.55)
        grad.setColorAt(0.0, QColor("#FFFFFF"))
        grad.setColorAt(0.55, QColor("#EEF2F7"))
        grad.setColorAt(1.0, QColor("#C9D5E2"))
        g.setPen(QPen(QColor(styles.RENGLON_FUERTE), 1.2))
        g.setBrush(grad)
        g.drawEllipse(QPointF(cx, cy), r, r)

        # meridianos y paralelos cada 30° (se ven girar); ecuador y meridiano 0 más marcados
        for i, linea in enumerate(self._malla):
            destacada = i == 0 or i == len(self._malla) - 3
            color = QColor(styles.TINTA)
            color.setAlpha(110 if destacada else 38)
            self._polilinea(g, linea, QPen(color, 1.3 if destacada else 1.0), None)

        grosor = max(3.0, r / 65)

        # trazos del usuario en tinta (los de atrás, tenues y punteados)
        atras = QColor(styles.TINTA)
        atras.setAlpha(55)
        pen_frente = QPen(QColor(styles.TINTA), grosor, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        pen_atras = QPen(atras, grosor * 0.6, Qt.DotLine, Qt.RoundCap)
        for trazo in self.trazos:
            self._polilinea(g, np.asarray(trazo), pen_frente, pen_atras)

        # proyección en lápiz rojo punteado, sobre la superficie
        pen_proy = QPen(QColor(styles.LAPIZ_ROJO), grosor * 0.75, Qt.CustomDashLine, Qt.RoundCap)
        pen_proy.setDashPattern([0.1, 2.2])
        for p0, p1, abajo in self.proyeccion:
            if abajo:
                self._polilinea(g, np.array([p0, p1]), pen_proy, None)

        # inicio de cada trazo con su número (si se ve)
        f = g.font()
        f.setPixelSize(int(grosor * 2.2))
        f.setBold(True)
        g.setFont(f)
        rr = grosor * 1.6
        for n, trazo in enumerate(self.trazos, start=1):
            cam = self.R @ trazo[0]
            if cam[2] > 1.0 / D:
                x, y = self._a_pantalla(cam[None, :])[0]
                g.setPen(Qt.NoPen)
                g.setBrush(QColor(styles.TINTA))
                g.drawEllipse(QPointF(x, y), rr, rr)
                g.setPen(QColor(styles.HOJA))
                g.drawText(QRectF(x - rr, y - rr, 2 * rr, 2 * rr), Qt.AlignCenter, str(n))

        # ángulos del cursor (conversión a coordenadas esféricas en vivo)
        if self.cursor_angulos:
            th, ph = self.cursor_angulos
            f.setPixelSize(12)
            f.setBold(False)
            g.setFont(f)
            g.setPen(QColor(styles.GRAFITO))
            g.drawText(QRectF(0, self.height() - 24, self.width() - 8, 20), Qt.AlignRight | Qt.AlignVCenter,
                       f"θ {th:5.1f}°   φ {ph:6.1f}°")
        g.end()
