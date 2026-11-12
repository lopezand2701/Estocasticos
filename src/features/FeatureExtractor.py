"""
FeatureExtractor.py
===================
Convierte un dibujo (lista de trazos con puntos x, y) en una secuencia de
símbolos discretos O = O_1 ... O_T para el HMM.

Pasos:
    1. Remuestreo por longitud de arco: puntos a distancia fija STEP.
       Así no importa si el usuario dibuja rápido o lento.
    2. Código de cadena: cada desplazamiento (Δx, Δy) se cuantiza en
       K = 16 direcciones (sectores de 22.5°).
    3. Levantar el lápiz: el salto entre el final de un trazo y el inicio
       del siguiente también se cuantiza, pero con símbolos 16..31
       ("movimiento con el lápiz arriba en la dirección k").

Espacio de coordenadas: 0..1000 en x e y, con la y creciendo hacia abajo
(el mismo del dataset Glifo y del lienzo de la aplicación).
"""
from __future__ import annotations

import numpy as np

ESPACIO = 1000.0     # tamaño del lienzo lógico
STEP = 25.0          # distancia entre puntos remuestreados (en el espacio normalizado)
TAM_NORMAL = (200.0, 550.0)  # rango de tamaños "normales" de una letra (lado mayor, como en Glifo)
TAM_MIN = 40.0       # evita escalas enormes al comienzo del trazo (cuando aún mide casi nada)
K = 16               # número de direcciones
PEN_UP = K           # los símbolos K..2K-1 son movimientos con el lápiz arriba
M = 2 * K            # tamaño total del alfabeto (32)

# Vectores unitarios de cada dirección en coordenadas de pantalla (y hacia abajo).
# La dirección 0 es → (derecha), 4 es ↑, 8 es ←, 12 es ↓.
_ANGULOS = np.arange(K) * (2 * np.pi / K)
UNIT = np.stack([np.cos(_ANGULOS), -np.sin(_ANGULOS)], axis=1)

FLECHAS = ["→", "↗", "↗", "↗", "↑", "↖", "↖", "↖", "←", "↙", "↙", "↙", "↓", "↘", "↘", "↘"]


def quantize(dx: float, dy: float) -> int:
    """Dirección (dx, dy) en pantalla → sector 0..K-1. Se invierte dy para usar ángulos matemáticos."""
    ang = np.arctan2(-dy, dx) % (2 * np.pi)
    return int(np.round(ang / (2 * np.pi / K))) % K


def resample(points: np.ndarray, step: float = STEP) -> np.ndarray:
    """
    Recorre la polilínea y devuelve puntos separados exactamente `step`
    a lo largo del camino (el primero es el punto inicial).
    Al agregar puntos al final, los ya remuestreados no cambian.
    """
    pts = np.asarray(points, dtype=float)[:, :2]
    if len(pts) == 0:
        return pts
    salida = [pts[0]]
    resto = step                       # distancia que falta para el próximo punto
    previo = pts[0]
    for actual in pts[1:]:
        seg = actual - previo
        largo = float(np.hypot(*seg))
        while largo >= resto:
            nuevo = previo + seg * (resto / largo)
            salida.append(nuevo)
            seg = actual - nuevo
            largo = float(np.hypot(*seg))
            previo = nuevo
            resto = step
        resto -= largo
        previo = actual
    return np.array(salida)


def _direcciones(pts: np.ndarray, desplazar: int = 0) -> list[int]:
    d = np.diff(pts, axis=0)
    return [quantize(dx, dy) + desplazar for dx, dy in d]


def escala(strokes) -> float:
    """
    Normalización de tamaño "con límites":
        lado = max(ancho, alto)
        s    = clip(lado, 200, 550) / max(lado, TAM_MIN)
    - Si la letra tiene un tamaño normal (200 a 550), s = 1: no se toca, y se
      conserva la diferencia de altura entre letras como "h" y "n".
    - Si es muy pequeña se agranda hasta 200; si es muy grande se achica hasta 550.
    Así se puede dibujar en cualquier parte y de cualquier tamaño. (La posición
    no importa: las direcciones Δx, Δy no cambian al trasladar el dibujo.)
    """
    pts = [p[:2] for t in strokes for p in t]
    if not pts:
        return 1.0
    pts = np.asarray(pts, dtype=float)
    lado = float((pts.max(axis=0) - pts.min(axis=0)).max())
    lo, hi = TAM_NORMAL
    return float(np.clip(lado, lo, hi)) / max(lado, TAM_MIN)


def strokes_to_symbols(strokes, step: float = STEP, normalizar: bool = True) -> list[int]:
    """
    strokes: lista de trazos; cada trazo es una lista de puntos [x, y] o [x, y, t].
    Devuelve la secuencia de símbolos (enteros 0..M-1).
    normalizar=True escala el dibujo a TAM_REF antes de remuestrear.
    """
    s = escala(strokes) if normalizar else 1.0
    simbolos: list[int] = []
    ultimo = None                                # último punto remuestreado del trazo anterior
    for trazo in strokes:
        if len(trazo) == 0:
            continue
        pts = np.asarray(trazo, dtype=float)[:, :2] * s
        if ultimo is not None:                   # salto con el lápiz arriba
            salto = resample(np.array([ultimo, pts[0]]), step)
            simbolos += _direcciones(salto, desplazar=PEN_UP)
        rs = resample(pts, step)
        simbolos += _direcciones(rs)
        ultimo = rs[-1]
    return simbolos


def symbol_vector(k: int, step: float = STEP) -> np.ndarray:
    """Desplazamiento (dx, dy) que representa un símbolo."""
    return UNIT[k % K] * step


def describe(symbols) -> str:
    """Texto legible de una secuencia, p. ej. '↓ ↓ → ⇡↑' (⇡ = lápiz arriba)."""
    return " ".join(("⇡" if s >= PEN_UP else "") + FLECHAS[s % K] for s in symbols)
