"""
FeatureExtractor3D.py
=====================
Convierte un dibujo hecho sobre la SUPERFICIE de una esfera en la misma
secuencia de símbolos O_1 ... O_T que usa el HMM (16 direcciones + 16 de
lápiz arriba = 32 símbolos), usando coordenadas esféricas.

Marco de referencia ("marco de escritura")
------------------------------------------
Los puntos se expresan en el marco de la cámara en el momento en que se
empezó la letra:   x → derecha,  y → arriba,  z → hacia quien mira.
Así la letra se lee "derecha" tal como el usuario la ve, aunque antes haya
girado la esfera (la bola puede rotar 360° en cualquier dirección).

Coordenadas esféricas (polo norte = arriba de la pantalla)
    θ = arccos(y)          ángulo polar (0 en el polo norte, π en el sur)
    φ = atan2(x, z)        longitud (0 en el centro de la vista, crece a la derecha)

Dirección de un paso entre dos puntos de la superficie
    Δ_este  = Δφ · sin θ   (desplazamiento horizontal real sobre la esfera)
    Δ_sur   = Δθ           (θ crece hacia abajo, igual que la y de la pantalla)
    símbolo = cuantizar(Δ_este, Δ_sur)  con las mismas 16 direcciones del 2D
"""
from __future__ import annotations

import numpy as np

from src.features import FeatureExtractor as FE

# 1000 unidades del lienzo 2D equivalen a 100° de esfera → 1 unidad = 0.1°
GRADOS_POR_UNIDAD = 0.1
STEP = np.radians(FE.STEP * GRADOS_POR_UNIDAD)                         # 2.5° por paso
TAM_NORMAL = tuple(np.radians(t * GRADOS_POR_UNIDAD) for t in FE.TAM_NORMAL)  # 20° a 55°
TAM_MIN = np.radians(FE.TAM_MIN * GRADOS_POR_UNIDAD)                   # 4°


# ---------------------------------------------------------------------- #
# Conversión de coordenadas
# ---------------------------------------------------------------------- #
def a_esfericas(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x, y, z) unitario → (θ, φ)."""
    p = np.atleast_2d(p)
    theta = np.arccos(np.clip(p[:, 1], -1.0, 1.0))
    phi = np.arctan2(p[:, 0], p[:, 2])
    return theta, phi


def desde_esfericas(theta, phi) -> np.ndarray:
    """(θ, φ) → (x, y, z) unitario."""
    st = np.sin(theta)
    return np.stack([st * np.sin(phi), np.cos(theta), st * np.cos(phi)], axis=-1)


def base_tangente(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectores unitarios 'este' (φ creciente) y 'norte' (θ decreciente) en el punto p."""
    theta, phi = a_esfericas(p)
    theta, phi = float(theta[0]), float(phi[0])
    este = np.array([np.cos(phi), 0.0, -np.sin(phi)])
    norte = np.array([-np.cos(theta) * np.sin(phi), np.sin(theta), -np.cos(theta) * np.cos(phi)])
    return este, norte


def _angulo(p: np.ndarray, q: np.ndarray) -> float:
    """Distancia geodésica (ángulo central) entre dos puntos unitarios."""
    return float(np.arctan2(np.linalg.norm(np.cross(p, q)), np.dot(p, q)))


def slerp(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    """Interpolación sobre el círculo máximo que une p y q."""
    w = _angulo(p, q)
    if w < 1e-9:
        return p.copy()
    return (np.sin((1 - t) * w) * p + np.sin(t * w) * q) / np.sin(w)


def mover(p: np.ndarray, d_este: float, d_sur: float) -> np.ndarray:
    """Avanza sobre la superficie desde p: d_este, d_sur en radianes (mapa exponencial)."""
    este, norte = base_tangente(p)
    v = d_este * este - d_sur * norte
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return p.copy()
    return np.cos(a) * p + np.sin(a) * v / a


# ---------------------------------------------------------------------- #
# Remuestreo y símbolos
# ---------------------------------------------------------------------- #
def remuestrear(puntos: np.ndarray, paso: float) -> np.ndarray:
    """Puntos separados exactamente `paso` radianes a lo largo de la superficie."""
    pts = np.asarray(puntos, dtype=float)
    if len(pts) == 0:
        return pts
    salida = [pts[0]]
    resto = paso
    previo = pts[0]
    for actual in pts[1:]:
        largo = _angulo(previo, actual)
        while largo >= resto:
            nuevo = slerp(previo, actual, resto / largo)
            salida.append(nuevo)
            largo = _angulo(nuevo, actual)
            previo = nuevo
            resto = paso
        resto -= largo
        previo = actual
    return np.array(salida)


def escala(strokes) -> float:
    """Igual que en 2D, pero con el tamaño angular del dibujo (en radianes)."""
    pts = [p for t in strokes for p in t]
    if not pts:
        return 1.0
    theta, phi = a_esfericas(np.asarray(pts, dtype=float))
    phi_c = np.angle(np.mean(np.exp(1j * phi)))                  # longitud media (sin saltos en ±π)
    dphi = np.angle(np.exp(1j * (phi - phi_c)))
    ancho = float((dphi.max() - dphi.min()) * np.sin(np.mean(theta)))
    alto = float(theta.max() - theta.min())
    lado = max(ancho, alto)
    lo, hi = TAM_NORMAL
    return float(np.clip(lado, lo, hi)) / max(lado, TAM_MIN)


def _direcciones(pts: np.ndarray, desplazar: int = 0) -> list[int]:
    if len(pts) < 2:
        return []
    theta, phi = a_esfericas(pts)
    d_theta = np.diff(theta)
    d_phi = np.angle(np.exp(1j * np.diff(phi)))                # Δφ en (−π, π]
    sen = np.sin((theta[1:] + theta[:-1]) / 2)
    return [FE.quantize(dp * s, dt) + desplazar for dp, dt, s in zip(d_phi, d_theta, sen)]


def strokes_to_symbols(strokes, normalizar: bool = True) -> list[int]:
    """
    strokes: lista de trazos; cada trazo es una lista de puntos unitarios (x, y, z)
    en el marco de escritura. Devuelve la secuencia de símbolos (0..31).
    """
    paso = STEP / (escala(strokes) if normalizar else 1.0)
    simbolos: list[int] = []
    ultimo = None
    for trazo in strokes:
        if len(trazo) == 0:
            continue
        pts = np.asarray(trazo, dtype=float)
        if ultimo is not None:                                  # salto con el lápiz arriba
            salto = remuestrear(np.array([ultimo, pts[0]]), paso)
            simbolos += _direcciones(salto, desplazar=FE.PEN_UP)
        rs = remuestrear(pts, paso)
        simbolos += _direcciones(rs)
        ultimo = rs[-1]
    return simbolos


# ---------------------------------------------------------------------- #
# Para entrenar: "pegar" un dibujo 2D sobre la esfera
# ---------------------------------------------------------------------- #
def pegar_en_esfera(trazos_2d, centro_lat: float = 0.0, centro_lon: float = 0.0) -> list[np.ndarray]:
    """
    Lleva un dibujo del lienzo 2D (0..1000, y hacia abajo) a la superficie de la
    esfera con el mapa exponencial alrededor de un centro (lat, lon en radianes):
    el dibujo se "envuelve" sobre la bola sin estirarse cerca del centro.
    """
    centro = desde_esfericas(np.pi / 2 - centro_lat, centro_lon)
    este, norte = base_tangente(centro)
    escala_rad = np.radians(GRADOS_POR_UNIDAD)
    salida = []
    for t in trazos_2d:
        if len(t) == 0:
            continue
        xy = np.asarray(t, dtype=float)[:, :2]
        u = (xy[:, 0] - 500.0) * escala_rad            # hacia el este
        v = (560.0 - xy[:, 1]) * escala_rad            # hacia el norte (la y del lienzo crece hacia abajo)
        w = u[:, None] * este + v[:, None] * norte
        a = np.linalg.norm(w, axis=1, keepdims=True)
        dirw = np.divide(w, a, out=np.zeros_like(w), where=a > 1e-12)
        salida.append(np.cos(a) * centro + np.sin(a) * dirw)
    return salida
