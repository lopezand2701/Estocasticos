"""Pruebas del modo 3D: coordenadas esféricas, rotación y extractor de la esfera."""
import numpy as np
import pytest

from src.features import FeatureExtractor as FE
from src.features import FeatureExtractor3D as F3


def test_esfericas_ida_y_vuelta():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(50, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    theta, phi = F3.a_esfericas(p)
    assert np.allclose(F3.desde_esfericas(theta, phi), p)


def test_centro_de_la_vista():
    # el punto que mira a la cámara está en el ecuador (θ = 90°) y en φ = 0
    theta, phi = F3.a_esfericas(np.array([0.0, 0.0, 1.0]))
    assert theta[0] == pytest.approx(np.pi / 2) and phi[0] == pytest.approx(0.0)


def test_rotacion_es_ortonormal_y_da_la_vuelta_completa():
    from src.gui.Sphere3D import rotacion
    R = rotacion(np.array([0.3, 1.0, 0.2]), np.radians(77))
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    # 360° en pasos de 30°: vuelve exactamente al inicio (giro sin límites)
    total = np.eye(3)
    for _ in range(12):
        total = rotacion(np.array([1.0, 0.0, 0.0]), np.radians(30)) @ total
    assert np.allclose(total, np.eye(3))


def test_moverse_sobre_la_esfera_no_se_sale():
    p = F3.desde_esfericas(1.1, 0.4)
    q = F3.mover(p, 0.05, -0.02)
    assert np.linalg.norm(q) == pytest.approx(1.0)


def test_letra_L_igual_que_en_2D():
    """Una "L" pegada en el centro de la esfera produce los mismos símbolos que en el plano."""
    L = [[[400, 300], [400, 650], [600, 650]]]
    puntos_2d = [[list(p) for p in np.linspace(L[0][0], L[0][1], 30)] +
                 [list(p) for p in np.linspace(L[0][1], L[0][2], 20)]]
    s2 = FE.strokes_to_symbols(puntos_2d)
    s3 = F3.strokes_to_symbols(F3.pegar_en_esfera(puntos_2d))
    # sobre la superficie curva la letra puede medir 1 paso menos; el resto coincide
    assert abs(len(s3) - len(s2)) <= 1
    n = min(len(s2), len(s3))
    assert s3[:n] == s2[:n]


def test_posicion_en_la_esfera_no_importa():
    """La misma letra pegada en otra parte de la esfera da casi la misma secuencia."""
    trazo = [[[500, 420], [500, 700]], [[500, 320], [501, 321]]]          # una "i"
    a = F3.strokes_to_symbols(F3.pegar_en_esfera(trazo, 0.0, 0.0))
    b = F3.strokes_to_symbols(F3.pegar_en_esfera(trazo, np.radians(20), np.radians(-30)))
    assert abs(len(a) - len(b)) <= 1
    n = min(len(a), len(b))
    assert np.mean(np.array(a[:n]) == np.array(b[:n])) > 0.9
