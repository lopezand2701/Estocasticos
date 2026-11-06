"""Pruebas del extractor de características."""
import numpy as np

from src.features import FeatureExtractor as FE


def test_direcciones_basicas():
    # y crece hacia abajo en pantalla
    assert FE.quantize(10, 0) == 0     # →
    assert FE.quantize(0, -10) == 4    # ↑
    assert FE.quantize(-10, 0) == 8    # ←
    assert FE.quantize(0, 10) == 12    # ↓


def test_remuestreo_distancia_fija():
    linea = np.array([[0, 0], [100, 0]])
    rs = FE.resample(linea, step=25)
    assert len(rs) == 5
    assert np.allclose(np.diff(rs[:, 0]), 25)


def test_letra_L():
    trazo = [[100, 100], [100, 300], [300, 300]]     # bajar 200 y derecha 200
    s = FE.strokes_to_symbols([trazo], step=25)
    assert s == [12] * 8 + [0] * 8


def test_levantar_lapiz():
    i_palito = [[500, 400], [500, 700]]              # palito de la "i"
    i_punto = [[500, 250], [501, 251]]               # el punto, arriba
    s = FE.strokes_to_symbols([i_palito, i_punto], step=25)
    assert s[:12] == [12] * 12                       # bajar con el lápiz abajo
    assert all(x == FE.PEN_UP + 4 for x in s[12:])   # subir con el lápiz arriba


def test_prefijo_estable():
    """Al agregar puntos al final, los símbolos anteriores no cambian (necesario para Forward incremental)."""
    trazo = [[100, 100], [130, 160], [200, 210], [260, 230], [330, 300]]
    corto = FE.strokes_to_symbols([trazo[:3]])
    largo = FE.strokes_to_symbols([trazo])
    assert largo[:len(corto)] == corto
