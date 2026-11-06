"""
Pruebas del motor HMM. Se ejecutan con:  pytest
Usan el mismo ejemplo de la letra "L" que resolvimos a mano en el día 2.
"""
import itertools

import numpy as np
import pytest

from src.hmm.HMMEngine import HMM

ABAJO, DERECHA, ARRIBA = 0, 1, 2


def modelo_L() -> HMM:
    A = [[0.7, 0.3], [0.0, 1.0]]
    B = [[0.7, 0.2, 0.1],     # fase 1: bajar
         [0.1, 0.6, 0.3]]     # fase 2: derecha
    return HMM(2, 3, A=A, B=B, pi=[1.0, 0.0])


def fuerza_bruta(hmm: HMM, O) -> float:
    """P(O|λ) sumando todos los caminos posibles (ec. 17). Solo sirve para T pequeño."""
    total = 0.0
    for Q in itertools.product(range(hmm.N), repeat=len(O)):
        p = hmm.pi[Q[0]] * hmm.B[Q[0], O[0]]
        for t in range(1, len(O)):
            p *= hmm.A[Q[t - 1], Q[t]] * hmm.B[Q[t], O[t]]
        total += p
    return total


def test_forward_ejemplo_a_mano():
    O = [ABAJO, ABAJO, DERECHA]
    assert np.exp(modelo_L().log_likelihood(O)) == pytest.approx(0.12236)


def test_alpha_escalado_ejemplo_a_mano():
    alpha, c = modelo_L().forward([ABAJO, ABAJO, DERECHA])
    assert alpha[1] == pytest.approx([0.9423, 0.0577], abs=1e-4)
    assert alpha[2] == pytest.approx([0.3924, 0.6076], abs=1e-4)
    assert c == pytest.approx([1.4286, 1.9231, 2.9748], abs=1e-4)


def test_forward_igual_a_fuerza_bruta():
    rng = np.random.default_rng(0)
    hmm = HMM(3, 4)
    hmm.B = rng.dirichlet(np.ones(4), size=3)
    for _ in range(5):
        O = rng.integers(0, 4, size=5)
        assert np.exp(hmm.log_likelihood(O)) == pytest.approx(fuerza_bruta(hmm, O))


def test_viterbi_ejemplo_a_mano():
    camino, logp = modelo_L().viterbi([ABAJO, ABAJO, DERECHA])
    assert camino == [0, 0, 1]                       # fase 1 → fase 1 → fase 2
    assert np.exp(logp) == pytest.approx(0.06174)


def test_secuencia_larga_no_da_cero():
    O = [ABAJO] * 300 + [DERECHA] * 300
    ll = modelo_L().log_likelihood(O)
    assert np.isfinite(ll) and ll < 0                # sin escalamiento sería 0 → log = -inf


def test_forward_incremental_igual_al_completo():
    hmm, O = modelo_L(), [ABAJO, ABAJO, DERECHA, DERECHA, ARRIBA]
    alpha, total = None, 0.0
    for o in O:
        alpha, inc = hmm.forward_step(alpha, o)
        total += inc
    assert total == pytest.approx(hmm.log_likelihood(O))


def test_probabilidad_de_terminar():
    """Una "c" completa no debe parecer una "a" completa solo por ser su comienzo."""
    hmm = modelo_L()
    hmm.end_probs = np.array([0.001, 1.0])           # la L casi nunca termina en la fase 1
    solo_bajar = [ABAJO, ABAJO, ABAJO]               # parece el comienzo de una L, no una L completa
    # α̂_3 ≈ [0.93, 0.07] → Σ α̂·η ≈ 0.07 → la verosimilitud "completa" baja ~2.7 en logaritmo
    assert hmm.log_likelihood(solo_bajar, completo=True) < hmm.log_likelihood(solo_bajar) - 2


def test_baum_welch_nunca_baja():
    rng = np.random.default_rng(1)
    datos = [[0] * rng.integers(3, 8) + [1] * rng.integers(3, 8) for _ in range(30)]
    hmm = HMM(2, 3, max_jump=1)
    hmm.init_from_segments(datos)
    historial = hmm.fit(datos, n_iter=15, tol=0)
    assert all(b >= a - 1e-6 for a, b in zip(historial, historial[1:]))
    assert hmm.A[1, 0] == 0.0                        # izquierda-derecha se conserva
    assert hmm.B[0, 0] > 0.9 and hmm.B[1, 1] > 0.9   # aprendió "bajar" y luego "derecha"
