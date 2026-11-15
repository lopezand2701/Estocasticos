"""
Recognizer.py
=============
Usa los modelos de las 27 letras para:
    1. Clasificar en tiempo real: P(c | O_1:t) con Forward incremental + Bayes.
    2. Proyectar la trayectoria futura de la letra más probable.

Cada letra puede tener VARIAS variantes (alógrafos), por ejemplo una "k"
hecha en 1, 2 o 3 trazos. Cada variante v es un HMM λ_{c,v} con un peso w_{c,v}
(fracción de ejemplos de entrenamiento de ese estilo). La verosimilitud de la
letra es una mezcla:
    P(O | c) = Σ_v  w_{c,v} · P(O | λ_{c,v})
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.features import FeatureExtractor as FE
from src.hmm.HMMEngine import HMM


# Calibración de las probabilidades (ajustada con los datos de prueba, ver la guía matemática).
# El HMM supone que cada flecha es independiente, pero en un trazo las flechas seguidas están
# correlacionadas: "cuenta varias veces" la misma evidencia y, entre más largo el trazo, más
# exagera su seguridad (llegaba a mostrar 100 %). Se corrige en dos pasos:
#   1. Verosimilitud POR SÍMBOLO:   P_κ(c|O) ∝ exp( κ · ℓ_c / T ),   ℓ_c = log P(O|c), T = nº de símbolos
#   2. Mezcla con una versión suave del mismo modelo (κ_s pequeño), que reparte una parte ε de
#      la probabilidad entre las letras MÁS PARECIDAS:  P(c|O) = (1−ε)·P_κ + ε·P_κs
# Ninguna letra llega a 100 % y la más parecida a la ganadora conserva una probabilidad visible.
# No cambia el orden de las letras: el acierto es el mismo.
# Con el lápiz levantado, si P(fin) ≥ UMBRAL_FIN la letra se da por terminada y no se proyecta.
# Medido con los datos de prueba: 93 % de las letras terminadas quedan sin proyección de sobra y
# 95 % de las letras de varios trazos siguen mostrando el trazo que falta entre un trazo y otro.
UMBRAL_FIN = 0.2

KAPPA = 12.0
KAPPA_SUAVE = 1.5
MEZCLA = 0.08


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


class Recognizer:
    def __init__(self, models: dict[str, list[tuple[float, HMM]]], step: float = FE.STEP,
                 kappa: float = KAPPA, kappa_suave: float = KAPPA_SUAVE, mezcla: float = MEZCLA) -> None:
        # models[letra] = [(peso, hmm), (peso, hmm), ...]
        self.models = models
        self.kappa = kappa
        self.kappa_suave = kappa_suave
        self.mezcla = mezcla
        self.letters = list(models.keys())
        self.step = step
        self.reset()

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path) -> "Recognizer":
        """Carga el archivo JSON generado por training/train.py."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        models = {}
        for letra, d in data["modelos"].items():
            if isinstance(d, dict):                          # formato antiguo: un solo HMM por letra
                models[letra] = [(1.0, HMM.from_dict(d))]
            else:                                            # formato nuevo: lista de variantes
                models[letra] = [(v["peso"], HMM.from_dict(v["hmm"])) for v in d]
        return cls(models, step=data.get("step", FE.STEP))

    def reset(self) -> None:
        """Olvida el trazo actual (botón Reiniciar captura)."""
        self.symbols: list[int] = []
        # Por cada letra y variante: α̂_t y log P(O_1:t | λ_{c,v})
        self.alpha = {c: [None] * len(v) for c, v in self.models.items()}
        self.loglik = {c: np.zeros(len(v)) for c, v in self.models.items()}

    # ------------------------------------------------------------------ #
    # 1. Clasificación
    # ------------------------------------------------------------------ #
    def update(self, symbols: list[int]) -> None:
        """
        Recibe la secuencia completa hasta ahora. Si solo se agregaron símbolos al
        final, aplica un paso de Forward (ec. 20) por cada símbolo nuevo; si cambió
        el comienzo (deshacer, limpiar o cambio de escala), recalcula desde cero.
        """
        n = len(self.symbols)
        if symbols[:n] != self.symbols:
            self.reset()
            n = 0
        for o in symbols[n:]:
            for c, variantes in self.models.items():
                for v, (_, hmm) in enumerate(variantes):
                    self.alpha[c][v], inc = hmm.forward_step(self.alpha[c][v], o)
                    self.loglik[c][v] += inc
        self.symbols = list(symbols)

    def _log_letra(self, c: str, completo: bool) -> tuple[float, int]:
        """log P(O | c) = log Σ_v w_v P(O | λ_v)  y el índice de la mejor variante."""
        puntajes = []
        for v, (peso, hmm) in enumerate(self.models[c]):
            ll = self.loglik[c][v]
            if completo:
                ll += hmm.log_end(self.alpha[c][v])
            puntajes.append(np.log(peso) + ll)
        puntajes = np.array(puntajes)
        return _logsumexp(puntajes), int(np.argmax(puntajes))

    def posteriors(self, completo: bool = False) -> list[tuple[str, float]]:
        """
        Bayes con prior uniforme P(c) = 1/27, calibrado:
            P_κ(c | O) = exp(κ·ℓ_c/T) / Σ_k exp(κ·ℓ_k/T)            (verosimilitud por símbolo)
            P(c | O)   = (1 − ε) · P_κ(c | O) + ε · P_κs(c | O)       (mezcla con versión suave)
        Se calcula en logaritmos (softmax) para no tener desbordamientos.
        completo=True (lápiz levantado): exige que la letra pueda terminar aquí (η).
        Devuelve [(letra, prob)] ordenado de mayor a menor.
        """
        if not self.symbols:
            p = 1.0 / len(self.letters)
            return [(c, p) for c in self.letters]
        ll = np.array([self._log_letra(c, completo)[0] for c in self.letters]) / len(self.symbols)

        def softmax(z: np.ndarray) -> np.ndarray:
            e = np.exp(z - z.max())
            return e / e.sum()

        p = (1.0 - self.mezcla) * softmax(self.kappa * ll) + self.mezcla * softmax(self.kappa_suave * ll)
        orden = np.argsort(-p)
        return [(self.letters[i], float(p[i])) for i in orden]

    # ------------------------------------------------------------------ #
    # 2. Proyección de la trayectoria futura
    # ------------------------------------------------------------------ #
    def prob_terminar(self, letter: str, v: int) -> float:
        """
        Probabilidad de que la letra ya esté terminada (variante v):
            P(fin) = Σ_i α̂_t(i) · η_i / max_i η_i
        Vale 1 si toda la probabilidad está en las fases donde las letras suelen terminar.
        """
        hmm = self.models[letter][v][1]
        alpha = self.alpha[letter][v]
        if alpha is None:
            return 0.0
        return float(np.asarray(alpha) @ hmm.end_probs / max(float(hmm.end_probs.max()), 1e-12))

    def estado_actual(self, letter: str, completo: bool = False) -> dict | None:
        """
        Para la cadena de estados de la interfaz: de la mejor variante de `letter`,
        cuántas fases tiene (N), en cuál está ahora (s = argmax α̂_t) y la
        distribución α̂_t completa (probabilidad de cada fase en este instante).
        """
        if not self.symbols:
            return None
        _, v = self._log_letra(letter, completo=completo)
        hmm = self.models[letter][v][1]
        alpha = self.alpha[letter][v]
        if alpha is None:
            return None
        return {"N": hmm.N, "fase": int(np.argmax(alpha)), "alpha": np.asarray(alpha),
                "variante": v + 1, "variantes": len(self.models[letter])}

    def project_steps(self, letter: str, completo: bool = False) -> list[tuple[np.ndarray, bool]]:
        """
        Pasos futuros (dirección, lápiz_abajo) según la mejor variante de `letter`:
            - fase actual   s = argmax_i α̂_t(i)  (estado más probable ahora)
            - pasos ya hechos en la fase actual n_s: camino de Viterbi
            - ritmo del usuario (qué tan grande escribe respecto a lo aprendido):
                  progreso esperado  E = Σ_i α̂_t(i) · ( d_1 + … + d_{i-1} + d_i / 2 )
                  ρ = t / E            (t = símbolos ya dibujados), limitado a [0.5, 3]
            - fase actual: faltan  ρ·d_s − n_s  pasos;  fases siguientes: ρ·d_j pasos,
              cada una en su dirección esperada  v_j = Σ_k b_j(k) · u_k
        Así una letra el doble de grande tiene una proyección el doble de larga.
        Con el lápiz levantado (completo=True) la variante se elige como letra COMPLETA
        (usando η), y si P(fin) ≥ UMBRAL_FIN no se proyecta nada: la letra ya está terminada.
        La dirección viene en coordenadas de pantalla (x → derecha, y → abajo) y
        mide como mucho 1 paso. Sirve igual para el plano (2D) y la esfera (3D).
        """
        if not self.symbols:
            return []
        _, v = self._log_letra(letter, completo=completo)
        if completo and self.prob_terminar(letter, v) >= UMBRAL_FIN:
            return []
        hmm = self.models[letter][v][1]
        alpha = self.alpha[letter][v]
        if alpha is None:
            return []
        s = int(np.argmax(alpha))
        camino, _ = hmm.viterbi(self.symbols)
        hechos = 0
        for q in reversed(camino):
            if q != s:
                break
            hechos += 1
        rho = self.ritmo(hmm, alpha, len(self.symbols))

        pasos_futuros = []
        for j in range(s, hmm.N):
            pasos = rho * hmm.durations[j] - (hechos if j == s else 0)
            pasos = int(round(max(pasos, 0)))
            if pasos == 0:
                continue
            b = hmm.B[j]
            abajo = b[FE.PEN_UP:].sum() < 0.5
            pesos = b[:FE.PEN_UP] if abajo else b[FE.PEN_UP:]
            vec = (pesos[:, None] * FE.UNIT).sum(axis=0) / max(pesos.sum(), 1e-12)
            pasos_futuros += [(vec, abajo)] * pasos
        return pasos_futuros

    @staticmethod
    def ritmo(hmm, alpha: np.ndarray, t: int) -> float:
        """
        ρ = símbolos dibujados / progreso esperado según el modelo.
        El progreso esperado promedia, con las probabilidades α̂_t(i) de cada fase,
        cuántos pasos se llevarían si se estuviera a mitad de esa fase:
            E = Σ_i α̂_t(i) · ( Σ_{j<i} d_j + d_i / 2 )
        ρ > 1: el usuario escribe más grande que los ejemplos de entrenamiento;
        ρ < 1: más pequeño. Así la proyección es proporcional al trazo.
        """
        acumulado = np.concatenate([[0.0], np.cumsum(hmm.durations)[:-1]])
        esperado = float(np.asarray(alpha) @ (acumulado + hmm.durations / 2))
        if esperado < 3:                     # aún muy poco dibujado para estimarlo
            return 1.0
        return float(np.clip(t / esperado, 0.5, 3.0))

    def project(self, letter: str, start_point, escala: float = 1.0,
                completo: bool = False) -> list[tuple[np.ndarray, np.ndarray, bool]]:
        """
        Proyección en el PLANO: segmentos (p0, p1, lápiz_abajo) desde start_point.
        `escala` es el factor de normalización del dibujo: un paso de STEP en el
        espacio normalizado mide STEP / escala en el lienzo.
        """
        paso = self.step / max(escala, 1e-9)
        segmentos = []
        p = np.asarray(start_point, dtype=float)
        for vec, abajo in self.project_steps(letter, completo):
            q = np.clip(p + vec * paso, 0, FE.ESPACIO)
            segmentos.append((p, q, abajo))
            p = q
        return segmentos
