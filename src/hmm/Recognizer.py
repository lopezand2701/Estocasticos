"""
Recognizer.py
=============
Usa los 27 modelos λ_c para:
    1. Clasificar en tiempo real: P(c | O_1:t) con Forward incremental + Bayes.
    2. Proyectar la trayectoria futura de la letra más probable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.hmm.HMMEngine import HMM
from src.features import FeatureExtractor as FE


class Recognizer:
    def __init__(self, models: dict[str, HMM], step: float = FE.STEP) -> None:
        self.models = models
        self.letters = list(models.keys())
        self.step = step
        self.reset()

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path) -> "Recognizer":
        """Carga el archivo JSON generado por training/train.py."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        models = {letra: HMM.from_dict(d) for letra, d in data["modelos"].items()}
        return cls(models, step=data.get("step", FE.STEP))

    def reset(self) -> None:
        """Olvida el trazo actual (botón Reiniciar captura)."""
        self.symbols: list[int] = []
        self.alpha = {c: None for c in self.letters}        # α̂_t de cada letra
        self.loglik = {c: 0.0 for c in self.letters}        # log P(O_1:t | λ_c)

    # ------------------------------------------------------------------ #
    # 1. Clasificación
    # ------------------------------------------------------------------ #
    def update(self, symbols: list[int]) -> None:
        """
        Recibe la secuencia completa hasta ahora. Si solo se agregaron símbolos al
        final, aplica un paso de Forward (ec. 20) por cada símbolo nuevo; si cambió
        el comienzo (deshacer / limpiar), recalcula desde cero.
        """
        n = len(self.symbols)
        if symbols[:n] != self.symbols:
            self.reset()
            n = 0
        for o in symbols[n:]:
            for c, hmm in self.models.items():
                self.alpha[c], inc = hmm.forward_step(self.alpha[c], o)
                self.loglik[c] += inc
        self.symbols = list(symbols)

    def posteriors(self, completo: bool = False) -> list[tuple[str, float]]:
        """
        Bayes con prior uniforme P(c) = 1/27:
            P(c | O) = P(O | λ_c) / Σ_k P(O | λ_k)
        Se calcula en logaritmos (softmax) para no tener desbordamientos.
        completo=True (lápiz levantado): exige que la letra pueda terminar aquí (η).
        Devuelve [(letra, prob)] ordenado de mayor a menor.
        """
        if not self.symbols:
            p = 1.0 / len(self.letters)
            return [(c, p) for c in self.letters]
        ll = np.array([self.loglik[c] + (self.models[c].log_end(self.alpha[c]) if completo else 0.0)
                       for c in self.letters])
        p = np.exp(ll - ll.max())
        p /= p.sum()
        orden = np.argsort(-p)
        return [(self.letters[i], float(p[i])) for i in orden]

    # ------------------------------------------------------------------ #
    # 2. Proyección de la trayectoria futura
    # ------------------------------------------------------------------ #
    def project(self, letter: str, start_point) -> list[tuple[np.ndarray, np.ndarray, bool]]:
        """
        Devuelve una lista de segmentos (p0, p1, lápiz_abajo) que continúan el dibujo
        desde start_point según el modelo de `letter`:
            - fase actual   s = argmax_i α̂_t(i)  (estado más probable ahora)
            - pasos ya hechos en s: se cuentan con el camino de Viterbi
            - por cada fase restante j se avanza durations[j] pasos en su
              dirección esperada  Σ_k b_j(k) · u_k
        """
        hmm = self.models[letter]
        if not self.symbols or self.alpha[letter] is None:
            return []
        s = int(np.argmax(self.alpha[letter]))
        camino, _ = hmm.viterbi(self.symbols)
        hechos = 0
        for q in reversed(camino):
            if q != s:
                break
            hechos += 1

        segmentos = []
        p = np.asarray(start_point, dtype=float)
        for j in range(s, hmm.N):
            pasos = hmm.durations[j] - (hechos if j == s else 0)
            pasos = int(round(max(pasos, 0)))
            if pasos == 0:
                continue
            b = hmm.B[j]
            p_arriba = b[FE.PEN_UP:].sum()
            abajo = p_arriba < 0.5
            pesos = b[:FE.PEN_UP] if abajo else b[FE.PEN_UP:]
            vec = (pesos[:, None] * FE.UNIT).sum(axis=0) / max(pesos.sum(), 1e-12)
            for _ in range(pasos):
                q = np.clip(p + vec * self.step, 0, FE.ESPACIO)
                segmentos.append((p, q, abajo))
                p = q
        return segmentos
