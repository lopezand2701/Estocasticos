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


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


class Recognizer:
    def __init__(self, models: dict[str, list[tuple[float, HMM]]], step: float = FE.STEP) -> None:
        # models[letra] = [(peso, hmm), (peso, hmm), ...]
        self.models = models
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
        Bayes con prior uniforme P(c) = 1/27:
            P(c | O) = P(O | c) / Σ_k P(O | k)
        Se calcula en logaritmos (softmax) para no tener desbordamientos.
        completo=True (lápiz levantado): exige que la letra pueda terminar aquí (η).
        Devuelve [(letra, prob)] ordenado de mayor a menor.
        """
        if not self.symbols:
            p = 1.0 / len(self.letters)
            return [(c, p) for c in self.letters]
        ll = np.array([self._log_letra(c, completo)[0] for c in self.letters])
        p = np.exp(ll - ll.max())
        p /= p.sum()
        orden = np.argsort(-p)
        return [(self.letters[i], float(p[i])) for i in orden]

    # ------------------------------------------------------------------ #
    # 2. Proyección de la trayectoria futura
    # ------------------------------------------------------------------ #
    def estado_actual(self, letter: str) -> dict | None:
        """
        Para la cadena de estados de la interfaz: de la mejor variante de `letter`,
        cuántas fases tiene (N), en cuál está ahora (s = argmax α̂_t) y la
        distribución α̂_t completa (probabilidad de cada fase en este instante).
        """
        if not self.symbols:
            return None
        _, v = self._log_letra(letter, completo=False)
        hmm = self.models[letter][v][1]
        alpha = self.alpha[letter][v]
        if alpha is None:
            return None
        return {"N": hmm.N, "fase": int(np.argmax(alpha)), "alpha": np.asarray(alpha),
                "variante": v + 1, "variantes": len(self.models[letter])}

    def project_steps(self, letter: str) -> list[tuple[np.ndarray, bool]]:
        """
        Pasos futuros (dirección, lápiz_abajo) según la mejor variante de `letter`:
            - fase actual   s = argmax_i α̂_t(i)  (estado más probable ahora)
            - pasos ya hechos en s: se cuentan con el camino de Viterbi
            - por cada fase restante j se dan durations[j] pasos en su
              dirección esperada  v_j = Σ_k b_j(k) · u_k
        La dirección viene en coordenadas de pantalla (x → derecha, y → abajo) y
        mide como mucho 1 paso. Sirve igual para el plano (2D) y la esfera (3D).
        """
        if not self.symbols:
            return []
        _, v = self._log_letra(letter, completo=False)
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

        pasos_futuros = []
        for j in range(s, hmm.N):
            pasos = hmm.durations[j] - (hechos if j == s else 0)
            pasos = int(round(max(pasos, 0)))
            if pasos == 0:
                continue
            b = hmm.B[j]
            abajo = b[FE.PEN_UP:].sum() < 0.5
            pesos = b[:FE.PEN_UP] if abajo else b[FE.PEN_UP:]
            vec = (pesos[:, None] * FE.UNIT).sum(axis=0) / max(pesos.sum(), 1e-12)
            pasos_futuros += [(vec, abajo)] * pasos
        return pasos_futuros

    def project(self, letter: str, start_point, escala: float = 1.0) -> list[tuple[np.ndarray, np.ndarray, bool]]:
        """
        Proyección en el PLANO: segmentos (p0, p1, lápiz_abajo) desde start_point.
        `escala` es el factor de normalización del dibujo: un paso de STEP en el
        espacio normalizado mide STEP / escala en el lienzo.
        """
        paso = self.step / max(escala, 1e-9)
        segmentos = []
        p = np.asarray(start_point, dtype=float)
        for vec, abajo in self.project_steps(letter):
            q = np.clip(p + vec * paso, 0, FE.ESPACIO)
            segmentos.append((p, q, abajo))
            p = q
        return segmentos
