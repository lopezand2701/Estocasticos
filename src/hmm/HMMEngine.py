"""
HMMEngine.py
============
Implementación nativa (solo NumPy) de un Modelo Oculto de Markov discreto
λ = (A, B, π), siguiendo la notación de Rabiner (1989).

Contiene:
    - Forward escalado            (problema 1: evaluación)        ecs. 18-21
    - Backward escalado           (usado en el entrenamiento)     ecs. 23-25
    - Viterbi en logaritmos       (problema 2: decodificación)    ecs. 30-35
    - Baum-Welch multisecuencia   (problema 3: entrenamiento)     ecs. 36-40
    - Paso Forward incremental    (clasificación en tiempo real)  ec. 20
    - Topología izquierda-derecha (Bakis)

Prohibido por el enunciado: hmmlearn, scikit-learn, scipy.stats, etc.
"""
from __future__ import annotations

import numpy as np


class HMM:
    """Modelo Oculto de Markov discreto con N estados y M símbolos."""

    def __init__(self, n_states: int, n_symbols: int, max_jump: int = 2,
                 A: np.ndarray | None = None, B: np.ndarray | None = None,
                 pi: np.ndarray | None = None) -> None:
        self.N = int(n_states)          # número de estados ocultos (fases del trazo)
        self.M = int(n_symbols)         # número de símbolos observables
        self.max_jump = int(max_jump)   # Δ: cuántos estados puede avanzar en un paso
        self.A = np.array(A, dtype=float) if A is not None else self._left_right_A()
        self.B = np.array(B, dtype=float) if B is not None else np.full((self.N, self.M), 1.0 / self.M)
        if pi is not None:
            self.pi = np.array(pi, dtype=float)
        else:
            self.pi = np.zeros(self.N)
            self.pi[0] = 1.0            # siempre se empieza en la fase 1
        # Duración promedio (en pasos) de cada fase, medida en el entrenamiento.
        self.durations = np.ones(self.N)
        # Probabilidad de que el trazo TERMINE estando en la fase i: P(parar | q_T = i).
        # Con todos en 1, Σ α_T(i)·η_i = Σ α_T(i)  (equivale a no usarla).
        self.end_probs = np.ones(self.N)

    # ------------------------------------------------------------------ #
    # Topología
    # ------------------------------------------------------------------ #
    def _left_right_A(self, stay: float = 0.6) -> np.ndarray:
        """Matriz izquierda-derecha: desde i solo se puede ir a i, i+1, ..., i+Δ."""
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            destinos = list(range(i, min(i + self.max_jump, self.N - 1) + 1))
            if len(destinos) == 1:              # último estado: absorbente
                A[i, i] = 1.0
                continue
            A[i, i] = stay
            resto = (1.0 - stay) / (len(destinos) - 1)
            for j in destinos[1:]:
                A[i, j] = resto
        return A

    # ------------------------------------------------------------------ #
    # Problema 1: Forward / Backward escalados
    # ------------------------------------------------------------------ #
    def forward(self, O) -> tuple[np.ndarray, np.ndarray]:
        """
        Devuelve (alpha_hat, c):
            alpha_hat[t, i] = α_t(i) normalizado para que cada fila sume 1
            c[t]            = coeficiente de escala 1 / Σ_i α_t(i)
        log P(O|λ) = -Σ_t log c[t]
        """
        O = np.asarray(O, dtype=int)
        T = len(O)
        alpha = np.zeros((T, self.N))
        c = np.zeros(T)

        a = self.pi * self.B[:, O[0]]                     # inicialización (19)
        c[0] = 1.0 / max(a.sum(), 1e-300)
        alpha[0] = a * c[0]

        for t in range(1, T):                             # inducción (20)
            a = (alpha[t - 1] @ self.A) * self.B[:, O[t]]
            c[t] = 1.0 / max(a.sum(), 1e-300)
            alpha[t] = a * c[t]
        return alpha, c

    def forward_step(self, alpha_prev: np.ndarray | None, o: int) -> tuple[np.ndarray, float]:
        """
        Un solo paso de la inducción (20), para usar en tiempo real.
        Recibe α̂_{t-1} (o None si es el primer símbolo) y el símbolo nuevo.
        Devuelve α̂_t y el incremento de log-verosimilitud ( = -log c_t ).
        """
        if alpha_prev is None:
            a = self.pi * self.B[:, o]
        else:
            a = (alpha_prev @ self.A) * self.B[:, o]
        s = max(a.sum(), 1e-300)
        return a / s, float(np.log(s))

    def backward(self, O, c: np.ndarray) -> np.ndarray:
        """β̂ escalado con los mismos coeficientes c de Forward (ecs. 24-25)."""
        O = np.asarray(O, dtype=int)
        T = len(O)
        beta = np.zeros((T, self.N))
        beta[T - 1] = c[T - 1]                            # β_T(i) = 1, escalado
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (self.B[:, O[t + 1]] * beta[t + 1])) * c[t]
        return beta

    def log_likelihood(self, O, completo: bool = False) -> float:
        """
        log P(O | λ) usando Forward escalado.
        - completo=False: ¿O podría ser el COMIENZO de esta letra?  Σ_i α_T(i)
        - completo=True:  ¿O podría ser la letra COMPLETA?         Σ_i α_T(i) · η_i
        """
        if len(O) == 0:
            return 0.0
        alpha, c = self.forward(O)
        ll = float(-np.log(c).sum())
        if completo:
            ll += self.log_end(alpha[-1])
        return ll

    def log_end(self, alpha_hat: np.ndarray) -> float:
        """log Σ_i α̂_T(i) · η_i : corrección para exigir que el trazo termine aquí."""
        return float(np.log(max(alpha_hat @ self.end_probs, 1e-300)))

    # ------------------------------------------------------------------ #
    # Problema 2: Viterbi (en logaritmos)
    # ------------------------------------------------------------------ #
    def viterbi(self, O) -> tuple[list[int], float]:
        """Devuelve (camino de estados más probable, log P*)."""
        O = np.asarray(O, dtype=int)
        T = len(O)
        with np.errstate(divide="ignore"):
            logA, logB, logpi = np.log(self.A), np.log(self.B), np.log(self.pi)
        delta = logpi + logB[:, O[0]]                     # (32a)
        psi = np.zeros((T, self.N), dtype=int)            # (32b)
        for t in range(1, T):
            cand = delta[:, None] + logA                  # cand[i, j] = δ_{t-1}(i) + log a_ij
            psi[t] = np.argmax(cand, axis=0)              # (33b)
            delta = cand[psi[t], np.arange(self.N)] + logB[:, O[t]]   # (33a)
        q = [int(np.argmax(delta))]                       # (34b)
        for t in range(T - 1, 0, -1):                     # retroceso (35)
            q.insert(0, int(psi[t][q[0]]))
        return q, float(np.max(delta))

    # ------------------------------------------------------------------ #
    # Problema 3: Baum-Welch con varias secuencias
    # ------------------------------------------------------------------ #
    def init_from_segments(self, sequences, smoothing: float = 0.5) -> None:
        """
        Inicialización por segmentación uniforme (Rabiner V-C):
        cada secuencia se parte en N tramos iguales; el tramo j alimenta al estado j.
        """
        counts = np.full((self.N, self.M), smoothing)
        lengths = []
        for O in sequences:
            T = len(O)
            bordes = np.linspace(0, T, self.N + 1)
            for j in range(self.N):
                ini, fin = int(round(bordes[j])), int(round(bordes[j + 1]))
                for o in O[ini:fin]:
                    counts[j, o] += 1
            lengths.append(T / self.N)
        self.B = counts / counts.sum(axis=1, keepdims=True)
        L = max(np.mean(lengths), 1.0)                    # pasos promedio por fase
        self.A = self._left_right_A(stay=float(np.clip(1.0 - 1.0 / L, 0.05, 0.95)))

    def fit(self, sequences, n_iter: int = 30, tol: float = 1e-4,
            floor: float = 1e-3, verbose: bool = False) -> list[float]:
        """
        Baum-Welch (EM). Devuelve la lista de log-verosimilitudes totales por iteración.
        - floor: piso ε para B (Rabiner V-D), evita probabilidades 0.
        - π se mantiene fijo en (1, 0, ..., 0) por ser izquierda-derecha.
        """
        sequences = [np.asarray(O, dtype=int) for O in sequences if len(O) > 0]
        historial: list[float] = []
        for it in range(n_iter):
            num_A = np.zeros((self.N, self.N))
            den_A = np.zeros(self.N)
            num_B = np.zeros((self.N, self.M))
            den_B = np.zeros(self.N)
            ocupacion = np.zeros(self.N)
            finales = np.zeros(self.N)
            total = 0.0

            for O in sequences:
                alpha, c = self.forward(O)
                beta = self.backward(O, c)
                total += -np.log(c).sum()

                # γ_t(i) ∝ α̂_t(i) β̂_t(i)   (ec. 27)
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)

                # Σ_t ξ_t(i,j) = Σ_t α̂_t(i) a_ij b_j(O_{t+1}) β̂_{t+1}(j)   (ec. 37)
                if len(O) > 1:
                    Bo = self.B[:, O[1:]].T                # (T-1, N)
                    num_A += self.A * (alpha[:-1].T @ (Bo * beta[1:]))
                    den_A += gamma[:-1].sum(axis=0)

                # conteo esperado de símbolos por estado (ec. 40c)
                G = np.zeros((self.M, self.N))
                np.add.at(G, O, gamma)
                num_B += G.T
                den_B += gamma.sum(axis=0)
                ocupacion += gamma.sum(axis=0)
                finales += gamma[-1]                       # en qué fase terminó este ejemplo

            # Reestimación (40b) y (40c)
            filas = den_A > 0
            nuevo_A = self.A.copy()
            nuevo_A[filas] = num_A[filas] / num_A[filas].sum(axis=1, keepdims=True)
            self.A = nuevo_A

            filas = den_B > 0
            nuevo_B = self.B.copy()
            nuevo_B[filas] = num_B[filas] / den_B[filas, None]
            nuevo_B = np.maximum(nuevo_B, floor)
            self.B = nuevo_B / nuevo_B.sum(axis=1, keepdims=True)

            self.durations = np.maximum(ocupacion / len(sequences), 0.0)
            # η_i = (veces que se terminó en i) / (veces que se estuvo en i), con un piso pequeño
            self.end_probs = np.maximum(finales / np.maximum(ocupacion, 1e-12), floor)
            historial.append(total)
            if verbose:
                print(f"  iter {it + 1:2d}  logP = {total:.2f}")
            if it > 0 and abs(historial[-1] - historial[-2]) < tol * abs(historial[-2]):
                break
        return historial

    # ------------------------------------------------------------------ #
    # Guardar / cargar
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "N": self.N, "M": self.M, "max_jump": self.max_jump,
            "A": self.A.round(8).tolist(), "B": self.B.round(8).tolist(),
            "pi": self.pi.tolist(), "durations": self.durations.round(4).tolist(),
            "end_probs": self.end_probs.round(6).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HMM":
        m = cls(d["N"], d["M"], d.get("max_jump", 2), d["A"], d["B"], d["pi"])
        m.durations = np.array(d.get("durations", np.ones(d["N"])), dtype=float)
        m.end_probs = np.array(d.get("end_probs", np.ones(d["N"])), dtype=float)
        return m
