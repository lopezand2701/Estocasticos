# Decisiones técnicas

| # | Tema | Decisión | Por qué |
|---|------|----------|---------|
| 1 | Stack | Python + NumPy + PySide6 + PyQtGraph (GL) | Integración directa Qt/3D; NumPy es lo único permitido para el HMM |
| 2 | Topología | Izquierda-derecha (Bakis), N = 6–10 estados por letra | Un trazo avanza en fases; menos parámetros, aprende con pocos datos |
| 3 | Observaciones | Remuestreo por longitud de arco + código de cadena K = 16 + símbolo 16 = "levantar lápiz" (M = 17) | Independiente de la velocidad; soporta letras multitrazo (i, j, t, x, ñ, f, k) |
| 4 | Estabilidad | Forward/Backward escalados (c_t); Viterbi en log | Evitar underflow |
| 5 | Clasificación | P(c\|O) ∝ P(O\|λc)·P(c), softmax de log-verosimilitudes | Bayes; lista que suma 100 % |
| 6 | Tiempo real | Forward incremental: α_t → α_{t+1} con cada símbolo, O(N²) | No recalcular toda la secuencia |
| 7 | Proyección | Estado actual = argmax α_t; recorrer estados restantes emitiendo argmax B durante 1/(1−a_ii) pasos; histéresis al cambiar de letra | Simple, explicable en tablero, sin parpadeo |
| 8 | 3D | Raycast cámara→esfera; (x,y,z)→(θ,φ); dirección en plano tangente (este/norte); datos 3D generados desde 2D | Funciona cerca de polos; ahorra recolección |
| 9 | Arquitectura | HMMEngine / FeatureExtractor / GUI separados | Exigido por el enunciado; facilita pruebas |

## Notación (Rabiner, 1989)
- N: número de estados, M: número de símbolos, T: longitud de la secuencia
- A = [a_ij], B = [b_j(k)], π = [π_i], λ = (A, B, π)
- α_t(i) = P(O_1..O_t, q_t = i | λ)
- β_t(i) = P(O_{t+1}..O_T | q_t = i, λ)
- δ_t(i) = max sobre caminos de P(q_1..q_t = i, O_1..O_t | λ)
