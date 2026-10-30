# Reconocedor de Caracteres 2D/3D con HMM

Proyecto final — Procesos Estocásticos (603504), Universidad de los Llanos.

Aplicación de escritorio que reconoce en tiempo real las 27 letras minúsculas del
alfabeto español dibujadas a mano alzada (lienzo plano 2D y esfera 3D), usando
Cadenas Ocultas de Markov implementadas desde cero (solo NumPy).

## Estructura
```
src/hmm/        HMMEngine.py        -> Forward, Backward, Viterbi, Baum-Welch, proyección
src/features/   FeatureExtractor.py -> remuestreo, código de cadena, coordenadas esféricas
src/gui/        Vistas (menú, 2D, 3D)
training/       Scripts offline: recolección, entrenamiento, evaluación
models/2d, 3d/  Parámetros λ = (A, B, π) exportados en JSON
data/           Muestras de trazos
tests/          Pruebas unitarias
assets/icons/   Íconos de referencia de cada letra
docs/           Decisiones técnicas y material de sustentación
```

## Instalación (Windows)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
