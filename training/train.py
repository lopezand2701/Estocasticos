"""
train.py
========
Entrenamiento OFFLINE de los 27 modelos λ_c = (A_c, B_c, π_c) con Baum-Welch.

Uso (desde la carpeta Proyecto, con el .venv activo):
    python -m training.train                     # usa el JSON más reciente de data/raw
    python -m training.train --datos data/raw/archivo.json --todos

Por defecto entrena solo con el 80 % de los autores y guarda la lista de
autores de prueba, para que training/evaluate.py mida la precisión con
personas que el modelo nunca vio. Con --todos entrena con todos los datos
(úsalo para el modelo final de la entrega).

Salida: models/2d/hmm_2d.json  (lo carga la interfaz al iniciar)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from src.features import FeatureExtractor as FE
from src.hmm.HMMEngine import HMM
from training.cargar_datos import LETRAS, buscar_json, cargar, dividir_por_autor, es_valida


def numero_de_estados(secuencias: list[list[int]], pasos_por_estado: float = 4.0,
                      minimo: int = 4, maximo: int = 14) -> int:
    """N depende de qué tan larga es la letra: unas 4 flechas por fase (mediana)."""
    mediana = float(np.median([len(s) for s in secuencias]))
    return int(np.clip(round(mediana / pasos_por_estado), minimo, maximo))


def preparar(muestras: list[dict]) -> dict[str, list[list[int]]]:
    """Convierte cada muestra en su secuencia de símbolos y las agrupa por letra."""
    por_letra: dict[str, list[list[int]]] = {c: [] for c in LETRAS}
    for m in muestras:
        s = FE.strokes_to_symbols(m["trazos"])
        if 3 <= len(s) <= 250:
            por_letra[m["letra"]].append(s)
    return por_letra


def entrenar(por_letra: dict[str, list[list[int]]], max_por_letra: int = 120,
             n_iter: int = 30, semilla: int = 0) -> dict[str, HMM]:
    rnd = random.Random(semilla)
    modelos = {}
    for c in LETRAS:
        secs = por_letra[c]
        if not secs:
            print(f"  [{c}] sin datos, se omite")
            continue
        if len(secs) > max_por_letra:                # balancear letras con demasiadas muestras
            secs = rnd.sample(secs, max_por_letra)
        N = numero_de_estados(secs)
        hmm = HMM(N, FE.M, max_jump=2)
        hmm.init_from_segments(secs)
        t0 = time.time()
        hist = hmm.fit(secs, n_iter=n_iter)
        print(f"  [{c}] muestras={len(secs):3d}  N={N:2d}  iter={len(hist):2d}  "
              f"logP/muestra={hist[-1] / len(secs):8.2f}  ({time.time() - t0:.1f}s)")
        modelos[c] = hmm
    return modelos


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena los HMM de las 27 letras (2D).")
    ap.add_argument("--datos", help="JSON exportado de Glifo (por defecto el más reciente de data/raw)")
    ap.add_argument("--salida", default="models/2d/hmm_2d.json")
    ap.add_argument("--todos", action="store_true", help="entrenar con todos los autores (modelo final)")
    ap.add_argument("--max-por-letra", type=int, default=120)
    ap.add_argument("--iter", type=int, default=30)
    args = ap.parse_args()

    ruta = Path(args.datos) if args.datos else buscar_json()
    print(f"Datos: {ruta}")
    muestras = [m for m in cargar(ruta) if es_valida(m)]
    print(f"Muestras válidas: {len(muestras)}")

    if args.todos:
        entrenamiento, prueba = muestras, []
    else:
        entrenamiento, prueba = dividir_por_autor(muestras)
    autores_prueba = sorted({m["autor"] for m in prueba})
    print(f"Entrenamiento: {len(entrenamiento)} muestras | Prueba: {len(prueba)} muestras "
          f"de {len(autores_prueba)} autores")

    modelos = entrenar(preparar(entrenamiento), args.max_por_letra, args.iter)

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({
        "version": 1,
        "modo": "2D",
        "datos": ruta.name,
        "step": FE.STEP, "K": FE.K, "M": FE.M,
        "autores_prueba": autores_prueba,
        "modelos": {c: m.to_dict() for c, m in modelos.items()},
    }, ensure_ascii=False), encoding="utf-8")
    print(f"\nModelos guardados en {salida}")


if __name__ == "__main__":
    main()
