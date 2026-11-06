"""
evaluate.py
===========
Mide qué tan bien reconoce el sistema, usando los autores de PRUEBA
(personas que no se usaron para entrenar).

Uso:
    python -m training.evaluate

Muestra:
    - precisión top-1 y top-3
    - precisión según el porcentaje del trazo dibujado (25 %, 50 %, 75 %, 100 %)
    - las parejas de letras que más se confunden
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.features import FeatureExtractor as FE
from src.hmm.Recognizer import Recognizer
from training.cargar_datos import buscar_json, cargar, es_valida


def ranking(rec: Recognizer, simbolos: list[int], completo: bool = False) -> list[str]:
    rec.reset()
    rec.update(simbolos)
    return [c for c, _ in rec.posteriors(completo)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default="models/2d/hmm_2d.json")
    ap.add_argument("--datos")
    args = ap.parse_args()

    info = json.loads(Path(args.modelos).read_text(encoding="utf-8"))
    autores = set(info.get("autores_prueba", []))
    if not autores:
        print("El modelo se entrenó con --todos: no hay autores de prueba separados.")
        return
    rec = Recognizer.load(args.modelos)
    ruta = Path(args.datos) if args.datos else buscar_json()
    prueba = [m for m in cargar(ruta) if es_valida(m) and m["autor"] in autores]
    prueba = [(m["letra"], FE.strokes_to_symbols(m["trazos"])) for m in prueba]
    prueba = [(c, s) for c, s in prueba if len(s) >= 3]
    print(f"Muestras de prueba: {len(prueba)} ({len(autores)} autores)\n")

    for frac in (0.25, 0.5, 0.75, 1.0):
        top1 = top3 = 0
        for c, s in prueba:
            r = ranking(rec, s[:max(1, int(len(s) * frac))], completo=frac == 1.0)
            top1 += r[0] == c
            top3 += c in r[:3]
        print(f"Trazo al {int(frac * 100):3d} %:  top-1 = {top1 / len(prueba):6.1%}   "
              f"top-3 = {top3 / len(prueba):6.1%}")

    errores = Counter()
    por_letra = Counter()
    aciertos = Counter()
    for c, s in prueba:
        pred = ranking(rec, s, completo=True)[0]
        por_letra[c] += 1
        if pred == c:
            aciertos[c] += 1
        else:
            errores[(c, pred)] += 1

    print("\nPrecisión por letra:")
    fila = [f"{c}:{aciertos[c] / por_letra[c]:4.0%}" for c in sorted(por_letra)]
    for i in range(0, len(fila), 9):
        print("  " + "  ".join(fila[i:i + 9]))
    print("\nConfusiones más frecuentes (real → predicha):")
    for (c, p), n in errores.most_common(10):
        print(f"  {c} → {p}: {n}")


if __name__ == "__main__":
    main()
