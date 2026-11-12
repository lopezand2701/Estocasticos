"""
evaluate.py
===========
Mide qué tan bien reconoce el sistema, usando las muestras de PRUEBA
(las que se separaron al entrenar y el modelo nunca vio).

Uso:
    python -m training.evaluate              # modelos 2D
    python -m training.evaluate --modo 3d    # modelos 3D (las letras se pegan sobre la esfera)

Muestra:
    - precisión top-1 y top-3 según el porcentaje del trazo dibujado
    - prueba de robustez: las mismas letras dibujadas más pequeñas o más
      grandes (×0.2 a ×1.6) y en otra posición del lienzo
    - precisión por letra y las confusiones más frecuentes
"""
from __future__ import annotations

import argparse
import json
import random
import zlib
from collections import Counter
from pathlib import Path

from src.features import FeatureExtractor as FE
from src.hmm.Recognizer import Recognizer
from training.cargar_datos import cargar_varios, es_valida, todos_los_json
from training.train import simbolos

MODO = "2d"


def ranking(rec: Recognizer, trazos, frac: float = 1.0) -> list[str]:
    """Clasifica las primeras `frac` partes de los símbolos del dibujo."""
    # en 3D cada letra se pega en una posición de la esfera que depende de la propia letra
    rnd = random.Random(zlib.crc32(str(trazos[0][:2]).encode()))
    s = simbolos(trazos, MODO, rnd)
    rec.reset()
    rec.update(s[:max(1, int(len(s) * frac))])
    return [c for c, _ in rec.posteriors(completo=frac == 1.0)]


def transformar(trazos, rnd: random.Random):
    """Cambia tamaño y posición del dibujo, como si se hubiera hecho en otra parte del lienzo."""
    pts = [p for t in trazos for p in t]
    x0 = min(p[0] for p in pts); y0 = min(p[1] for p in pts)
    x1 = max(p[0] for p in pts); y1 = max(p[1] for p in pts)
    k = rnd.uniform(0.2, 1.6)
    k = min(k, 950 / max(x1 - x0, y1 - y0, 1))
    dx = rnd.uniform(20, 980 - (x1 - x0) * k)
    dy = rnd.uniform(20, 980 - (y1 - y0) * k)
    return [[[(p[0] - x0) * k + dx, (p[1] - y0) * k + dy] for p in t] for t in trazos]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["2d", "3d"], default="2d")
    ap.add_argument("--modelos", help="por defecto models/<modo>/hmm_<modo>.json")
    ap.add_argument("--datos", nargs="*")
    args = ap.parse_args()
    global MODO
    MODO = args.modo
    args.modelos = args.modelos or f"models/{MODO}/hmm_{MODO}.json"
    print(f"Modo: {MODO.upper()}  ·  modelos: {args.modelos}")

    info = json.loads(Path(args.modelos).read_text(encoding="utf-8"))
    autores = set(info.get("autores_prueba", []))
    if not autores:
        print("El modelo se entrenó con --todos: no hay muestras de prueba separadas.")
        return
    rec = Recognizer.load(args.modelos)
    rutas = [Path(r) for r in args.datos] if args.datos else todos_los_json()
    prueba = [m for m in cargar_varios(rutas) if es_valida(m) and m["autor"] in autores]
    prueba = [(m["letra"], [t for t in m["trazos"] if t]) for m in prueba]
    prueba = [(c, t) for c, t in prueba if len(FE.strokes_to_symbols(t)) >= 3]
    print(f"Muestras de prueba: {len(prueba)}\n")

    for frac in (0.25, 0.5, 0.75, 1.0):
        top1 = top3 = 0
        for c, t in prueba:
            r = ranking(rec, t, frac)
            top1 += r[0] == c
            top3 += c in r[:3]
        print(f"Trazo al {int(frac * 100):3d} %:  top-1 = {top1 / len(prueba):6.1%}   "
              f"top-3 = {top3 / len(prueba):6.1%}")

    rnd = random.Random(1)
    top1 = sum(ranking(rec, transformar(t, rnd))[0] == c for c, t in prueba)
    print(f"\nRobustez (tamaño ×0.2–×1.6 y posición al azar{' sobre la esfera' if MODO == '3d' else ''}):  top-1 = {top1 / len(prueba):6.1%}")

    errores, por_letra, aciertos = Counter(), Counter(), Counter()
    for c, t in prueba:
        pred = ranking(rec, t)[0]
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
