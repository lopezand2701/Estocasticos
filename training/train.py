"""
train.py
========
Entrenamiento OFFLINE de los modelos de las 27 letras con Baum-Welch.

Uso (desde la carpeta Proyecto, con el .venv activo):
    python -m training.train                 # usa TODOS los JSON de data/raw (sin repetidos)
    python -m training.train --datos data/raw/archivo.json
    python -m training.train --todos         # modelo final: entrena con el 100 % de los datos
    python -m training.train --modo 3d       # modelos para la esfera (models/3d/hmm_3d.json)

Novedades:
    - Variantes por número de trazos (alógrafos): cada letra puede tener un HMM
      para la versión de 1 trazo, otro para 2 trazos y otro para 3 o más,
      siempre que haya al menos MIN_VARIANTE ejemplos de ese estilo.
      El peso de cada variante es la fracción de ejemplos que la usan.
    - Los dibujos se normalizan de tamaño antes de convertirlos en símbolos
      (ver FeatureExtractor.escala).

Por defecto separa un 20 % para prueba (por autor si el JSON trae autores)
y guarda esa lista para que training/evaluate.py mida la precisión.

Modo 3D: los mismos dibujos se "pegan" sobre la esfera en posiciones al azar
y se convierten en símbolos con coordenadas esféricas (FeatureExtractor3D).

Salida: models/2d/hmm_2d.json  o  models/3d/hmm_3d.json  (los carga la interfaz al iniciar)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.features import FeatureExtractor as FE
from src.features import FeatureExtractor3D as F3
from src.hmm.HMMEngine import HMM
from training.cargar_datos import (LETRAS, cargar_varios, dividir_por_autor, es_valida,
                                   todos_los_json)

MIN_VARIANTE = 15      # ejemplos mínimos para que un estilo tenga su propio HMM
MAX_TRAZOS = 3         # 3 significa "3 o más trazos"


def numero_de_estados(secuencias: list[list[int]], pasos_por_estado: float = 4.0,
                      minimo: int = 4, maximo: int = 14) -> int:
    """N depende de qué tan larga es la letra: unas 4 flechas por fase (mediana)."""
    mediana = float(np.median([len(s) for s in secuencias]))
    return int(np.clip(round(mediana / pasos_por_estado), minimo, maximo))


def simbolos(trazos, modo: str = "2d", rnd: random.Random | None = None) -> list[int]:
    """
    2D: símbolos directos del dibujo.
    3D: el dibujo se "pega" sobre la esfera en una posición al azar
        (latitud ±25°, longitud ±35°) y se extraen los símbolos con (θ, φ).
    """
    if modo == "3d":
        rnd = rnd or random.Random(0)
        lat = np.radians(rnd.uniform(-25, 25))
        lon = np.radians(rnd.uniform(-35, 35))
        return F3.strokes_to_symbols(F3.pegar_en_esfera(trazos, lat, lon))
    return FE.strokes_to_symbols(trazos)


def preparar(muestras: list[dict], modo: str = "2d", semilla: int = 0) -> dict[str, dict[int, list[list[int]]]]:
    """
    Convierte cada muestra en su secuencia de símbolos y las agrupa
    por letra y por número de trazos: grupos[letra][n_trazos] = [secuencias].
    """
    rnd = random.Random(semilla)
    grupos: dict[str, dict[int, list[list[int]]]] = {c: defaultdict(list) for c in LETRAS}
    for m in muestras:
        trazos = [t for t in m["trazos"] if len(t) > 0]
        s = simbolos(trazos, modo, rnd)
        if 3 <= len(s) <= 250:
            grupos[m["letra"]][min(len(trazos), MAX_TRAZOS)].append(s)
    return grupos


def variantes_de(grupos: dict[int, list[list[int]]]) -> list[list[list[int]]]:
    """Los estilos con pocos ejemplos se unen al estilo más frecuente."""
    orden = sorted(grupos, key=lambda k: -len(grupos[k]))
    if not orden:
        return []
    principal = list(grupos[orden[0]])
    otras = []
    for k in orden[1:]:
        if len(grupos[k]) >= MIN_VARIANTE:
            otras.append(list(grupos[k]))
        else:
            principal += grupos[k]
    return [principal] + otras


def entrenar(grupos, max_por_variante: int = 150, n_iter: int = 30, semilla: int = 0):
    rnd = random.Random(semilla)
    modelos = {}
    for c in LETRAS:
        variantes = variantes_de(grupos[c])
        if not variantes:
            print(f"  [{c}] sin datos, se omite")
            continue
        total = sum(len(v) for v in variantes)
        lista = []
        for secs in variantes:
            peso = len(secs) / total
            if len(secs) > max_por_variante:
                secs = rnd.sample(secs, max_por_variante)
            N = numero_de_estados(secs)
            hmm = HMM(N, FE.M, max_jump=2)
            hmm.init_from_segments(secs)
            t0 = time.time()
            hist = hmm.fit(secs, n_iter=n_iter)
            print(f"  [{c}] variante {len(lista) + 1}: muestras={len(secs):3d}  peso={peso:4.2f}  "
                  f"N={N:2d}  iter={len(hist):2d}  logP/muestra={hist[-1] / len(secs):8.2f}  "
                  f"({time.time() - t0:.1f}s)")
            lista.append((peso, hmm))
        modelos[c] = lista
    return modelos


MARGEN_LL = 1.1       # margen de seguridad sobre el percentil 1 de la verosimilitud por símbolo
MARGEN_T = 1.3        # margen sobre el percentil 99 de la longitud
MARGEN_GIRO = 1.3     # margen sobre el percentil 99 del giro total
# Con estos márgenes (medido con las muestras de prueba): ~1 % de letras reales rechazadas,
# ~88 % de tachones de ida y vuelta y ~86 % de garabatos al azar rechazados.


def umbrales_rechazo(modelos, grupos, max_muestras: int = 200, semilla: int = 0) -> dict:
    """
    Para cada letra, con SUS ejemplos de entrenamiento:
        ll_min   = MARGEN_LL   · percentil 1 de  ℓ/T   (ℓ = log P(O | letra completa), T = nº de símbolos)
        T_max    = MARGEN_T    · percentil 99 de T
        giro_max = MARGEN_GIRO · percentil 99 del giro total (FE.giro_total)
    Un trazo cuya letra ganadora quede por debajo de ll_min o por encima de T_max o giro_max
    se rechaza: no se parece lo suficiente a ninguna letra (por ejemplo, un garabato).
    """
    rnd = random.Random(semilla)
    umbrales = {}
    for c, lista in modelos.items():
        secs = [s for k in grupos[c] for s in grupos[c][k]]
        if len(secs) > max_muestras:
            secs = rnd.sample(secs, max_muestras)
        llT, Ts, giros = [], [], []
        for O in secs:
            ll = [np.log(w) + h.log_likelihood(O, completo=True) for w, h in lista]
            m = max(ll)
            llT.append((m + np.log(np.sum(np.exp(np.array(ll) - m)))) / len(O))
            Ts.append(len(O))
            giros.append(FE.giro_total(O))
        umbrales[c] = {"ll_min": round(float(np.percentile(llT, 1)) * MARGEN_LL, 4),
                       "T_max": round(float(np.percentile(Ts, 99)) * MARGEN_T, 1),
                       "giro_max": round(float(np.percentile(giros, 99)) * MARGEN_GIRO, 1)}
    return umbrales


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena los HMM de las 27 letras (2D).")
    ap.add_argument("--datos", nargs="*", help="JSON de Glifo (por defecto: todos los de data/raw)")
    ap.add_argument("--modo", choices=["2d", "3d"], default="2d")
    ap.add_argument("--salida", help="por defecto models/2d/hmm_2d.json o models/3d/hmm_3d.json")
    ap.add_argument("--todos", action="store_true", help="entrenar con todos los datos (modelo final)")
    ap.add_argument("--max-por-variante", type=int, default=150)
    ap.add_argument("--iter", type=int, default=30)
    args = ap.parse_args()

    rutas = [Path(r) for r in args.datos] if args.datos else todos_los_json()
    print("Datos:", ", ".join(str(r) for r in rutas))
    muestras = [m for m in cargar_varios(rutas) if es_valida(m)]
    print(f"Muestras válidas (sin repetidas): {len(muestras)}")

    if args.todos:
        entrenamiento, prueba = muestras, []
    else:
        entrenamiento, prueba = dividir_por_autor(muestras)
    autores_prueba = sorted({m["autor"] for m in prueba})
    print(f"Entrenamiento: {len(entrenamiento)} muestras | Prueba: {len(prueba)} muestras")

    print(f"Modo: {args.modo.upper()}")
    grupos = preparar(entrenamiento, args.modo)
    modelos = entrenar(grupos, args.max_por_variante, args.iter)
    rechazo = umbrales_rechazo(modelos, grupos)
    print("Umbrales de rechazo calculados (para reconocer garabatos que no son letras).")

    salida = Path(args.salida or f"models/{args.modo}/hmm_{args.modo}.json")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({
        "version": 2,
        "modo": args.modo.upper(),
        "datos": [r.name for r in rutas],
        "step": FE.STEP, "K": FE.K, "M": FE.M, "tam_normal": list(FE.TAM_NORMAL),
        "autores_prueba": autores_prueba,
        "rechazo": rechazo,
        "modelos": {c: [{"peso": round(p, 6), "hmm": m.to_dict()} for p, m in lista]
                    for c, lista in modelos.items()},
    }, ensure_ascii=False), encoding="utf-8")
    n_var = sum(len(v) for v in modelos.values())
    print(f"\n{n_var} modelos ({len(modelos)} letras) guardados en {salida}")


if __name__ == "__main__":
    main()
