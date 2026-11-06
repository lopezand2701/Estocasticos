"""
cargar_datos.py
===============
Lee el JSON exportado por la página Glifo (dataset compartido del curso).

Acepta los dos formatos de exportación de Glifo:
    1. "Descargar con detalles":  {"muestras": [{"letra", "autor", "trazos", ...}, ...]}
    2. "Descargar dataset .json":  {"a": [trazos, trazos, ...], "b": [...], ...}

Cada muestra se devuelve como un diccionario:
    {"letra": "a", "autor": "andres lopez", "trazos": [[[x, y, t], ...], ...]}
"""
from __future__ import annotations

import json
from pathlib import Path

LETRAS = list("abcdefghijklmnñopqrstuvwxyz")


def cargar(ruta: str | Path) -> list[dict]:
    data = json.loads(Path(ruta).read_text(encoding="utf-8"))
    muestras = []
    if isinstance(data, dict) and "muestras" in data:              # formato con detalles
        for m in data["muestras"]:
            muestras.append({
                "letra": m.get("letra"),
                "autor": str(m.get("autor") or m.get("usuario") or "desconocido").strip().lower(),
                "trazos": m.get("trazos") or [],
            })
    else:                                                           # formato por letra (sin autor)
        for letra, lista in data.items():
            for i, trazos in enumerate(lista):
                muestras.append({"letra": letra, "autor": f"anonimo-{letra}-{i}", "trazos": trazos})
    return [m for m in muestras if m["letra"] in LETRAS]


def es_valida(m: dict, min_puntos: int = 8, min_tamano: float = 40.0) -> bool:
    """Descarta muestras vacías, con muy pocos puntos o diminutas (clics accidentales)."""
    puntos = [p for trazo in m["trazos"] for p in trazo]
    if len(puntos) < min_puntos:
        return False
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    return max(max(xs) - min(xs), max(ys) - min(ys)) >= min_tamano


def dividir_por_autor(muestras: list[dict], fraccion_prueba: float = 0.2, semilla: int = 7):
    """
    Separa entrenamiento y prueba POR AUTOR: las personas del conjunto de prueba
    no aparecen en el de entrenamiento. Así la precisión medida es honesta.
    """
    import random
    autores = sorted({m["autor"] for m in muestras})
    random.Random(semilla).shuffle(autores)
    n_prueba = max(1, int(len(autores) * fraccion_prueba))
    prueba = set(autores[:n_prueba])
    entrenamiento = [m for m in muestras if m["autor"] not in prueba]
    test = [m for m in muestras if m["autor"] in prueba]
    return entrenamiento, test


def buscar_json(carpeta: str | Path = "data/raw") -> Path:
    """Devuelve el JSON más reciente de la carpeta (por si no se indica uno)."""
    archivos = sorted(Path(carpeta).glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not archivos:
        raise FileNotFoundError(f"No hay archivos .json en {carpeta}. Exporta el dataset desde Glifo.")
    return archivos[-1]
