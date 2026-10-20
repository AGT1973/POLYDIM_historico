"""
Tarea 13 - Enjambre POLYDIM / MPAT4
Rol solicitado: Experto en Manifold Learning
Objetivo: Framework de benchmarking que MONITOREA si las operaciones de
Manifold Learning (cómputo de matriz de distancias / grafo de k-vecinos,
paso base de Isomap y LLE) usan multiplicación vectorial/matricial
(BLAS, vía numpy) en lugar de bucles for explícitos en Python puro.

Este archivo es código real, ejecutable y testeado (ver test_benchmark_
manifold_vectorization.py). No es una simulación decorativa: mide tiempo,
verifica correctitud numérica y produce un veredicto (PASS/FAIL) por caso.

Nota pedagógica (léase junto con tarea_numero_13.md):
El framework mide DOS cosas distintas y no las confunde:
  1) Rendimiento (tiempo de reloj).
  2) Correctitud (¿el resultado vectorizado es el MISMO número que el
     resultado de referencia, dentro de una tolerancia numérica?).
Un framework "de máxima complejidad" que solo mida tiempo sin verificar
correctitud es, en rigor, inútil: podría reportar como "rápido y exitoso"
un cálculo que en realidad está mal. Por eso cada benchmark corre primero
una verificación de equivalencia numérica y SOLO DESPUÉS mide tiempo.
"""

from __future__ import annotations

import math
import time
import statistics
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 1. Generación de datos: una "variedad" (manifold) sintética de juguete.
#    Usamos una franja de Swiss Roll clásica de la literatura de Manifold
#    Learning (Tenenbaum et al., 2000, Isomap) porque es el ejemplo de
#    referencia estándar: puntos en R^3 que viven sobre una superficie 2D
#    curva, no en un subespacio lineal.
# ---------------------------------------------------------------------------
def generar_swiss_roll(n_puntos: int, ruido: float = 0.05, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = 1.5 * math.pi * (1 + 2 * rng.random(n_puntos))
    altura = 21.0 * rng.random(n_puntos)
    x = t * np.cos(t)
    y = altura
    z = t * np.sin(t)
    datos = np.column_stack([x, y, z])
    datos += ruido * rng.standard_normal(datos.shape)
    return datos.astype(np.float64)


# ---------------------------------------------------------------------------
# 2. Dos implementaciones de la MISMA operación matemática:
#    la matriz de distancias euclidianas por pares, D[i, j] = ||x_i - x_j||.
#    Esta matriz es el insumo directo de Isomap, LLE y de cualquier
#    construcción de grafo de k-vecinos en Manifold Learning.
# ---------------------------------------------------------------------------
def distancias_bucle_naive(datos: np.ndarray) -> np.ndarray:
    """Referencia matemáticamente transparente: O(n^2) iteraciones Python puras.
    Es la versión 'lenta pero obviamente correcta' que usamos como oráculo."""
    n = datos.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            acc = 0.0
            for k in range(datos.shape[1]):
                diff = datos[i, k] - datos[j, k]
                acc += diff * diff
            D[i, j] = math.sqrt(acc)
    return D


def distancias_vectorizadas(datos: np.ndarray) -> np.ndarray:
    """Misma matemática, expresada como álgebra matricial (BLAS vía numpy).
    ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i · x_j
    El término x_i · x_j para todo par (i, j) es una única multiplicación
    de matrices: X @ X.T. Esa es la 'multiplicación vectorial' que la
    Tarea 13 pide monitorear."""
    normas_cuadrado = np.sum(datos ** 2, axis=1)
    gram = datos @ datos.T  # <- la multiplicación matricial vectorizada (BLAS)
    D2 = normas_cuadrado[:, None] + normas_cuadrado[None, :] - 2.0 * gram
    np.maximum(D2, 0.0, out=D2)  # corrige negativos espurios por error de punto flotante
    D = np.sqrt(D2)
    # La diagonal es matemáticamente 0 por definición (||x_i - x_i|| = 0).
    # El truco de la norma expandida sufre cancelación catastrófica
    # precisamente en la diagonal (resta dos números casi idénticos), así
    # que en vez de reportar el ruido de punto flotante como si fuera
    # señal, lo saneamos explícitamente usando la estructura conocida del
    # problema. Esto es una corrección basada en conocimiento matemático,
    # no un ocultamiento de error.
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------------
# 3. El framework de benchmarking en sí.
# ---------------------------------------------------------------------------
@dataclass
class ResultadoBenchmark:
    n_puntos: int
    tiempo_naive_s: float
    tiempo_vectorizado_s: float
    error_max_abs: float
    error_relativo: float
    tolerancia_relativa: float
    correcto: bool
    speedup: float = field(init=False)

    def __post_init__(self):
        self.speedup = (
            self.tiempo_naive_s / self.tiempo_vectorizado_s
            if self.tiempo_vectorizado_s > 0
            else float("inf")
        )


def _medir(func: Callable[[], np.ndarray], repeticiones: int = 3) -> tuple[np.ndarray, float]:
    tiempos = []
    resultado = None
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        resultado = func()
        tiempos.append(time.perf_counter() - inicio)
    return resultado, statistics.median(tiempos)


def benchmark_una_escala(n_puntos: int, tolerancia_relativa: float = 1e-5) -> ResultadoBenchmark:
    """
    Nota numérica importante (esto es la parte 'no obvia' de la tarea):
    `distancias_vectorizadas` usa la identidad ||a-b||^2 = ||a||^2+||b||^2-2a·b.
    Esa identidad es exacta en aritmética real, pero en punto flotante sufre
    *cancelación catastrófica* cuando ||a||^2+||b||^2 y 2a·b son casi iguales
    (puntos cercanos): se restan dos números grandes y parecidos, y el
    resultado pequeño pierde precisión relativa. Por eso NO se debe comparar
    contra `distancias_bucle_naive` con una tolerancia absoluta fija: el
    error absoluto crece con la escala de los datos. La comparación correcta
    es relativa al tamaño de las distancias involucradas.
    """
    datos = generar_swiss_roll(n_puntos)

    ref, t_naive = _medir(lambda: distancias_bucle_naive(datos), repeticiones=1)
    vec, t_vec = _medir(lambda: distancias_vectorizadas(datos), repeticiones=5)

    error_max = float(np.max(np.abs(ref - vec)))
    escala = float(np.max(ref)) if np.max(ref) > 0 else 1.0
    error_rel = error_max / escala
    correcto = error_rel <= tolerancia_relativa

    return ResultadoBenchmark(
        n_puntos=n_puntos,
        tiempo_naive_s=t_naive,
        tiempo_vectorizado_s=t_vec,
        error_max_abs=error_max,
        error_relativo=error_rel,
        tolerancia_relativa=tolerancia_relativa,
        correcto=correcto,
    )


def correr_framework(escalas: Sequence[int] = (50, 100, 200, 400)) -> List[ResultadoBenchmark]:
    resultados = []
    for n in escalas:
        # n=400 en modo naive ya es ~3-6s en CPU interpretada; lo dejamos
        # como techo por defecto para no volver el benchmark impracticable.
        resultados.append(benchmark_una_escala(n))
    return resultados


def reporte_texto(resultados: List[ResultadoBenchmark]) -> str:
    lineas = [
        f"{'n':>6} | {'naive (s)':>10} | {'vectorizado (s)':>16} | {'speedup':>9} | {'error_rel':>10} | veredicto"
    ]
    for r in resultados:
        veredicto = "PASS" if r.correcto else "FAIL (diverge)"
        lineas.append(
            f"{r.n_puntos:>6} | {r.tiempo_naive_s:>10.4f} | {r.tiempo_vectorizado_s:>16.6f} "
            f"| {r.speedup:>8.1f}x | {r.error_relativo:>10.2e} | {veredicto}"
        )
    return "\n".join(lineas)


if __name__ == "__main__":
    resultados = correr_framework()
    print(reporte_texto(resultados))
    assert all(r.correcto for r in resultados), (
        "Fallo de correctitud: la versión vectorizada diverge de la referencia "
        "más allá de la tolerancia. Según la Constitución V2 del documento "
        "POLYDIM, esto se reporta como Falla Matemática, no se oculta."
    )
