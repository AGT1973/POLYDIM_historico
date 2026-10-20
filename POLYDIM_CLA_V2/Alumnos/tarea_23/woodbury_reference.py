"""
Tarea 23 — Referencia en Python (numpy) de la identidad de Woodbury
aplicada a la inversión de covarianzas diagonal + bajo rango (DPLR):

    Sigma = D + U U^T ,  D = diag(d_1..d_d) > 0,  U in R^{d x r},  r << d

    Sigma^{-1} = D^{-1} - D^{-1} U (I_r + U^T D^{-1} U)^{-1} U^T D^{-1}

Este script:
1. Verifica numéricamente que la fórmula de Woodbury coincide con la
   inversión densa (np.linalg.inv) en un caso bien condicionado.
2. Mide el costo empírico (tiempo) de Woodbury vs. inversión densa al
   crecer d, manteniendo r fijo, para contrastar la afirmación de
   complejidad O(d*r) del paper POLYDIM (Sección 5.1) frente a O(d^3).
3. Reproduce el caso de falla numérica (D con una entrada ~0) y
   documenta qué pasa en vez de ocultarlo.

Cómo correrlo:
    pip install numpy --break-system-packages   # si hace falta
    python3 woodbury_reference.py
"""

import time
import numpy as np


def woodbury_inv(d_diag: np.ndarray, U: np.ndarray, rcond: float = 1e-12):
    """Invierte Sigma = diag(d_diag) + U @ U.T vía Woodbury.

    Devuelve (Sigma_inv, ok). ok=False si D no es invertible dentro de rcond.
    """
    if np.any(np.abs(d_diag) < rcond):
        return None, False

    d_inv = 1.0 / d_diag                      # (d,)
    Dinv_U = d_inv[:, None] * U               # D^{-1} U      -> (d, r)

    r = U.shape[1]
    inner = np.eye(r) + U.T @ Dinv_U          # (r, r), el único bloque "caro"

    try:
        inner_inv = np.linalg.inv(inner)
    except np.linalg.LinAlgError:
        return None, False

    mid = Dinv_U @ inner_inv @ Dinv_U.T       # (d, d)
    Sigma_inv = np.diag(d_inv) - mid
    return Sigma_inv, True


def naive_inv(d_diag: np.ndarray, U: np.ndarray):
    Sigma = np.diag(d_diag) + U @ U.T
    return np.linalg.inv(Sigma)


def check_case_1(seed: int = 42, d: int = 50, r: int = 3):
    rng = np.random.default_rng(seed)
    d_diag = rng.uniform(0.5, 2.0, size=d)
    U = rng.normal(size=(d, r))

    sigma_inv_wb, ok = woodbury_inv(d_diag, U)
    assert ok, "Woodbury debería converger en el caso bien condicionado"
    sigma_inv_naive = naive_inv(d_diag, U)

    diff = np.linalg.norm(sigma_inv_wb - sigma_inv_naive, ord="fro")
    print(f"[Caso 1] d={d}, r={r} -> ||Woodbury - naive||_F = {diff:.3e}")
    assert diff < 1e-8, "Woodbury y la inversión densa deberían coincidir"
    print("[Caso 1] OK: la identidad de Woodbury reproduce la inversión densa.\n")


def benchmark_scaling(r: int = 3, sizes=(50, 200, 800, 2000)):
    print(f"[Benchmark] r fijo = {r}, variando d (promedio de 3 corridas)")
    print(f"{'d':>6} | {'t_naive (s)':>12} | {'t_woodbury (s)':>15} | speedup")
    rng = np.random.default_rng(0)
    for d in sizes:
        d_diag = rng.uniform(0.5, 2.0, size=d)
        U = rng.normal(size=(d, r))

        t0 = time.perf_counter()
        for _ in range(3):
            naive_inv(d_diag, U)
        t_naive = (time.perf_counter() - t0) / 3

        t0 = time.perf_counter()
        for _ in range(3):
            woodbury_inv(d_diag, U)
        t_woodbury = (time.perf_counter() - t0) / 3

        speedup = t_naive / t_woodbury if t_woodbury > 0 else float("inf")
        print(f"{d:>6} | {t_naive:12.6f} | {t_woodbury:15.6f} | {speedup:6.1f}x")
    print()


def check_case_2_failure(d: int = 20, r: int = 2):
    """D con una entrada ~0: Sigma sigue siendo invertible (porque U U^T puede
    compensar esa fila), pero la fórmula de Woodbury tal como está planteada
    ASUME D invertible por separado, así que debe fallar/reportarlo, no
    inventar un resultado."""
    rng = np.random.default_rng(7)
    d_diag = rng.uniform(0.5, 2.0, size=d)
    d_diag[5] = 1e-16  # rompe la invertibilidad de D
    U = rng.normal(size=(d, r))

    sigma_inv_wb, ok = woodbury_inv(d_diag, U)
    print(f"[Caso 2] D[5]={d_diag[5]:.1e} -> Woodbury reporta ok={ok}")
    assert not ok, "Se espera que la ruta Woodbury falle explícitamente"

    # La matriz Sigma completa puede o no ser invertible; lo chequeamos aparte,
    # para dejar claro que la falla es de la RUTA (D^{-1} no definida),
    # no necesariamente de Sigma en su conjunto.
    Sigma = np.diag(d_diag) + U @ U.T
    cond = np.linalg.cond(Sigma)
    print(f"[Caso 2] Número de condición de Sigma completa: {cond:.3e}")
    print("[Caso 2] Conclusión: Woodbury con este split (D, U) no es aplicable "
          "cuando D no es invertible por sí sola, aunque Sigma global lo sea; "
          "es una limitación documentada de la ruta de cómputo, no un bug oculto.\n")


if __name__ == "__main__":
    check_case_1()
    benchmark_scaling()
    check_case_2_failure()
