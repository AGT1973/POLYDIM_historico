"""
Tarea 11 - Identidad de Woodbury para inversion de matrices de covarianza
de proyecciones de bajo rango.

Sigma = D + U C U^T   con:
  D  : (n x n) diagonal positiva
  U  : (n x k) matriz de proyeccion de bajo rango (k << n)
  C  : (k x k) invertible

Woodbury: Sigma^-1 = D^-1 - D^-1 U (C^-1 + U^T D^-1 U)^-1 U^T D^-1
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def construir_covarianza_bajo_rango(n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.5, 2.0, size=n)
    D = np.diag(d)
    U = rng.normal(size=(n, k))
    A = rng.normal(size=(k, k))
    C = A @ A.T + k * np.eye(k)
    Sigma = D + U @ C @ U.T
    return D, U, C, Sigma


def inversa_directa(Sigma: np.ndarray) -> np.ndarray:
    return np.linalg.inv(Sigma)


def inversa_woodbury(D: np.ndarray, U: np.ndarray, C: np.ndarray, verbose: bool = True):
    grafo = {}

    D_inv = np.diag(1.0 / np.diag(D))
    grafo["01_D_inv"] = D_inv

    C_inv = np.linalg.inv(C)
    grafo["02_C_inv"] = C_inv

    Ut_Dinv = U.T @ D_inv
    grafo["03_Ut_Dinv"] = Ut_Dinv

    M = C_inv + Ut_Dinv @ U
    grafo["04_M_igual_Cinv_mas_UtDinvU"] = M

    M_inv = np.linalg.inv(M)
    grafo["05_M_inv"] = M_inv

    Dinv_U = D_inv @ U
    grafo["06_Dinv_U"] = Dinv_U

    correccion = Dinv_U @ M_inv @ Ut_Dinv
    grafo["07_correccion"] = correccion

    Sigma_inv = D_inv - correccion
    grafo["08_Sigma_inv"] = Sigma_inv

    if verbose:
        for nombre, tensor in grafo.items():
            print(f"--- {nombre} | shape={tensor.shape} ---")
            print(tensor)
            print()

    return Sigma_inv, grafo


def validar(Sigma, Sigma_inv_directa, Sigma_inv_woodbury):
    n = Sigma.shape[0]
    I = np.eye(n)

    error_relativo = np.linalg.norm(Sigma_inv_directa - Sigma_inv_woodbury, ord="fro") / \
        np.linalg.norm(Sigma_inv_directa, ord="fro")

    residual_directo = np.linalg.norm(Sigma @ Sigma_inv_directa - I, ord="fro")
    residual_woodbury = np.linalg.norm(Sigma @ Sigma_inv_woodbury - I, ord="fro")

    return {
        "error_relativo_frobenius_directo_vs_woodbury": float(error_relativo),
        "residual_directo": float(residual_directo),
        "residual_woodbury": float(residual_woodbury),
    }


if __name__ == "__main__":
    n, k = 6, 2
    D, U, C, Sigma = construir_covarianza_bajo_rango(n, k, seed=42)

    print("========== Matriz Sigma (n x n) ==========")
    print(Sigma)
    print()

    Sigma_inv_directa = inversa_directa(Sigma)

    print("========== Cadena Woodbury (volcado de grafo intermedio) ==========")
    Sigma_inv_woodbury, grafo = inversa_woodbury(D, U, C, verbose=True)

    metricas = validar(Sigma, Sigma_inv_directa, Sigma_inv_woodbury)

    print("========== Validacion numerica ==========")
    for kk, vv in metricas.items():
        print(f"{kk}: {vv:.3e}")
