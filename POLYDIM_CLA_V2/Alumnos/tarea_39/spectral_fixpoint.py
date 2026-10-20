"""
Tarea 39 - Enjambre POLYDIM
============================
Nucleo matematico: Teorema del Punto Fijo de Banach aplicado a un operador
de actualizacion tipo GNN implicita, usando el mayor valor singular del
Jacobiano (estimado por iteracion de potencias) como certificado de
contraccion para el bucle FIXPOINT.

Este script es 100% funcional en CPU con NumPy. No requiere GPU ni Triton.
El alumno debe ejecutarlo el mismo para comprobar los resultados con sus
propios ojos (ver instrucciones al final del .md).

Dependencias: numpy  (pip install numpy)
Ejecucion:    python spectral_fixpoint.py
"""

import numpy as np

rng = np.random.default_rng(seed=39)


# ---------------------------------------------------------------------------
# 1. Iteracion de potencias para estimar el mayor valor singular de una matriz
# ---------------------------------------------------------------------------
def power_iteration_spectral_norm(W: np.ndarray, n_iters: int = 50, tol: float = 1e-9):
    """
    Estima sigma_max(W) (el mayor valor singular) sin calcular la SVD completa.

    Metodo: iteracion de potencias sobre W^T W.
        v_{k+1} = (W^T W) v_k / || (W^T W) v_k ||
        sigma_max^2  ~=  v_k^T (W^T W) v_k      (cociente de Rayleigh)

    Complejidad por iteracion: O(n^2) (dos productos matriz-vector),
    frente a O(n^3) de una SVD completa. Esta es exactamente la razon
    por la que en produccion (POLYDIM, Miyato et al. 2018 "Spectral
    Normalization for GANs") esta operacion se implementa como un
    kernel fusionado (matmul + normalizacion) en Triton/CUDA: dos
    mat-vec por iteracion, sin materializar W^T W ni la SVD completa.
    """
    n = W.shape[1]
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)

    sigma_prev = 0.0
    for _ in range(n_iters):
        Wv = W @ v
        WtWv = W.T @ Wv
        norm = np.linalg.norm(WtWv)
        if norm < 1e-15:
            break
        v = WtWv / norm
        sigma = np.linalg.norm(W @ v)
        if abs(sigma - sigma_prev) < tol:
            sigma_prev = sigma
            break
        sigma_prev = sigma

    return sigma_prev


def reference_spectral_norm(W: np.ndarray) -> float:
    """Valor de referencia (ground truth) via SVD completa de LAPACK."""
    return np.linalg.svd(W, compute_uv=False)[0]


# ---------------------------------------------------------------------------
# 2. Operador de actualizacion tipo GNN implicita: T(x) = tanh(W x + b)
#    (una capa de propagacion; en una IGNN real W absorbe tambien la
#    matriz de adyacencia normalizada del grafo, pero para aislar el
#    fenomeno matematico -contraccion vs. no contraccion- alcanza con
#    una matriz densa generica).
# ---------------------------------------------------------------------------
def make_operator(W: np.ndarray, b: np.ndarray):
    def T(x: np.ndarray) -> np.ndarray:
        return np.tanh(W @ x + b)
    return T


def jacobian_operator_norm_bound(W: np.ndarray) -> float:
    """
    Cota superior teorica de la constante de Lipschitz de T(x)=tanh(Wx+b):

        ||J_T(x)||_2 = || diag(1 - tanh(Wx+b)^2) @ W ||_2
                     <= ||diag(1 - tanh(.)^2)||_2 * ||W||_2
                     <= 1 * ||W||_2        (porque |tanh'| <= 1 siempre)

    Es decir: sigma_max(W) es una cota GLOBAL (valida para todo x) de la
    constante de contraccion de T. Esto es lo que permite aplicar Banach
    de forma global y no solo puntual.
    """
    return reference_spectral_norm(W)


# ---------------------------------------------------------------------------
# 3. Normalizacion espectral: reescalar W para forzar sigma_max(W) = lambda_target < 1
# ---------------------------------------------------------------------------
def spectral_normalize(W: np.ndarray, lambda_target: float, sigma_estimate: float) -> np.ndarray:
    return W * (lambda_target / sigma_estimate)


# ---------------------------------------------------------------------------
# 4. Verificacion empirica del Teorema de Punto Fijo de Banach
# ---------------------------------------------------------------------------
def run_fixpoint_experiment(W: np.ndarray, b: np.ndarray, n_steps: int = 60,
                             n_inits: int = 4, label: str = ""):
    T = make_operator(W, b)
    n = W.shape[0]

    trajectories = []
    for i in range(n_inits):
        x = rng.standard_normal(n) * 3.0  # puntos de partida bien separados
        traj = [x.copy()]
        for _ in range(n_steps):
            x = T(x)
            traj.append(x.copy())
        trajectories.append(np.array(traj))

    # Distancia entre iteraciones sucesivas (para medir tasa de contraccion empirica)
    steps_norms = np.linalg.norm(trajectories[0][1:] - trajectories[0][:-1], axis=1)

    # Distancia final entre las trayectorias que partieron de puntos distintos
    # (si converge al MISMO punto fijo, esta distancia debe ir a 0 -> unicidad)
    final_points = np.array([traj[-1] for traj in trajectories])
    max_pairwise_final_dist = 0.0
    for i in range(n_inits):
        for j in range(i + 1, n_inits):
            d = np.linalg.norm(final_points[i] - final_points[j])
            max_pairwise_final_dist = max(max_pairwise_final_dist, d)

    print(f"--- Experimento: {label} ---")
    print(f"sigma_max(W) real (SVD)      : {reference_spectral_norm(W):.4f}")
    print(f"||x_1 - x_0||                : {steps_norms[0]:.6f}")
    print(f"||x_n - x_(n-1)|| (ultimo)   : {steps_norms[-1]:.6e}")
    if steps_norms[0] > 1e-12:
        tasa_empirica = (steps_norms[-1] / steps_norms[0]) ** (1.0 / (len(steps_norms) - 1))
        print(f"tasa de contraccion empirica : {tasa_empirica:.4f}")
    print(f"distancia final entre las {n_inits} trayectorias "
          f"(deberia -> 0 si hay punto fijo UNICO): {max_pairwise_final_dist:.6e}")
    print()
    return steps_norms, max_pairwise_final_dist


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n = 12
    lambda_target = 0.9  # constante de contraccion deseada, debe ser < 1

    # Matriz base aleatoria (sin ninguna restriccion)
    W_raw = rng.standard_normal((n, n)) * 0.9
    b = rng.standard_normal(n) * 0.1

    # --- Paso A: estimar sigma_max con iteracion de potencias y validar contra SVD ---
    sigma_pi = power_iteration_spectral_norm(W_raw, n_iters=100)
    sigma_svd = reference_spectral_norm(W_raw)
    error_relativo = abs(sigma_pi - sigma_svd) / sigma_svd
    print("=== Paso A: validacion del estimador (iteracion de potencias vs SVD) ===")
    print(f"sigma_max estimado (power iteration): {sigma_pi:.6f}")
    print(f"sigma_max exacto (SVD, LAPACK)       : {sigma_svd:.6f}")
    print(f"error relativo                       : {error_relativo:.2e}")
    assert error_relativo < 1e-3, "La iteracion de potencias no convergio lo suficiente."
    print("OK: el estimador reproduce el valor exacto dentro de tolerancia.\n")

    # --- Paso B: caso SIN normalizar (sigma_max puede ser > 1: no hay garantia) ---
    run_fixpoint_experiment(W_raw, b, label="W SIN normalizar (sin garantia de Banach)")

    # --- Paso C: caso CON normalizacion espectral (sigma_max = lambda_target < 1) ---
    W_normalized = spectral_normalize(W_raw, lambda_target, sigma_pi)
    run_fixpoint_experiment(W_normalized, b,
                             label=f"W normalizada espectralmente (lambda_target = {lambda_target})")

    # --- Paso D: caso deliberadamente expansivo (sigma_max >> 1) para mostrar
    #     que la ausencia de la cota rompe la convergencia / unicidad ---
    W_expansive = spectral_normalize(W_raw, 2.5, sigma_pi)
    run_fixpoint_experiment(W_expansive, b,
                             label="W expansiva (sigma_max = 2.5, viola la hipotesis de Banach)")
