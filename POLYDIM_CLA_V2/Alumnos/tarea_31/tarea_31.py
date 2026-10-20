"""
Tarea 31 - LoRA vía Producto Geométrico de un Álgebra de Clifford truncada
============================================================================

Ver tarea_31.md para el desarrollo matemático completo. Este script:

1. Construye un "rotor-LoRA": en vez de una actualización aditiva de bajo
   rango DeltaW = B @ A (LoRA estándar), usa un generador bivectorial
   truncado (pares de índices disjuntos e_i ^ e_j) y su exponencial
   (Producto Geométrico / mapa exponencial de Clifford) como una rotación
   ortogonal R, aplicada multiplicativamente.
2. Verifica de forma EXACTA (no estadística) que R preserva la norma L2,
   mientras que la actualización aditiva estándar generalmente no.
3. Documenta honestamente el límite de expresividad: con generadores
   bivectoriales fijos en los ejes coordenados, el rotor NO puede alcanzar
   una rotación genérica de SO(n) (solo un subtoro de dimensión r), y eso
   se demuestra empíricamente, no se oculta.

Requisitos: numpy, scipy (pip install numpy scipy --break-system-packages)
Ejecución:  python3 tarea_31.py
"""

import numpy as np
from scipy.linalg import expm


# ---------------------------------------------------------------------------
# 1. Rotor truncado (Producto Geométrico de Clifford, grado <= 2, generadores
#    bivectoriales en pares de ejes DISJUNTOS -> generadores que conmutan).
# ---------------------------------------------------------------------------

def bivector_generator(n: int, i: int, j: int) -> np.ndarray:
    """Generador infinitesimal K_ij = e_i ^ e_j representado como matriz
    antisimétrica elemental (base del álgebra de Lie so(n))."""
    K = np.zeros((n, n))
    K[i, j] = 1.0
    K[j, i] = -1.0
    return K


def build_rotor(n: int, pairs: list[tuple[int, int]], thetas: np.ndarray):
    """
    Construye el rotor R = exp(K), K = sum_k theta_k * (e_{i_k} ^ e_{j_k}).

    Si los pares (i_k, j_k) son disjuntos entre sí, los generadores K_k
    conmutan, y R es EXACTAMENTE el producto de r rotaciones planas
    independientes (rotaciones de Givens) en esos planos -- coincide con
    lo que en el marco POLYDIM se llama Rotor: R = e^{theta B / 2} salvo
    normalización del ángulo.
    """
    K = np.zeros((n, n))
    for (i, j), theta in zip(pairs, thetas):
        K += theta * bivector_generator(n, i, j)
    R = expm(K)
    return R, K


def check_orthogonality(R: np.ndarray, atol: float = 1e-9):
    """Un rotor genuino cumple R^T R = I de forma EXACTA (a precisión de
    punto flotante), porque proviene de exponenciar una matriz antisimétrica."""
    n = R.shape[0]
    err = np.linalg.norm(R.T @ R - np.eye(n))
    return err, err < atol


# ---------------------------------------------------------------------------
# 2. LoRA estándar (aditiva, bajo rango) para comparar.
# ---------------------------------------------------------------------------

def standard_lora_delta(n: int, r: int, rng: np.random.Generator, scale: float = 0.3):
    """DeltaW = B @ A, B in R^{n x r}, A in R^{r x n}: actualización aditiva
    de rango <= r. Parámetros entrenables: 2*n*r."""
    B = rng.normal(size=(n, r)) * scale
    A = rng.normal(size=(r, n)) * scale
    return B @ A


# ---------------------------------------------------------------------------
# 3. Experimento 1: preservación de norma (resultado EXACTO, no estadístico)
# ---------------------------------------------------------------------------

def experiment_norm_preservation(n: int, rng: np.random.Generator):
    print("=" * 70)
    print("EXPERIMENTO 1: ¿el rotor-LoRA preserva la norma? ¿y el LoRA estándar?")
    print("=" * 70)

    pairs = [(0, 1), (2, 3), (4, 5)]  # 3 pares disjuntos -> r = 3
    thetas = rng.uniform(-np.pi, np.pi, size=len(pairs))
    R, K = build_rotor(n, pairs, thetas)

    err_orth, is_orth = check_orthogonality(R)
    print(f"\n[Rotor-LoRA] ||R^T R - I|| = {err_orth:.3e}  -> ortogonal exacto: {is_orth}")

    x = rng.normal(size=(n,))
    Rx = R @ x
    print(f"[Rotor-LoRA] ||x||   = {np.linalg.norm(x):.6f}")
    print(f"[Rotor-LoRA] ||R x|| = {np.linalg.norm(Rx):.6f}  (deben coincidir a precisión de máquina)")

    # LoRA estándar: W fija + actualización aditiva de bajo rango.
    W = rng.normal(size=(n, n)) * 0.5
    deltaW = standard_lora_delta(n, r=len(pairs), rng=rng)
    y0 = W @ x
    y1 = (W + deltaW) @ x
    print(f"\n[LoRA estándar] ||W x||         = {np.linalg.norm(y0):.6f}")
    print(f"[LoRA estándar] ||(W+BA) x||    = {np.linalg.norm(y1):.6f}  (generalmente NO coincide)")

    params_rotor = len(pairs)
    params_lora = 2 * n * len(pairs)
    print(f"\nParámetros entrenables -> rotor-LoRA: {params_rotor}   |   LoRA estándar (r={len(pairs)}): {params_lora}")


# ---------------------------------------------------------------------------
# 4. Experimento 2: límite de expresividad (documentado, no ocultado)
# ---------------------------------------------------------------------------

def rotor_loss(n, pairs, thetas, R_target):
    R, _ = build_rotor(n, pairs, thetas)
    return np.linalg.norm(R - R_target, ord="fro") ** 2


def fit_rotor_to_target(n, pairs, R_target, iters=1500, lr=0.08, eps=1e-4):
    """Descenso de gradiente por diferencias finitas (simple y transparente,
    sin dependencias de autograd) sobre los angulos theta."""
    thetas = np.zeros(len(pairs))
    for _ in range(iters):
        loss0 = rotor_loss(n, pairs, thetas, R_target)
        grad = np.zeros_like(thetas)
        for k in range(len(thetas)):
            th_eps = thetas.copy()
            th_eps[k] += eps
            grad[k] = (rotor_loss(n, pairs, th_eps, R_target) - loss0) / eps
        thetas = thetas - lr * grad
    R_final, _ = build_rotor(n, pairs, thetas)
    residual = np.linalg.norm(R_final - R_target, ord="fro")
    return thetas, residual


def experiment_expressivity_limit(n: int, rng: np.random.Generator):
    print("\n" + "=" * 70)
    print("EXPERIMENTO 2: límite de expresividad del rotor de ejes fijos")
    print("=" * 70)

    pairs = [(0, 1), (2, 3), (4, 5)]  # subtoro de dimensión 3 dentro de SO(6) (dim 15)

    # Caso A: target GENÉRICO de SO(n) (generador antisimétrico denso al azar).
    K_full = rng.normal(size=(n, n))
    K_full = (K_full - K_full.T) / 2
    R_generic = expm(K_full)
    _, residual_generic = fit_rotor_to_target(n, pairs, R_generic)

    # Caso B: target CONSTRUIDO para vivir en el mismo subtoro (block-diagonal
    # en los mismos pares de ejes) -> debería ser alcanzable casi exactamente.
    thetas_true = rng.uniform(-np.pi, np.pi, size=len(pairs))
    R_aligned, _ = build_rotor(n, pairs, thetas_true)
    _, residual_aligned = fit_rotor_to_target(n, pairs, R_aligned)

    print(f"\nDim(SO({n})) = {n*(n-1)//2}   |   Dim(subtoro alcanzable) = {len(pairs)}")
    print(f"Residual al ajustar una rotación GENÉRICA de SO({n}):        {residual_generic:.4f}  (no debe ser ~0)")
    print(f"Residual al ajustar una rotación YA ALINEADA al subtoro:     {residual_aligned:.6f}  (debe ser ~0)")
    print("\nConclusión empírica: el rotor de ejes fijos NO es universal para SO(n);")
    print("solo reproduce exactamente rotaciones que ya viven en su subtoro de origen.")
    print("Esto se documenta como limitación real del método, no se disimula.")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 6
    experiment_norm_preservation(n, rng)
    experiment_expressivity_limit(n, rng)
