"""
Tarea 24 - LoRA especializado con inicializacion geometrica (rotor de Clifford / Lie so(n))
en lugar del producto matricial ordinario.

Comparacion:
  - LoRA estandar:   W' = W0 + (alpha/r) * B @ A
  - LoRA geometrico: W' = R @ W0,  R = expm(Theta), Theta = U V^T - V U^T  (antisimetrica, bajo rango)

Metricas:
  - Error de ortogonalidad: || R^T R - I ||_F
  - Norma espectral (mayor valor singular) de la actualizacion total
  - Determinante de R (debe ser 1 para un rotor puro)

No se ejecuta como parte de la entrega formal (ver skill). Correr manualmente:
  pip install numpy scipy
  python tarea_24_lora_clifford.py
"""

import numpy as np
from scipy.linalg import expm

RNG = np.random.default_rng(42)


def standard_lora_update(n: int, r: int, sigma: float):
    """LoRA clasico: B @ A, con A ~ N(0, sigma^2), B inicializado no-cero para
    poder medir sus propiedades geometricas (en la practica B arranca en 0,
    aca lo perturbamos levemente para poder comparar normas en igualdad de
    condiciones con el rotor, que tambien parte de una perturbacion chica)."""
    A = RNG.normal(0, sigma, size=(r, n))
    B = RNG.normal(0, sigma, size=(n, r))
    delta = B @ A
    return delta


def geometric_rotor_update(n: int, r: int, sigma: float):
    """LoRA geometrico: R = expm(Theta), Theta antisimetrica de bajo rango."""
    U = RNG.normal(0, sigma, size=(n, r))
    V = RNG.normal(0, sigma, size=(n, r))
    Theta = U @ V.T - V @ U.T
    R = expm(Theta)
    return R, Theta


def orthogonality_error(M: np.ndarray) -> float:
    n = M.shape[0]
    return float(np.linalg.norm(M.T @ M - np.eye(n), ord="fro"))


def spectral_norm(M: np.ndarray) -> float:
    return float(np.linalg.svd(M, compute_uv=False)[0])


def run_comparison():
    n = 64
    ranks = [2, 4, 8, 16]
    sigmas = [0.01, 0.1, 0.5]

    header = (
        f"{'rank':>5} {'sigma':>7} | "
        f"{'ortho_err_std':>14} {'ortho_err_rotor':>16} | "
        f"{'smax_std':>10} {'smax_rotor':>11} | {'det_rotor':>12}"
    )
    print(header)
    print("-" * len(header))

    for r in ranks:
        for sigma in sigmas:
            delta_std = standard_lora_update(n, r, sigma)
            W_std = np.eye(n) + delta_std  # actualizacion total "cerca de identidad"

            R, _theta = geometric_rotor_update(n, r, sigma)

            ortho_err_std = orthogonality_error(W_std)
            ortho_err_rotor = orthogonality_error(R)

            smax_std = spectral_norm(delta_std)
            smax_rotor = spectral_norm(R)

            det_rotor = float(np.linalg.det(R))

            print(
                f"{r:>5} {sigma:>7.2f} | "
                f"{ortho_err_std:>14.6e} {ortho_err_rotor:>16.6e} | "
                f"{smax_std:>10.4f} {smax_rotor:>11.4f} | {det_rotor:>12.8f}"
    )

    print()
    print("Lectura esperada:")
    print(" - ortho_err_rotor y |det_rotor - 1| deberian quedar en el orden de")
    print("   1e-14 a 1e-12 (error numerico puro), independientemente de sigma y rank.")
    print(" - ortho_err_std debería crecer con sigma y con rank, porque BA no tiene")
    print("   ninguna restriccion algebraica de ortogonalidad.")
    print(" - smax_rotor deberia mantenerse en 1.0 (isometria), smax_std no.")


def _sanity_check():
    """Chequeo minimo de sintaxis/ejecucion sobre una instancia chica,
    NO es la comprobacion final (eso lo hace el alumno)."""
    n, r, sigma = 8, 2, 0.1
    R, _ = geometric_rotor_update(n, r, sigma)
    assert R.shape == (n, n)
    delta = standard_lora_update(n, r, sigma)
    assert delta.shape == (n, n)


if __name__ == "__main__":
    _sanity_check()
    run_comparison()
