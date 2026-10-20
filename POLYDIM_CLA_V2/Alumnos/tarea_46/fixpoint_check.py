"""
fixpoint_check.py

Verificacion empirica independiente (Python/numpy) del operador FIXPOINT
implementado en fixpoint_vm.rs (Tarea 46, POLYDIM).

Modelo:
    T(x) = tanh(W x + b)
    Si sigma_max(W) = ||W||_2 <= lambda < 1  =>  T es lambda-Lipschitz
    (porque tanh es 1-Lipschitz), y por el Teorema del Punto Fijo de Banach
    la iteracion x_{k+1} = T(x_k) converge geometricamente a un unico x*:
        ||x_k - x*|| <= lambda^k ||x_0 - x*||

Este script:
  1. Genera una W aleatoria y mide su norma espectral exacta (numpy.linalg.svd),
     comparando contra la estimacion por power iteration (la misma tecnica
     usada en el codigo Rust).
  2. Aplica spectral normalization para forzar sigma_max(W) <= lambda_target.
  3. Itera T hasta convergencia y mide el error ||x_{k+1}-x_k||_2 en cada paso.
  4. Ajusta una recta a log(error) vs. iteracion para estimar la tasa de
     decaimiento empirica y la compara con lambda_target (deberia ser <= ).

Dependencias: numpy (pip install numpy --break-system-packages)
Uso:
    python3 fixpoint_check.py
"""

import numpy as np


def spectral_norm_power_iteration(W: np.ndarray, iters: int = 100, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    u = rng.standard_normal(n)
    u /= np.linalg.norm(u)
    for _ in range(iters):
        v = W @ u
        w = W.T @ v
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-14:
            break
        u = w / norm_w
    return float(np.linalg.norm(W @ u))


def enforce_spectral_bound(W: np.ndarray, lambda_target: float, iters: int = 100):
    sigma_before = spectral_norm_power_iteration(W, iters, seed=42)
    W_out = W.copy()
    if sigma_before > lambda_target:
        W_out *= lambda_target / sigma_before
    sigma_after = spectral_norm_power_iteration(W_out, iters, seed=43)
    return W_out, sigma_before, sigma_after


def fixpoint_iterate(W, b, x0, tol=1e-12, max_iter=500):
    x = x0.copy()
    errors = []
    converged = False
    for k in range(max_iter):
        x_next = np.tanh(W @ x + b)
        err = np.linalg.norm(x_next - x)
        errors.append(err)
        x = x_next
        if err < tol:
            converged = True
            break
    return x, errors, converged


def main():
    n = 4
    rng = np.random.default_rng(7)
    W = rng.uniform(-1, 1, size=(n, n))
    b = np.array([0.05, -0.02, 0.01, 0.0])
    x0 = np.ones(n)

    lambda_target = 0.9

    # --- 1. Comparar power iteration vs SVD exacto ---
    sigma_exact = float(np.linalg.svd(W, compute_uv=False)[0])
    sigma_power = spectral_norm_power_iteration(W, iters=100, seed=42)
    print(f"sigma_max exacto (SVD)        : {sigma_exact:.6f}")
    print(f"sigma_max estimado (power it.): {sigma_power:.6f}")
    print(f"error relativo                : {abs(sigma_exact - sigma_power) / sigma_exact:.2e}")

    # --- 2. Normalizacion espectral ---
    W_norm, sigma_before, sigma_after = enforce_spectral_bound(W, lambda_target)
    print(f"\nsigma_max antes de normalizar : {sigma_before:.6f}")
    print(f"sigma_max despues de normalizar: {sigma_after:.6f} (objetivo lambda={lambda_target})")

    # --- 3. Iterar FIXPOINT y medir convergencia ---
    x_star, errors, converged = fixpoint_iterate(W_norm, b, x0)
    print(f"\nconvergio: {converged}")
    print(f"iteraciones: {len(errors)}")
    print(f"x*: {x_star}")

    # --- 4. Tasa de decaimiento empirica (ajuste log-lineal) ---
    errors = np.array(errors)
    errors = errors[errors > 0]  # evitar log(0)
    k = np.arange(len(errors))
    # log(err_k) ~ log(err_0) + k * log(rate)
    slope, intercept = np.polyfit(k, np.log(errors), 1)
    rate_empirica = float(np.exp(slope))

    print(f"\ntasa de decaimiento empirica (ajuste log-lineal): {rate_empirica:.6f}")
    print(f"cota teorica lambda_target                      : {lambda_target:.6f}")
    if rate_empirica <= lambda_target + 1e-3:
        print("=> OK: la tasa observada es consistente con la cota de Banach (rate <= lambda).")
    else:
        print("=> ATENCION: la tasa observada supera la cota teorica; revisar W, b o tol/max_iter.")

    print("\nPrimeros 10 errores ||x_(k+1)-x_k||_2:")
    for i, e in enumerate(errors[:10]):
        print(f"  k={i:2d}  err={e:.6e}")


if __name__ == "__main__":
    main()
