"""
Tarea 32 - POLYDIM V2
Mide la Divergencia KL (perdida semantica) inducida por imponer una
Spectral Lipschitz Penalty (mayor valor singular acotado) sobre un
operador FIXPOINT, comparando la convergencia del punto fijo con y sin
esa restriccion.

Dependencias: numpy (y opcionalmente scipy, no estrictamente necesario).
    pip install numpy --break-system-packages
"""

import numpy as np


def power_iteration_sigma_max(W, n_iter=100, seed=None):
    """Estima el mayor valor singular de W via iteracion de potencias
    sobre W^T W (O(d^2) por paso, en vez de O(d^3) de una SVD completa)."""
    rng = np.random.default_rng(seed)
    d = W.shape[1]
    v = rng.normal(size=d)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        u = W @ v
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            break
        u = u / u_norm
        v = W.T @ u
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            break
        v = v / v_norm
    sigma = np.linalg.norm(W @ v)
    return sigma


def spectral_normalize(W, lam, n_iter=100, seed=None):
    """Reescala W para que su mayor valor singular sea <= lam."""
    sigma = power_iteration_sigma_max(W, n_iter=n_iter, seed=seed)
    scale = lam / max(lam, sigma)
    return W * scale, sigma


def fixpoint_iterate(W, b, X0, K):
    """Itera x_{k+1} = tanh(W x_k + b) para un batch de puntos iniciales.
    Devuelve el estado final y la norma media por paso (para inspeccionar
    convergencia u oscilacion)."""
    X = X0.copy()
    norms_per_step = np.zeros(K)
    for k in range(K):
        X = np.tanh(X @ W.T + b)
        norms_per_step[k] = np.mean(np.linalg.norm(X, axis=1))
    return X, norms_per_step


def fit_gaussian(X):
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False)
    # Regularizacion minima para evitar singularidad numerica en cov.
    Sigma += 1e-6 * np.eye(Sigma.shape[0])
    return mu, Sigma


def kl_divergence_gaussians(mu_c, Sigma_c, mu_u, Sigma_u):
    """D_KL(N(mu_c,Sigma_c) || N(mu_u,Sigma_u)) en forma cerrada."""
    d = mu_c.shape[0]
    Sigma_u_inv = np.linalg.inv(Sigma_u)
    diff = mu_u - mu_c
    term_trace = np.trace(Sigma_u_inv @ Sigma_c)
    term_quad = diff.T @ Sigma_u_inv @ diff
    sign_u, logdet_u = np.linalg.slogdet(Sigma_u)
    sign_c, logdet_c = np.linalg.slogdet(Sigma_c)
    term_logdet = logdet_u - logdet_c
    kl = 0.5 * (term_trace + term_quad - d + term_logdet)
    return float(kl)


def run_experiment(d=8, N=2000, K=200, lam=0.9, seed=42):
    rng = np.random.default_rng(seed)

    # W deliberadamente inestable (sigma_max > 1) para no ocultar el caso
    # en que el operador SIN restringir puede no converger.
    W_raw = rng.normal(scale=1.0, size=(d, d))
    b = rng.normal(scale=0.1, size=d)

    sigma_before = power_iteration_sigma_max(W_raw, seed=seed)
    W_sn, sigma_check = spectral_normalize(W_raw, lam=lam, seed=seed)
    sigma_after = power_iteration_sigma_max(W_sn, seed=seed)

    X0 = rng.normal(size=(N, d))

    X_unc, norms_unc = fixpoint_iterate(W_raw, b, X0, K)
    X_con, norms_con = fixpoint_iterate(W_sn, b, X0, K)

    mu_u, Sigma_u = fit_gaussian(X_unc)
    mu_c, Sigma_c = fit_gaussian(X_con)

    kl = kl_divergence_gaussians(mu_c, Sigma_c, mu_u, Sigma_u)

    print("=== Tarea 32: KL inducida por Spectral Lipschitz Penalty en FIXPOINT ===")
    print(f"Dimension d={d}, N puntos={N}, pasos K={K}, lambda objetivo={lam}")
    print(f"sigma_max(W) sin restringir : {sigma_before:.4f}")
    print(f"sigma_max(W_sn) restringido : {sigma_after:.4f} (objetivo <= {lam})")
    print()
    print("Norma media ||x_k|| por paso (sin restringir), primeros y ultimos 5 pasos:")
    print("  inicio:", np.round(norms_unc[:5], 4))
    print("  final :", np.round(norms_unc[-5:], 4))
    oscillating = np.std(norms_unc[-20:]) > 1e-3
    print(f"  -> {'POSIBLE OSCILACION / no estabilizo' if oscillating else 'se estabilizo'} "
          f"(std ultimos 20 pasos = {np.std(norms_unc[-20:]):.6f})")
    print()
    print("Norma media ||x_k|| por paso (restringido), primeros y ultimos 5 pasos:")
    print("  inicio:", np.round(norms_con[:5], 4))
    print("  final :", np.round(norms_con[-5:], 4))
    stable = np.std(norms_con[-20:]) < 1e-3
    print(f"  -> {'se estabilizo (consistente con Banach)' if stable else 'NO se estabilizo (inesperado)'} "
          f"(std ultimos 20 pasos = {np.std(norms_con[-20:]):.6f})")
    print()
    print(f"D_KL(P_restringido || P_sin_restringir) = {kl:.6f}")
    print()
    print("Interpretacion: este numero es la perdida semantica (en nats) que introduce")
    print("la Spectral Lipschitz Penalty necesaria para garantizar convergencia topologica")
    print("estable en FIXPOINT. No se asume de antemano si es 'chica' o 'grande':")
    print("el alumno debe evaluarlo corriendo el script con distintas semillas/dimensiones.")

    return {
        "sigma_before": sigma_before,
        "sigma_after": sigma_after,
        "kl_divergence": kl,
        "unconstrained_oscillating": bool(oscillating),
        "constrained_stable": bool(stable),
    }


if __name__ == "__main__":
    run_experiment(d=8, N=2000, K=200, lam=0.9, seed=42)
