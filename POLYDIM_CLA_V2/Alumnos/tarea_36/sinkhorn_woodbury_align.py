"""
Tarea 36 - Alineacion estadistica via Sinkhorn (transporte optimo entropico)
usando un costo de Mahalanobis cuya matriz de precision (covarianza inversa)
se calcula de forma eficiente con la identidad de Woodbury para covarianzas
de bajo rango + diagonal: Sigma = D + U U^T.

Motivacion (ver tarea_36.md para el desarrollo completo): en vez de invertir
Sigma (d x d) directamente en O(d^3), se usa Woodbury:

    Sigma^{-1} = D^{-1} - D^{-1} U (I_r + U^T D^{-1} U)^{-1} U^T D^{-1}

para aplicar Sigma^{-1} a un lote de vectores en O(d*r + r^3), y con eso se
arma la matriz de costos de Mahalanobis entre dos nubes de puntos, que luego
se usa como costo del transporte optimo entropico (algoritmo de Sinkhorn).
"""

import numpy as np


# --------------------------------------------------------------------------
# 1) Estructura de covarianza de bajo rango + diagonal, e inversa via Woodbury
# --------------------------------------------------------------------------
def woodbury_apply_inverse(D_diag: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Calcula Sigma^{-1} @ V, donde Sigma = diag(D_diag) + U @ U.T,
    sin formar nunca Sigma^{-1} (d x d) explicitamente.

    D_diag: (d,)      -- diagonal positiva
    U:      (d, r)    -- factor de bajo rango, r << d
    V:      (d, k)    -- lote de k vectores columna a los que aplicar Sigma^{-1}

    Retorna: (d, k)
    """
    d, r = U.shape
    Dinv = 1.0 / D_diag                      # (d,)
    Dinv_V = Dinv[:, None] * V               # (d, k)   D^{-1} V
    Dinv_U = Dinv[:, None] * U               # (d, r)   D^{-1} U

    # Sistema pequeno r x r: (I_r + U^T D^{-1} U)
    M = np.eye(r) + U.T @ Dinv_U             # (r, r)
    rhs = U.T @ Dinv_V                       # (r, k)
    correction_small = np.linalg.solve(M, rhs)   # (r, k) -- resolver, no invertir
    correction = Dinv_U @ correction_small   # (d, k)

    return Dinv_V - correction               # (d, k)


def direct_inverse_apply(D_diag: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Referencia lenta (O(d^3)): forma Sigma completa y la invierte directo."""
    Sigma = np.diag(D_diag) + U @ U.T
    Sigma_inv = np.linalg.inv(Sigma)
    return Sigma_inv @ V


# --------------------------------------------------------------------------
# 2) Matriz de costos de Mahalanobis entre dos nubes de puntos, via Woodbury
# --------------------------------------------------------------------------
def mahalanobis_cost_matrix(X: np.ndarray, Y: np.ndarray, D_diag: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    C[i, j] = (x_i - y_j)^T Sigma^{-1} (x_i - y_j),  Sigma = diag(D_diag) + U U^T

    X: (n, d), Y: (m, d). Se explota bilinealidad para evitar O(n*m*d) llamadas
    individuales a woodbury_apply_inverse:

        C = diag(X Sigma^{-1} X^T)[:, None] + diag(Y Sigma^{-1} Y^T)[None, :]
            - 2 X Sigma^{-1} Y^T
    """
    SXt = woodbury_apply_inverse(D_diag, U, X.T)   # Sigma^{-1} X^T, (d, n)
    SYt = woodbury_apply_inverse(D_diag, U, Y.T)   # Sigma^{-1} Y^T, (d, m)

    xx = np.einsum("dn,dn->n", X.T, SXt)           # (n,)  = diag(X Sigma^{-1} X^T)
    yy = np.einsum("dm,dm->m", Y.T, SYt)           # (m,)
    xy = X @ SYt                                   # (n, m) = X Sigma^{-1} Y^T

    C = xx[:, None] + yy[None, :] - 2.0 * xy
    return np.maximum(C, 0.0)  # recorte numerico de negativos espurios (~1e-12)


# --------------------------------------------------------------------------
# 3) Sinkhorn: transporte optimo entropico entre dos distribuciones empiricas
# --------------------------------------------------------------------------
def sinkhorn(C: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 0.1,
             n_iter: int = 500, tol: float = 1e-9) -> np.ndarray:
    """
    Resuelve min_{P in U(a,b)} <P, C> - eps * H(P) via el algoritmo de
    Sinkhorn-Knopp (escalado matricial iterativo). Devuelve el plan P (n, m).
    """
    K = np.exp(-C / eps)
    n, m = C.shape
    u = np.ones(n) / n
    v = np.ones(m) / m
    for _ in range(n_iter):
        u_prev = u
        Kv = K @ v
        u = a / np.clip(Kv, 1e-300, None)
        Ktu = K.T @ u
        v = b / np.clip(Ktu, 1e-300, None)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    P = (u[:, None] * K) * v[None, :]
    return P


# --------------------------------------------------------------------------
# 4) Demostracion + verificaciones internas
# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    d, r = 50, 5      # dimension del embedding, rango de la correccion
    n, m = 40, 35      # tamanos de las dos nubes de puntos

    D_diag = rng.uniform(0.5, 2.0, size=d)
    U = rng.normal(scale=0.3, size=(d, r))

    # --- Verificacion 1: Woodbury vs inversa directa (deben coincidir) ---
    V_test = rng.normal(size=(d, 7))
    fast = woodbury_apply_inverse(D_diag, U, V_test)
    slow = direct_inverse_apply(D_diag, U, V_test)
    err = np.max(np.abs(fast - slow))
    assert err < 1e-8, f"Woodbury difiere de la inversa directa: err={err}"
    print(f"[OK] Woodbury coincide con la inversion directa (error maximo = {err:.2e})")

    # --- Dos nubes de puntos "desplazadas" que hay que alinear ---
    mean_shift = rng.normal(scale=1.5, size=d)
    X = rng.normal(size=(n, d)) @ np.linalg.cholesky(np.diag(D_diag) + U @ U.T).T
    Y = mean_shift + rng.normal(size=(m, d)) @ np.linalg.cholesky(np.diag(D_diag) + U @ U.T).T

    C = mahalanobis_cost_matrix(X, Y, D_diag, U)

    # --- Verificacion 2: costo de Mahalanobis via Woodbury coincide con el directo ---
    C_direct = np.array([[direct_inverse_apply(D_diag, U, (X[i] - Y[j])[:, None]).T @ (X[i] - Y[j])
                           for j in range(m)] for i in range(n)]).reshape(n, m)
    err_C = np.max(np.abs(C - C_direct))
    assert err_C < 1e-6, f"Costo via Woodbury difiere del directo: err={err_C}"
    print(f"[OK] Matriz de costos via Woodbury coincide con la version directa (error maximo = {err_C:.2e})")

    # --- Sinkhorn: alinear estadisticamente las dos nubes ---
    a = np.ones(n) / n
    b = np.ones(m) / m
    P = sinkhorn(C, a, b, eps=0.5, n_iter=1000)

    row_err = np.max(np.abs(P.sum(axis=1) - a))
    col_err = np.max(np.abs(P.sum(axis=0) - b))
    print(f"[OK] Plan de transporte P: {P.shape}, error marginal filas = {row_err:.2e}, columnas = {col_err:.2e}")
    print(f"     Costo de transporte <P, C> = {np.sum(P * C):.4f}")

    assert row_err < 1e-3 and col_err < 1e-3, "Las marginales del plan de Sinkhorn no convergieron"
    print("\nTodas las verificaciones pasaron.")


if __name__ == "__main__":
    main()
