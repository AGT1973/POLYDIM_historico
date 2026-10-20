"""
Tarea 38 - POLYDIM (banco de 200 tareas)
==========================================

Pregunta reconstruida:
Dada una secuencia de embeddings X (T x d), se aplica un average pooling
temporal con ventana w (operador lineal A), y luego la reconstrucción
lineal óptima en sentido de mínimos cuadrados B* = A^+ (pseudoinversa de
Moore-Penrose). Se mide, usando la librería POT (Python Optimal Transport)
y el algoritmo de Sinkhorn, la divergencia de Sinkhorn (Wasserstein
entrópica de-biased) entre la distribución empírica de los vectores
originales y la de los vectores reconstruidos, en función de:
  (a) el tamaño de ventana w (cuánto se comprime temporalmente), y
  (b) el parámetro de regularización entrópica epsilon del propio Sinkhorn.

Requisitos para correrlo (el alumno lo ejecuta en su propia cuenta):
    pip install numpy pot

Ejecución:
    python tarea_38.py
"""

import numpy as np

try:
    import ot  # Python Optimal Transport (POT)
except ImportError as e:
    raise SystemExit(
        "Falta la librería POT. Instalala con: pip install pot\n"
        f"Error original: {e}"
    )


# ---------------------------------------------------------------------------
# 1. Operador de average pooling temporal como matriz lineal A (T' x T)
# ---------------------------------------------------------------------------
def pooling_matrix(T: int, w: int) -> np.ndarray:
    """
    Construye el operador de average pooling no solapado con ventana w.
    Requiere T % w == 0. Cada fila tiene w entradas iguales a 1/w.
    """
    if T % w != 0:
        raise ValueError(f"T={T} debe ser múltiplo de w={w} (pooling no solapado).")
    T_prime = T // w
    A = np.zeros((T_prime, T))
    for i in range(T_prime):
        A[i, i * w:(i + 1) * w] = 1.0 / w
    return A


# ---------------------------------------------------------------------------
# 2. Reconstrucción lineal óptima (mínimos cuadrados) = pseudoinversa de A
# ---------------------------------------------------------------------------
def optimal_linear_reconstruction(A: np.ndarray) -> np.ndarray:
    """
    B* = argmin_B E[|| B A X - X ||_F^2] = A^+ (pseudoinversa de Moore-Penrose).

    Para este A (filas ortogonales, no solapadas, cada una de norma 1/w):
        A A^T = (1/w) * I_{T'}
    por lo tanto A^+ = A^T (A A^T)^{-1} = w * A^T.

    Se calcula ambas formas (cerrada y numérica) y se verifica que coinciden,
    como chequeo de consistencia del modelo.
    """
    T_prime, T = A.shape
    w = T // T_prime

    B_closed = w * A.T                      # fórmula cerrada derivada a mano
    B_numeric = np.linalg.pinv(A)           # pseudoinversa numérica (SVD)

    if not np.allclose(B_closed, B_numeric, atol=1e-10):
        raise AssertionError(
            "La fórmula cerrada B*=wA^T no coincide con la pseudoinversa "
            "numérica; revisar la derivación."
        )
    return B_closed


# ---------------------------------------------------------------------------
# 3. Divergencia de Sinkhorn (de-biased) entre dos nubes de puntos, vía POT
# ---------------------------------------------------------------------------
def sinkhorn_divergence(X: np.ndarray, Y: np.ndarray, epsilon: float) -> float:
    """
    Divergencia de Sinkhorn de-biased:
        S_eps(mu, nu) = OT_eps(mu, nu) - 0.5*OT_eps(mu, mu) - 0.5*OT_eps(nu, nu)

    donde OT_eps es el costo de transporte óptimo entrópico (algoritmo de
    Sinkhorn) con costo cuadrático ||x - y||^2 y pesos uniformes.
    Restar los términos "self" (mu,mu) y (nu,nu) corrige el sesgo que
    introduce la regularización entrópica (S_eps(mu,mu) != 0 en general).
    """
    n, m = X.shape[0], Y.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m

    Mxy = ot.dist(X, Y, metric="sqeuclidean")
    Mxx = ot.dist(X, X, metric="sqeuclidean")
    Myy = ot.dist(Y, Y, metric="sqeuclidean")

    ot_xy = ot.sinkhorn2(a, b, Mxy, reg=epsilon)
    ot_xx = ot.sinkhorn2(a, a, Mxx, reg=epsilon)
    ot_yy = ot.sinkhorn2(b, b, Myy, reg=epsilon)

    return float(ot_xy - 0.5 * ot_xx - 0.5 * ot_yy)


# ---------------------------------------------------------------------------
# 4. Experimento: barrido en w (compresión) y epsilon (regularización)
# ---------------------------------------------------------------------------
def run_experiment(T=64, d=8, seed=0):
    rng = np.random.default_rng(seed)

    # Secuencia sintética de embeddings con estructura temporal
    # (un paseo aleatorio suavizado, más realista que ruido i.i.d. puro)
    steps = rng.normal(scale=0.5, size=(T, d))
    X = np.cumsum(steps, axis=0)
    X += rng.normal(scale=0.1, size=(T, d))  # ruido de observación

    window_sizes = [1, 2, 4, 8, 16]
    epsilons = [0.5, 1.0, 5.0]

    print(f"{'w':>4} | {'eps':>6} | {'Sinkhorn div.':>14} | {'MSE reconstr.':>14}")
    print("-" * 50)

    results = []
    for w in window_sizes:
        A = pooling_matrix(T, w)
        B = optimal_linear_reconstruction(A)

        Y = A @ X          # pooled
        X_hat = B @ Y       # reconstruido (bloques rellenados con el promedio)

        mse = float(np.mean((X_hat - X) ** 2))

        for eps in epsilons:
            div = sinkhorn_divergence(X, X_hat, epsilon=eps)
            results.append((w, eps, div, mse))
            print(f"{w:>4} | {eps:>6.2f} | {div:>14.6f} | {mse:>14.6f}")

    return results


if __name__ == "__main__":
    run_experiment()
