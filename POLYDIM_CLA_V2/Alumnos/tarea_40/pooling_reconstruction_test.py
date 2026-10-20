"""
POLYDIM - Tarea 40
Verificacion de la coherencia (idempotencia) de la operacion encadenada
Pooling promedio en tiempo -> Reconstruccion optima lineal (pseudoinversa),
y medicion del error de reconstruccion en funcion de la 'rugosidad' de la señal.

Rol: arquitecto de Sistemas Distribuidos.
"""
import numpy as np


def build_pooling_matrix(n, k):
    """P: R^n -> R^m, promedio en bloques de tamaño k (n = m*k)."""
    assert n % k == 0, "n debe ser multiplo de k"
    m = n // k
    P = np.zeros((m, n))
    for i in range(m):
        P[i, i * k:(i + 1) * k] = 1.0 / k
    return P


def optimal_linear_reconstruction(P):
    """R = P^+ (pseudoinversa de Moore-Penrose), reconstruccion lineal optima
    en sentido de minimos cuadrados / norma minima."""
    return np.linalg.pinv(P)


def graph_level_coherence_check(P, R, atol=1e-9):
    """Chequeo 'a nivel de volcado intermedio de grafo': verifica, sin usar
    ningun dato de entrada, que la composicion M = R @ P es un proyector
    idempotente (M @ M == M). Esto es una propiedad puramente estructural
    de las matrices (analoga a la coherencia estatica de Mac Lane sobre el
    grafo de computo, sin ejecutar la cadena sobre datos)."""
    M = R @ P
    diff = np.max(np.abs(M @ M - M))
    is_idempotent = diff < atol
    return M, diff, is_idempotent


def reconstruction_error(x, M):
    """Error real (dependiente de los datos) de la cadena Pool->Reconstruct."""
    xr = M @ x
    return np.linalg.norm(x - xr), xr


if __name__ == "__main__":
    n, k = 60, 6  # bloques de tamaño 6 -> m = 10 salidas del pooling
    P = build_pooling_matrix(n, k)
    R = optimal_linear_reconstruction(P)

    M, diff, ok = graph_level_coherence_check(P, R)
    print(f"Chequeo estructural (grafo, sin datos): max|M@M - M| = {diff:.3e}  -> idempotente: {ok}")
    print(f"rango(M) = {np.linalg.matrix_rank(M)}  (esperado = m = {n // k})")

    t = np.linspace(0, 4 * np.pi, n)

    # Señal A: suave (baja frecuencia relativa al bloque) -> reconstruccion casi perfecta
    x_smooth = np.sin(t)
    err_smooth, _ = reconstruction_error(x_smooth, M)

    # Señal B: alta frecuencia dentro de cada bloque -> reconstruccion con perdida
    x_rough = np.sin(t) + 0.9 * np.sin(20 * t)
    err_rough, _ = reconstruction_error(x_rough, M)

    # Señal C: ruido blanco -> perdida maxima (toda la energia intra-bloque se pierde)
    rng = np.random.default_rng(0)
    x_noise = rng.normal(size=n)
    err_noise, _ = reconstruction_error(x_noise, M)

    # Verificacion teorica: el error es exactamente la energia de la componente
    # "dentro de bloque" (I - M) x, cuya norma^2 = sum_bloques k * var_bloque
    def theoretical_within_block_energy(x, k):
        blocks = x.reshape(-1, k)
        means = blocks.mean(axis=1, keepdims=True)
        return np.sqrt(np.sum((blocks - means) ** 2))

    for name, x, err in [("suave", x_smooth, err_smooth),
                          ("rugosa", x_rough, err_rough),
                          ("ruido", x_noise, err_noise)]:
        theo = theoretical_within_block_energy(x, k)
        print(f"señal {name:8s}: error medido = {err:.5f}   error teorico (intra-bloque) = {theo:.5f}")
