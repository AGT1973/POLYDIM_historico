"""
Tarea 52 - Alineacion estadistica de nubes de puntos via Transporte Optimo (Sinkhorn)
y superposicion rigida con conservacion de volumen (rotacion propia, det(R) = +1).

Requiere: numpy, scipy, pot  (pip install numpy scipy pot --break-system-packages)

Este script es para que el alumno lo corra y verifique el modelo empiricamente.
No fue ejecutado como parte de la entrega final (ver .md), solo se hizo un
chequeo de sintaxis interno.
"""

import numpy as np
from scipy.linalg import svd

try:
    import ot  # POT: Python Optimal Transport
except ImportError as e:
    raise ImportError(
        "Falta la libreria POT. Instalala con: pip install pot --break-system-packages"
    ) from e


def generar_nubes(n=200, d=3, seed=0):
    """Genera dos nubes de puntos relacionadas por una rotacion conocida + traslacion + ruido.

    Esto permite verificar si el algoritmo recupera la rotacion verdadera.
    """
    rng = np.random.default_rng(seed)

    # Nube base
    X = rng.normal(size=(n, d))

    # Rotacion propia "verdadera" generada a partir de una matriz antisimetrica (algebra de Lie so(d))
    A = rng.normal(size=(d, d))
    A = A - A.T  # antisimetrica => exp(A) es una rotacion propia (det = +1)
    R_true = expm_antisimetrica(A)

    t_true = rng.normal(scale=2.0, size=d)

    ruido = rng.normal(scale=0.05, size=(n, d))
    Y = X @ R_true.T + t_true + ruido

    return X, Y, R_true, t_true


def expm_antisimetrica(A, terminos=30):
    """Exponencial de matriz para una matriz antisimetrica, via serie de Taylor.
    Evita depender de scipy.linalg.expm para dejar explicito el uso del algebra de Lie.
    """
    d = A.shape[0]
    resultado = np.eye(d)
    termino = np.eye(d)
    for k in range(1, terminos):
        termino = termino @ A / k
        resultado = resultado + termino
    return resultado


def plan_transporte_sinkhorn(X, Y, epsilon=0.05):
    """Calcula el plan de transporte optimo regularizado (Sinkhorn) entre
    las distribuciones empiricas uniformes sobre X e Y.

    Devuelve la matriz de plan P (n x m), con P.sum() = 1.
    """
    n, m = X.shape[0], Y.shape[0]
    mu = np.full(n, 1.0 / n)
    nu = np.full(m, 1.0 / m)

    C = ot.dist(X, Y, metric="sqeuclidean")
    C = C / C.max()  # normalizar costo para estabilidad numerica del Sinkhorn

    P = ot.sinkhorn(mu, nu, C, reg=epsilon)
    return P


def correspondencia_baricentrica(X, Y, P):
    """A partir del plan de transporte P, construye para cada x_i el punto
    correspondiente en Y como baricentro pesado por las filas de P (normalizadas).
    """
    filas = P.sum(axis=1, keepdims=True)
    filas[filas == 0] = 1e-12
    Y_hat = (P @ Y) / filas
    return Y_hat


def procrustes_rotacion_propia(X, Y_hat):
    """Resuelve el problema de Procrustes ortogonal restringido a rotaciones propias
    (SO(d), det = +1): minimiza || R X_c^T - Y_hat_c^T ||_F sobre R ortogonal con det(R)=+1.

    Algoritmo de Kabsch / Umeyama (1991).
    Devuelve R, t tales que  R @ x_i + t ~ y_hat_i
    """
    x_mean = X.mean(axis=0)
    y_mean = Y_hat.mean(axis=0)

    Xc = X - x_mean
    Yc = Y_hat - y_mean

    H = Xc.T @ Yc
    U, S, Vt = svd(H)

    d = X.shape[1]
    D = np.eye(d)
    # Correccion de signo: garantiza det(R) = +1 (rotacion propia, sin reflexion)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[-1, -1] = -1.0

    R = Vt.T @ D @ U.T
    t = y_mean - R @ x_mean

    return R, t


def error_alineacion(X, Y_hat, R, t):
    Y_pred = X @ R.T + t
    return np.linalg.norm(Y_pred - Y_hat) / np.sqrt(X.shape[0])


def main():
    X, Y, R_true, t_true = generar_nubes()

    P = plan_transporte_sinkhorn(X, Y, epsilon=0.05)
    Y_hat = correspondencia_baricentrica(X, Y, P)

    R, t = procrustes_rotacion_propia(X, Y_hat)

    det_R = np.linalg.det(R)
    err = error_alineacion(X, Y_hat, R, t)
    err_R = np.linalg.norm(R - R_true)

    print("=== Verificacion empirica: Tarea 52 ===")
    print(f"det(R) recuperado         : {det_R:.6f}  (debe ser ~ +1.0)")
    print(f"Error de alineacion (RMS) : {err:.6f}")
    print(f"|| R - R_true ||_F        : {err_R:.6f}  (debe ser pequeno si Sinkhorn corresponde bien)")
    print()
    print("Interpretacion: si det(R) ~ 1 y el error de alineacion es bajo, la")
    print("superposicion encontrada conserva el volumen (es una rotacion propia)")
    print("y logra alinear estadisticamente ambas nubes via transporte optimo.")


if __name__ == "__main__":
    main()
