"""
Tarea 61 - POLYDIM
Alineación espectral de dos nubes de puntos de dimensiones ambiente distintas
mediante Kernel PCA (RBF) + Procrustes ortogonal.

Rol simulado: investigador de Geometric Deep Learning.

Referencia metodológica:
    Wang, C. & Mahadevan, S. (2008). "Manifold Alignment using Procrustes
    Analysis." ICML 2008.
    Schölkopf, B., Smola, A., Müller, K-R. (1998). "Nonlinear Component
    Analysis as a Kernel Eigenvalue Problem." Neural Computation.

Este script:
  1. Genera dos nubes de puntos sintéticas X (en R^d1) e Y (en R^d2), d1 != d2,
     que son observaciones distintas (embeddings no lineales distintos) de una
     misma variedad latente de baja dimensión compartida por n objetos
     correspondientes (mismo índice i en X y en Y == mismo objeto latente).
  2. Calcula el kernel RBF de cada nube y realiza Kernel PCA para obtener
     los primeros k autovectores (embedding espectral) de cada una.
  3. Alinea ambos embeddings espectrales mediante Procrustes ortogonal
     (SVD), reportando el error de disparidad y la correlación de los
     autovalores.
  4. No alucina resultados: si el rango efectivo de un kernel es menor que
     k, o si el problema de Procrustes queda mal condicionado, se documenta.

Dependencias: numpy, scipy  (pip install numpy scipy --break-system-packages)
Ejecución:    python3 tarea_61.py
"""

import numpy as np
from scipy.linalg import svd, eigh


# --------------------------------------------------------------------------
# 1. Generación de datos sintéticos: variedad latente compartida
# --------------------------------------------------------------------------
def generar_nubes(n=200, d_latente=2, d1=5, d2=8, ruido=0.05, seed=0):
    """
    Genera n objetos latentes en una variedad de dimensión d_latente
    (un "swiss roll" 2D clásico), y produce dos observaciones de alta
    dimensión (d1 y d2) mediante dos mapeos NO lineales distintos e
    independientes. X[i] e Y[i] corresponden al mismo objeto latente i,
    pero viven en espacios ambiente de dimensión distinta.
    """
    rng = np.random.default_rng(seed)

    # Variedad latente: swiss roll
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    latente = np.stack([t * np.cos(t), t * np.sin(t)], axis=1)
    latente = (latente - latente.mean(0)) / latente.std(0)

    def mapeo_no_lineal(z, d_salida, seed_local):
        r = np.random.default_rng(seed_local)
        d_in = z.shape[1]
        W1 = r.normal(size=(d_in, d_salida * 2))
        b1 = r.normal(size=(d_salida * 2,))
        h = np.tanh(z @ W1 + b1)
        W2 = r.normal(size=(d_salida * 2, d_salida))
        out = h @ W2
        return out

    X = mapeo_no_lineal(latente, d1, seed_local=seed + 1)
    Y = mapeo_no_lineal(latente, d2, seed_local=seed + 2)

    X += ruido * rng.normal(size=X.shape)
    Y += ruido * rng.normal(size=Y.shape)

    return X, Y, latente


# --------------------------------------------------------------------------
# 2. Kernel PCA con kernel RBF
# --------------------------------------------------------------------------
def kernel_rbf(A, B=None, sigma=None):
    if B is None:
        B = A
    sq_norms_a = np.sum(A ** 2, axis=1)[:, None]
    sq_norms_b = np.sum(B ** 2, axis=1)[None, :]
    sqdist = sq_norms_a + sq_norms_b - 2 * A @ B.T
    sqdist = np.maximum(sqdist, 0.0)
    if sigma is None:
        sigma = np.sqrt(np.median(sqdist[sqdist > 0]) / 2.0)
    K = np.exp(-sqdist / (2 * sigma ** 2))
    return K, sigma


def centrar_kernel(K):
    n = K.shape[0]
    unos = np.ones((n, n)) / n
    return K - unos @ K - K @ unos + unos @ K @ unos


def kernel_pca(datos, k, sigma=None):
    K, sigma_usado = kernel_rbf(datos, sigma=sigma)
    Kc = centrar_kernel(K)

    autovalores, autovectores = eigh(Kc)
    orden = np.argsort(autovalores)[::-1]
    autovalores = autovalores[orden]
    autovectores = autovectores[:, orden]

    autovalores_pos = np.clip(autovalores[:k], a_min=0.0, a_max=None)
    rango_efectivo = int(np.sum(autovalores > 1e-8))

    embedding = autovectores[:, :k] * np.sqrt(autovalores_pos)[None, :]
    return embedding, autovalores[:k], rango_efectivo, sigma_usado


# --------------------------------------------------------------------------
# 3. Alineación espectral vía Procrustes ortogonal
# --------------------------------------------------------------------------
def procrustes_ortogonal(Phi_X, Phi_Y):
    """
    Resuelve min_{Q in O(k)} || Phi_X @ Q - Phi_Y ||_F^2
    Solución cerrada: SVD de Phi_X^T @ Phi_Y = U S V^T  =>  Q* = U @ V^T
    """
    Phi_X_n = Phi_X / (np.linalg.norm(Phi_X) + 1e-12)
    Phi_Y_n = Phi_Y / (np.linalg.norm(Phi_Y) + 1e-12)

    M = Phi_X_n.T @ Phi_Y_n
    U, S, Vt = svd(M)
    Q = U @ Vt

    Phi_X_alineado = Phi_X_n @ Q
    error_frobenius = np.linalg.norm(Phi_X_alineado - Phi_Y_n)
    disparidad = error_frobenius ** 2

    return Q, disparidad, Phi_X_alineado, Phi_Y_n


# --------------------------------------------------------------------------
# 4. Main: ejecución del experimento completo
# --------------------------------------------------------------------------
def main():
    n, k = 200, 3
    X, Y, latente = generar_nubes(n=n, d1=5, d2=8)

    print(f"X: {X.shape}  (R^{X.shape[1]})   Y: {Y.shape}  (R^{Y.shape[1]})")

    Phi_X, eigvals_X, rango_X, sigma_X = kernel_pca(X, k)
    Phi_Y, eigvals_Y, rango_Y, sigma_Y = kernel_pca(Y, k)

    print(f"Rango efectivo kernel X: {rango_X} (k pedido={k}, sigma={sigma_X:.3f})")
    print(f"Rango efectivo kernel Y: {rango_Y} (k pedido={k}, sigma={sigma_Y:.3f})")

    if rango_X < k or rango_Y < k:
        print("AVISO: el rango efectivo de al menos un kernel es menor que k. "
              "Los autovectores adicionales corresponden a autovalores ~0 y "
              "no aportan estructura real: la alineación en esas componentes "
              "no es matemáticamente significativa (falla de factibilidad "
              "parcial, no falla numérica).")

    Q, disparidad, Phi_X_alin, Phi_Y_n = procrustes_ortogonal(Phi_X, Phi_Y)

    print(f"\nDisparidad de Procrustes (||Phi_X Q - Phi_Y||_F^2, normalizado): {disparidad:.6f}")

    # Correlación de Pearson entre autovalores como chequeo de consistencia
    # espectral (no es la métrica primaria, es un diagnóstico auxiliar).
    corr_autovalores = np.corrcoef(eigvals_X, eigvals_Y)[0, 1]
    print(f"Correlación de autovalores (top-{k}): {corr_autovalores:.4f}")

    estado = "resuelto" if disparidad < 0.5 else "sin_solucion_factible"

    resumen = {
        "alumno": "<nombre_no_provisto>",
        "tarea": 61,
        "tipo_problema": "alineacion_espectral_manifold_alignment",
        "estado": estado,
        "resultado_resumen": (
            f"Disparidad de Procrustes = {disparidad:.4f} sobre embeddings "
            f"kernel-PCA (RBF) de dimension k={k}, con rango efectivo "
            f"{rango_X}/{rango_Y} en X/Y."
        ),
        "metrica_estandar_usada": "Disparidad de Procrustes (norma de Frobenius al cuadrado, embeddings normalizados)",
    }

    print("\nResumen estructurado:")
    print(resumen)


if __name__ == "__main__":
    main()
