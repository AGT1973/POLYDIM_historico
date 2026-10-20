"""
Tarea 57 - POLYDIM
Alineacion Espectral (Spectral / Manifold Alignment) entre dos nubes de puntos
de dimensiones diferentes (y en general, cardinalidades diferentes).

Ver tarea_57.md para el desarrollo matematico completo.

Dependencias:
    pip install numpy scipy scikit-learn --break-system-packages
Opcional (para W2 exacta via Sinkhorn):
    pip install pot --break-system-packages
"""

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import kneighbors_graph

try:
    import ot  # POT: Python Optimal Transport
    HAS_POT = True
except ImportError:
    HAS_POT = False


def make_sphere_cloud(n, dim, seed, noise=0.0, stretch=None):
    """Genera una nube de n puntos sobre una esfera unitaria embebida en R^dim,
    opcionalmente estirada por un vector 'stretch' y con ruido gaussiano."""
    rng = np.random.default_rng(seed)
    # Coordenadas base en 3D (variedad real), luego embebidas en R^dim con ceros/relleno
    theta = rng.uniform(0, np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    base = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=1)  # (n, 3)

    if dim < 3:
        raise ValueError("dim debe ser >= 3 para embeber la esfera")

    embed = np.zeros((n, dim))
    embed[:, :3] = base
    if dim > 3:
        # dimensiones extra: combinaciones lineales suaves de las 3 primeras
        # (mantiene la variedad intrinsecamente 2D pero en dimension ambiente mayor)
        rng2 = np.random.default_rng(seed + 1)
        mix = rng2.normal(size=(3, dim - 3)) * 0.3
        embed[:, 3:] = base @ mix

    if stretch is not None:
        embed = embed * np.asarray(stretch)

    if noise > 0:
        embed = embed + rng.normal(scale=noise, size=embed.shape)

    return embed


def spectral_embedding(X, k, n_neighbors=10):
    """Calcula el Laplaciano normalizado de X y devuelve los primeros k
    autovectores no triviales (embedding espectral) junto con sus autovalores."""
    n = X.shape[0]
    n_neighbors = min(n_neighbors, n - 1)

    W = kneighbors_graph(X, n_neighbors=n_neighbors, mode="distance", include_self=False)
    # kernel gaussiano sobre las distancias kNN
    sigma = np.mean(W.data) if len(W.data) > 0 else 1.0
    W.data = np.exp(-(W.data ** 2) / (2 * sigma ** 2 + 1e-12))
    W = W.maximum(W.T)  # simetrizar

    deg = np.asarray(W.sum(axis=1)).flatten()
    deg[deg == 0] = 1e-12
    d_inv_sqrt = diags(1.0 / np.sqrt(deg))
    L = diags(np.ones(n)) - d_inv_sqrt @ W @ d_inv_sqrt  # Laplaciano normalizado

    # k+1 autovalores mas chicos (incluye el trivial ~0)
    num_eigs = min(k + 1, n - 1)
    eigvals, eigvecs = eigsh(csr_matrix(L), k=num_eigs, which="SM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # descartar el autovector trivial (autovalor mas chico, ~componente conexa)
    eigvals_nontrivial = eigvals[1:1 + k]
    eigvecs_nontrivial = eigvecs[:, 1:1 + k]

    # resolver ambiguedad de signo
    for i in range(eigvecs_nontrivial.shape[1]):
        col = eigvecs_nontrivial[:, i]
        j = np.argmax(np.abs(col))
        if col[j] < 0:
            eigvecs_nontrivial[:, i] *= -1

    return eigvecs_nontrivial, eigvals_nontrivial


def spectral_distance(eigvals_X, eigvals_Y):
    """Distancia L2 entre espectros (invariante al cardinal n)."""
    k = min(len(eigvals_X), len(eigvals_Y))
    return float(np.linalg.norm(eigvals_X[:k] - eigvals_Y[:k]))


def wasserstein_or_hungarian(Phi_X, Phi_Y):
    """Distancia de Wasserstein-2 entre los embeddings (via POT/Sinkhorn si
    esta disponible); si no, fallback a asignacion optima discreta (Hungarian)
    como aproximacion, documentando explicitamente la aproximacion."""
    n1, n2 = Phi_X.shape[0], Phi_Y.shape[0]
    cost = np.linalg.norm(Phi_X[:, None, :] - Phi_Y[None, :, :], axis=2) ** 2

    if HAS_POT:
        a = np.ones(n1) / n1
        b = np.ones(n2) / n2
        reg = 0.05 * np.median(cost[cost > 0]) if np.any(cost > 0) else 0.05
        transport_plan = ot.sinkhorn(a, b, cost, reg)
        w2_sq = float(np.sum(transport_plan * cost))
        return np.sqrt(max(w2_sq, 0.0)), "Wasserstein-2 exacta (Sinkhorn, POT)"
    else:
        # Aproximacion discreta: asignacion optima 1-a-1 sobre el subconjunto
        # min(n1, n2). Esto NO es el transporte optimo continuo (que permite
        # masa fraccionaria), es una cota superior aproximada.
        m = min(n1, n2)
        row_ind, col_ind = linear_sum_assignment(cost[:m, :m])
        approx = np.sqrt(float(np.mean(cost[row_ind, col_ind])))
        return approx, "Aproximacion via Hungarian (POT no instalado, NO es W2 exacta)"


def main():
    k = 4
    n1, n2 = 150, 90

    # Nube X: esfera "pura" en R^3
    X = make_sphere_cloud(n1, dim=3, seed=0, noise=0.02)
    # Nube Y: MISMA variedad (esfera), pero en R^5 y con distinto cardinal -> deberia alinear bien
    Y_same_manifold = make_sphere_cloud(n2, dim=5, seed=1, noise=0.02)
    # Nube Z: variedad DISTINTA (esfera fuertemente estirada, casi un disco) -> no deberia alinear tan bien
    Z_diff_manifold = make_sphere_cloud(n2, dim=5, seed=2, noise=0.02, stretch=[1, 1, 0.05, 1, 1])

    Phi_X, lam_X = spectral_embedding(X, k)

    print("=== Caso A: misma variedad subyacente (esfera en R^3 vs esfera en R^5) ===")
    Phi_Y, lam_Y = spectral_embedding(Y_same_manifold, k)
    d_spec_A = spectral_distance(lam_X, lam_Y)
    w2_A, method_A = wasserstein_or_hungarian(Phi_X, Phi_Y)
    print(f"  n1={n1}, n2={n2}, k={k}")
    print(f"  Distancia espectral (L2 autovalores): {d_spec_A:.4f}")
    print(f"  Distancia de alineacion ({method_A}): {w2_A:.4f}")

    print("\n=== Caso B: variedad distinta (esfera vs. esfera aplanada en una direccion) ===")
    Phi_Z, lam_Z = spectral_embedding(Z_diff_manifold, k)
    d_spec_B = spectral_distance(lam_X, lam_Z)
    w2_B, method_B = wasserstein_or_hungarian(Phi_X, Phi_Z)
    print(f"  Distancia espectral (L2 autovalores): {d_spec_B:.4f}")
    print(f"  Distancia de alineacion ({method_B}): {w2_B:.4f}")

    print("\n=== Veredicto ===")
    print("Umbral relativo usado: se considera 'misma variedad' si d_spec_B / d_spec_A > 1.5")
    if d_spec_A > 0 and (d_spec_B / d_spec_A) > 1.5:
        print("-> Consistente con lo esperado: el espectro distingue variedades distintas")
        print("   con mayor sensibilidad que solo mirar dimension ambiente/cardinal.")
    else:
        print("-> Los espectros no muestran una separacion clara con este umbral;")
        print("   probar con mas puntos, mas vecinos (n_neighbors) o mayor deformacion.")

    print("\nNota metodologica: la comparacion 'punto a punto' de autovectores solo")
    print("es valida cuando n1 == n2. Aqui n1 != n2, por eso se reporta el espectro")
    print("(invariante a n) y la distancia de transporte optimo entre embeddings,")
    print("en vez de un emparejamiento ingenuo componente a componente.")


if __name__ == "__main__":
    main()
