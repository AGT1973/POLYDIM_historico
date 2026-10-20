"""
Tarea 60 - Alineacion Espectral entre nubes de puntos de dimensiones distintas
via Procrustes Ortogonal sobre sus primeros k autovectores.

Contexto reconstruido (ver tarea_60.md para el detalle completo):
El enunciado original pedia "una pasada de optimizacion que intercepte un AST
topologico y [alinee] dos nubes de puntos de dimensiones diferentes emparejando
sus primeros k autovectores". Se interpreta como: un "pass" de optimizacion de
grafo (al estilo de un compilador, aqui simulado como una lista de nodos de
computo) que reemplaza dos representaciones intermedias (activaciones) de
dimensiones ambientales distintas por una alineacion en un espacio espectral
comun de dimension k, usando Procrustes ortogonal.

Ejecucion (el alumno la corre en su propia cuenta):
    python3 tarea_60.py

Requiere: numpy (incluido en la instalacion estandar de Anaconda / pip install numpy)
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Construccion del grafo de vecinos (k-NN) y su Laplaciano normalizado
# ---------------------------------------------------------------------------

def knn_laplacian(X: np.ndarray, n_neighbors: int = 10, sigma: float = 1.0) -> np.ndarray:
    """
    Construye el Laplaciano normalizado de un grafo k-NN con pesos gaussianos.

    X: (n, d) nube de puntos en dimension ambiente d (d puede diferir entre
       las dos nubes que se van a alinear; el Laplaciano siempre es (n, n)).
    """
    n = X.shape[0]
    # Matriz de distancias euclidianas al cuadrado
    sq_dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2 * X @ X.T
    sq_dists = np.maximum(sq_dists, 0.0)

    # Pesos gaussianos (kernel RBF)
    W = np.exp(-sq_dists / (2 * sigma**2))
    np.fill_diagonal(W, 0.0)

    # Retener solo los n_neighbors mas cercanos (grafo disperso, simetrizado)
    idx_sorted = np.argsort(-W, axis=1)
    mask = np.zeros_like(W, dtype=bool)
    for i in range(n):
        mask[i, idx_sorted[i, :n_neighbors]] = True
    mask = mask | mask.T  # simetrizar
    W = W * mask

    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-12)))
    L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    return L_sym


# ---------------------------------------------------------------------------
# 2. Embedding espectral: primeros k autovectores del Laplaciano
# ---------------------------------------------------------------------------

def spectral_embedding(L: np.ndarray, k: int) -> np.ndarray:
    """
    Devuelve los k autovectores asociados a los k autovalores mas chicos
    (excluyendo el autovalor 0 trivial asociado al vector constante),
    normalizados a norma unitaria.
    """
    eigvals, eigvecs = np.linalg.eigh(L)  # L es simetrico -> eigh es estable
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]
    # Se descarta el primer autovector (constante, autovalor ~0)
    U = eigvecs[:, 1:k + 1]
    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
    return U


# ---------------------------------------------------------------------------
# 3. Procrustes ortogonal: alinear U_X con U_Y minimizando ||U_X Q - U_Y||_F
# ---------------------------------------------------------------------------

def orthogonal_procrustes(U_X: np.ndarray, U_Y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Resuelve min_Q ||U_X Q - U_Y||_F  sujeto a Q^T Q = I.
    Solucion clasica (Schonemann, 1966): Q = V W^T, donde U_X^T U_Y = W S V^T (SVD).

    Devuelve (Q, error_L2_normalizado).
    """
    M = U_X.T @ U_Y
    W, S, Vt = np.linalg.svd(M)
    Q = W @ Vt

    U_X_aligned = U_X @ Q
    error_l2 = np.linalg.norm(U_X_aligned - U_Y, ord="fro") / np.linalg.norm(U_Y, ord="fro")
    return Q, float(error_l2)


# ---------------------------------------------------------------------------
# 4. "Pasada de optimizacion" sobre un grafo de computo (simulacion del AST)
# ---------------------------------------------------------------------------

class ComputeNode:
    """Nodo minimo de un grafo de computo (analogo a un nodo de AST/FX)."""

    def __init__(self, name: str, op: str, data=None):
        self.name = name
        self.op = op
        self.data = data

    def __repr__(self):
        return f"ComputeNode(name={self.name!r}, op={self.op!r})"


class SpectralAlignmentPass:
    """
    Pasada de optimizacion: recorre un grafo de computo (lista de ComputeNode)
    y cuando encuentra dos nodos 'embedding_raw' con datos de dimensiones
    ambientales distintas, los reemplaza por un unico nodo 'aligned_embedding'
    que contiene ambas nubes proyectadas al mismo subespacio espectral.

    Esto es la version "graph optimization pass" del enunciado original:
    intercepta el grafo (el "AST topologico") y sustituye el par de nodos
    dispares por una representacion alineada y comparable.
    """

    def __init__(self, k: int, n_neighbors: int = 10):
        self.k = k
        self.n_neighbors = n_neighbors

    def run(self, graph: list[ComputeNode]) -> list[ComputeNode]:
        raw_nodes = [n for n in graph if n.op == "embedding_raw"]
        if len(raw_nodes) != 2:
            raise ValueError("La pasada espera exactamente 2 nodos 'embedding_raw'.")

        X, Y = raw_nodes[0].data, raw_nodes[1].data

        L_X = knn_laplacian(X, n_neighbors=self.n_neighbors)
        L_Y = knn_laplacian(Y, n_neighbors=self.n_neighbors)

        U_X = spectral_embedding(L_X, self.k)
        U_Y = spectral_embedding(L_Y, self.k)

        Q, error_l2 = orthogonal_procrustes(U_X, U_Y)
        U_X_aligned = U_X @ Q

        new_graph = [n for n in graph if n.op != "embedding_raw"]
        new_graph.append(
            ComputeNode(
                name="aligned_embedding",
                op="aligned_embedding",
                data={"U_X_aligned": U_X_aligned, "U_Y": U_Y, "Q": Q, "error_l2": error_l2},
            )
        )
        return new_graph


# ---------------------------------------------------------------------------
# 5. Demostracion empirica (sanity check interno, no reemplaza la corrida propia)
# ---------------------------------------------------------------------------

def run_case(X, Y, k, n_neighbors, label):
    graph = [
        ComputeNode("x_raw", "embedding_raw", data=X),
        ComputeNode("y_raw", "embedding_raw", data=Y),
    ]
    optimization_pass = SpectralAlignmentPass(k=k, n_neighbors=n_neighbors)
    new_graph = optimization_pass.run(graph)
    error_l2 = new_graph[0].data["error_l2"]
    print(f"[{label}] Error L2 normalizado (Procrustes): {error_l2:.4f}")
    return error_l2


def demo():
    rng = np.random.default_rng(42)
    n = 300
    k = 5

    # Nube X en R^7: generada a partir de 2 variables latentes (curva 2D)
    t = np.linspace(0, 4 * np.pi, n)
    latent = np.stack([np.cos(t), np.sin(t)], axis=1)
    A = rng.normal(size=(2, 7))
    X = latent @ A + 0.05 * rng.normal(size=(n, 7))  # d=7

    B = rng.normal(size=(2, 12))

    # CASO 1 (control): misma correspondencia punto-a-punto (indices alineados)
    Y_matched = latent @ B + 0.05 * rng.normal(size=(n, 12))
    err_matched = run_case(X, Y_matched, k, n_neighbors=12, label="correspondencia conocida")

    # CASO 2 (el planteado por el enunciado): sin correspondencia conocida
    perm = rng.permutation(n)
    Y_shuffled = latent[perm] @ B + 0.05 * rng.normal(size=(n, 12))
    err_shuffled = run_case(X, Y_shuffled, k, n_neighbors=12, label="sin correspondencia (shuffled)")

    print()
    print("Interpretacion:")
    print(f"  - Con correspondencia punto-a-punto conocida, el error es notablemente menor")
    print(f"    ({err_matched:.4f}) que en el caso shuffled: el metodo SI captura buena parte")
    print("    de la estructura latente compartida (no es cero por el ruido gaussiano y por")
    print("    ambiguedades de rotacion en autoespacios con autovalores casi degenerados).")
    print(f"  - Sin correspondencia conocida, el error sube marcadamente ({err_shuffled:.4f}):")
    print("    el k-NN Laplaciano se construye sobre la posicion de cada punto, y al mezclar")
    print("    el orden se rompe la estructura local de vecindad que el metodo necesita.")
    print("    Es decir: 'emparejar los primeros k autovectores' via Procrustes resuelve un")
    print("    problema de ALINEACION DE SUBESPACIOS, no un problema de CORRESPONDENCIA")
    print("    ENTRE PUNTOS -- son problemas distintos, y el enunciado original los confunde")
    print("    implicitamente. Esta distincion es el nucleo conceptual de la tarea.")


if __name__ == "__main__":
    demo()
