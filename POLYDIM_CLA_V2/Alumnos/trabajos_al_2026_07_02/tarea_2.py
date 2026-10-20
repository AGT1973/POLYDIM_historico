# =============================================================================
# TAREA #2 (ESCUADRON 1) - Descomposicion de Hodge sobre grafo k-NN
# Rol: docente experto en Graph Neural Networks y Matematica Diferenciable
# Contexto: analisis topologico de espacios latentes para POLYDIM
# =============================================================================

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lsqr


# -----------------------------------------------------------------------------
# PASO 1: Generar un batch de embeddings latentes en R^d
# -----------------------------------------------------------------------------
def generar_embeddings(N: int, d: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(N, d)


# -----------------------------------------------------------------------------
# PASO 2: Construir un grafo k-NN esparso
# -----------------------------------------------------------------------------
def construir_grafo_knn(X: torch.Tensor, k: int = 8):
    X_np = X.detach().cpu().numpy()
    N = X_np.shape[0]

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(X_np)
    distancias, indices = nn.kneighbors(X_np)

    distancias = distancias[:, 1:]
    indices = indices[:, 1:]

    filas = np.repeat(np.arange(N), k)
    columnas = indices.flatten()
    pesos = distancias.flatten()

    A = sp.coo_matrix((pesos, (filas, columnas)), shape=(N, N))
    A = A.maximum(A.T)
    return A.tocsr()


# -----------------------------------------------------------------------------
# PASO 3: Construir la matriz de incidencia B1 y el Laplaciano de grafo L0
# -----------------------------------------------------------------------------
def construir_incidencia_y_laplaciano(A: sp.csr_matrix):
    A = sp.triu(A, k=1).tocoo()
    n_edges = A.nnz
    N = A.shape[0]

    filas_inc = np.zeros(2 * n_edges, dtype=np.int64)
    cols_inc = np.zeros(2 * n_edges, dtype=np.int64)
    datos_inc = np.zeros(2 * n_edges, dtype=np.float64)

    for e, (i, j) in enumerate(zip(A.row, A.col)):
        filas_inc[2 * e] = i
        cols_inc[2 * e] = e
        datos_inc[2 * e] = 1.0

        filas_inc[2 * e + 1] = j
        cols_inc[2 * e + 1] = e
        datos_inc[2 * e + 1] = -1.0

    B1 = sp.coo_matrix((datos_inc, (filas_inc, cols_inc)), shape=(N, n_edges)).tocsr()
    L0 = (B1 @ B1.T).tocsr()

    return B1, L0, (A.row, A.col)


# -----------------------------------------------------------------------------
# PASO 4: Descomposicion de Hodge aproximada sobre una senal de arista
# -----------------------------------------------------------------------------
def descomposicion_hodge(B1: sp.csr_matrix, L0: sp.csr_matrix, f_arista: np.ndarray,
                          n_armonicos: int = 6):
    rhs = B1 @ f_arista
    phi, *_ = lsqr(L0, rhs, atol=1e-8, btol=1e-8)[:1] + (None,)
    grad_f = B1.T @ phi
    residuo = f_arista - grad_f

    N = L0.shape[0]
    n_armonicos = min(n_armonicos, N - 2)
    autovalores, autovectores = eigsh(L0 + 1e-10 * sp.eye(N), k=n_armonicos, which="SM")

    return {
        "phi": phi,
        "grad_f": grad_f,
        "residuo_armonico": residuo,
        "autovalores_L0": autovalores,
        "autovectores_armonicos": autovectores,
    }


# -----------------------------------------------------------------------------
# PASO 5: Orquestacion del pipeline completo
# -----------------------------------------------------------------------------
def pipeline_hodge_polydim(N: int = 2000, d: int = 32, k: int = 8, seed: int = 42):
    print(f"[1] Generando {N} embeddings en R^{d}...")
    X = generar_embeddings(N, d, seed)

    print(f"[2] Construyendo grafo k-NN esparso (k={k})...")
    A = construir_grafo_knn(X, k)
    print(f"    -> Aristas totales (no dirigidas, aprox N*k/2): {A.nnz // 2}")

    print("[3] Construyendo matriz de incidencia B1 y Laplaciano L0...")
    B1, L0, (fi, fj) = construir_incidencia_y_laplaciano(A)

    print("[4] Definiendo una senal de prueba sobre las aristas...")
    normas = X.norm(dim=1).numpy()
    f_arista = normas[fi] - normas[fj]

    print("[5] Ejecutando descomposicion de Hodge aproximada...")
    resultado = descomposicion_hodge(B1, L0, f_arista, n_armonicos=6)

    energia_total = float(np.sum(f_arista ** 2))
    energia_armonica = float(np.sum(resultado["residuo_armonico"] ** 2))
    proporcion = energia_armonica / max(energia_total, 1e-12)

    print("\n--- RESULTADOS ---")
    print(f"Autovalores mas chicos de L0 (topologia global): "
          f"{np.round(resultado['autovalores_L0'], 6)}")
    print(f"Energia total de la senal de arista:     {energia_total:.4f}")
    print(f"Energia en la componente armonica:       {energia_armonica:.4f}")
    print(f"Proporcion armonica (energia_h / energia_total): {proporcion:.4f}")

    return resultado


if __name__ == "__main__":
    pipeline_hodge_polydim(N=2000, d=32, k=8)
