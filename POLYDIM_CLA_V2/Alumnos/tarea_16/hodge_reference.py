"""
Tarea 16 - Descomposicion de Hodge combinatoria aproximada sobre un grafo,
con aislamiento de la componente armonica.

Referencias:
- Jiang, X., Lim, L.-H., Yao, Y., Ye, Y. (2011). "Statistical ranking and
  combinatorial Hodge theory". Mathematical Programming, 127(1), 203-244.
- Lim, L.-H. (2020). "Hodge Laplacians on graphs". SIAM Review, 62(3), 685-715.
- Diestel, R. Graph Theory (cap. sobre cut space / cycle space).

Idea central
------------
Un grafo G=(V,E) con |V|=n, |E|=m define un operador de borde orientado
B in R^{m x n} (matriz de incidencia con signo):

    B[e, i] = -1, B[e, j] = +1   si e = (i, j) con orientacion i -> j

Este operador cumple dos roles duales:
  - grad: R^V -> R^E,   (grad f)_e = f(j) - f(i) = (B f)_e
  - div : R^E -> R^V,   div = B^T  (adjunto de grad)

El Laplaciano combinatorio de vertices es L0 = B^T B = D - A (grado - adyacencia).

Todo flujo de arista phi in R^E se descompone de forma unica y ortogonal
(bajo el producto interno estandar) como:

    phi = grad(s*) + h,      h = phi - B s*

donde s* es la solucion (por minimos cuadrados, salvo una constante aditiva
global que fijamos ancorando un nodo de referencia) de las ecuaciones
normales:

    L0 s* = B^T phi

Esta h satisface B^T h = 0, es decir h pertenece a ker(B^T), el "cycle
space" del grafo (Diestel). Cuando el complejo simplicial asociado NO tiene
2-celdas (triangulos), este cycle space coincide exactamente con la
componente armonica de la teoria de Hodge combinatoria completa (no hay
termino de "curl" porque no hay 2-simplices sobre los cuales definirlo).
Por eso el enunciado la llama "aproximada": es la Hodge decomposition en el
caso particular del 1-esqueleto del grafo, no la version con 3 terminos
(grad + curl* + harmonic) que requeriria una estructura de complejo
simplicial de orden superior.

Resultado de control (para el alumno):
  - Si el grafo es un arbol (sin ciclos), b1 = m - n + c = 0, y h debe dar
    identicamente 0 (dentro de tolerancia numerica).
  - Si el grafo tiene ciclos independientes, dim(ker B^T) = b1 > 0 y h
    en general es no nulo.

Extension opcional (Hodge completo con 2-simplices)
----------------------------------------------------
Si se dispone ademas de una lista de triangulos (2-simplices) T, se puede
definir el operador de borde superior B1 in R^{|T| x m} y el Laplaciano de
Hodge en aristas:

    L1 = B0^T B0 + B1 B1^T   (con B0 := B de arriba)

La componente armonica "completa" es entonces el kernel de L1, que en
general es un subespacio MAS PEQUENO que ker(B^T) del caso sin triangulos
(los triangulos "rellenan" ciclos y los sacan de la componente armonica).
Se deja implementado como extension en `hodge_decompose_with_triangles`.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass


@dataclass
class HodgeResult:
    s: np.ndarray            # potencial de vertices (gauge fijado en nodo 0)
    grad_component: np.ndarray  # B @ s  (componente "gradiente")
    harmonic_component: np.ndarray  # phi - B @ s
    div_of_harmonic_norm: float  # ||B\^T h||_2 , debe ser ~0
    reconstruction_error: float  # ||phi - (grad + harmonic)||_2 , debe ser ~0
    energy_grad: float
    energy_harmonic: float


def build_incidence_matrix(n_nodes: int, edges: list[tuple[int, int]]) -> sp.csr_matrix:
    """B in R^{m x n}: fila e tiene -1 en edges[e][0] y +1 en edges[e][1]."""
    m = len(edges)
    rows = np.repeat(np.arange(m), 2)
    cols = np.array([[i, j] for (i, j) in edges]).ravel()
    data = np.tile([-1.0, 1.0], m)
    B = sp.csr_matrix((data, (rows, cols)), shape=(m, n_nodes))
    return B


def hodge_decompose(phi: np.ndarray, B: sp.csr_matrix, ref_node: int = 0) -> HodgeResult:
    """
    Resuelve L0 s = B^T phi fijando s[ref_node] = 0 (gauge) para eliminar
    la singularidad del Laplaciano (su kernel son las funciones constantes
    en cada componente conexa).
    """
    n = B.shape[1]
    L0 = (B.T @ B).tocsr()  # Laplaciano de vertices, n x n

    rhs = (B.T @ phi)

    # Fijamos el gauge: eliminamos la fila/columna ref_node y resolvemos
    # el sistema reducido (n-1) x (n-1), luego reinsertamos s[ref_node]=0.
    keep = [i for i in range(n) if i != ref_node]
    L0_red = L0[np.ix_(keep, keep)]
    rhs_red = rhs[keep]

    s_red = spla.spsolve(L0_red.tocsc(), rhs_red)

    s = np.zeros(n)
    s[keep] = s_red

    grad_component = B @ s
    harmonic_component = phi - grad_component

    div_h = B.T @ harmonic_component
    recon_err = np.linalg.norm(phi - (grad_component + harmonic_component))

    return HodgeResult(
        s=s,
        grad_component=grad_component,
        harmonic_component=harmonic_component,
        div_of_harmonic_norm=float(np.linalg.norm(div_h)),
        reconstruction_error=float(recon_err),
        energy_grad=float(np.linalg.norm(grad_component) ** 2),
        energy_harmonic=float(np.linalg.norm(harmonic_component) ** 2),
    )


def first_betti_number(n_nodes: int, edges: list[tuple[int, int]], n_components: int = 1) -> int:
    """b1 = m - n + c  (numero de ciclos independientes)."""
    m = len(edges)
    return m - n_nodes + n_components


# ---------------------------------------------------------------------------
# Extension opcional: Hodge completo con 2-simplices (triangulos)
# ---------------------------------------------------------------------------

def build_upper_incidence(edges: list[tuple[int, int]], triangles: list[tuple[int, int, int]]) -> sp.csr_matrix:
    """
    B1 in R^{|T| x m}: para cada triangulo (i,j,k) con i<j<k, su borde
    orientado es la suma con signo de sus 3 aristas: (i,j) + (j,k) - (i,k).
    """
    edge_index = {e: idx for idx, e in enumerate(edges)}
    m = len(edges)
    rows, cols, data = [], [], []
    for t_idx, (i, j, k) in enumerate(triangles):
        for (a, b), sign in [((i, j), 1.0), ((j, k), 1.0), ((i, k), -1.0)]:
            e = (a, b) if (a, b) in edge_index else (b, a)
            if e not in edge_index:
                raise ValueError(f"Arista {(a,b)} del triangulo {(i,j,k)} no esta en la lista de aristas")
            s = sign if e == (a, b) else -sign
            rows.append(t_idx)
            cols.append(edge_index[e])
            data.append(s)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(triangles), m))


def hodge_decompose_with_triangles(
    phi: np.ndarray,
    B0: sp.csr_matrix,
    B1: sp.csr_matrix,
    n_harmonic_probe: int = 200,
) -> dict:
    """
    Version completa (grad + curl* + harmonica) usando el Laplaciano de
    Hodge en aristas L1 = B0^T B0 + B1 B1^T. La componente armonica
    "verdadera" vive en ker(L1). Se estima una base de ker(L1) mediante
    los autovectores de menor autovalor (aproximacion numerica: no es
    exacta, de ahi tambien el caracter "aproximado" del ejercicio a esta
    escala).
    """
    L1 = (B0.T @ B0 + B1 @ B1.T).tocsr()
    m = L1.shape[0]
    k = min(n_harmonic_probe, m - 1) if m > 2 else 1
    # eigsh requiere simetrica (L1 lo es); buscamos los autovalores mas chicos
    vals, vecs = spla.eigsh(L1, k=k, which="SM")
    tol = 1e-8
    harmonic_basis = vecs[:, vals < tol]
    # proyeccion de phi sobre el subespacio armonico
    if harmonic_basis.shape[1] > 0:
        coeffs = harmonic_basis.T @ phi
        harmonic_component = harmonic_basis @ coeffs
    else:
        harmonic_component = np.zeros_like(phi)
    return {
        "eigenvalues": vals,
        "harmonic_dim_estimated": int(harmonic_basis.shape[1]),
        "harmonic_component": harmonic_component,
    }
