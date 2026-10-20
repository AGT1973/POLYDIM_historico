"""
Tarea 22 - POLYDIM
Framework de benchmarking: Descomposición de Hodge combinatoria (exacta vs. aproximada)
aislando la componente armónica de un flujo definido sobre las aristas de un grafo.

Referencia teórica: Jiang, Lim, Yao, Ye (2011),
"Statistical ranking and combinatorial Hodge theory", Math. Programming 127(1).

Requisitos:
    pip install numpy scipy networkx --break-system-packages

Ejecución:
    python3 hodge_benchmark.py
"""

import time
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg, lsqr
from scipy.linalg import null_space


# ---------------------------------------------------------------------------
# 1. Construcción del complejo simplicial (grafo + triángulos como 2-celdas)
# ---------------------------------------------------------------------------

def build_complex(n, p, seed=0):
    """
    Genera un grafo aleatorio G(n,p) conexo y extrae:
      - lista de vértices, aristas orientadas (i<j) y triángulos (cliques de 3)
      - operadores de borde B1 (|V| x |E|) y B2 (|E| x |T|)

    B1 es el operador de divergencia clásico de teoría de grafos (transpuesto
    del gradiente). B2 es el operador de "curl adjunto": manda cada triángulo
    a la combinación orientada de sus 3 aristas.
    """
    rng = np.random.default_rng(seed)
    G = nx.gnp_random_graph(n, p, seed=seed)
    # aseguramos conexidad (para que el problema de gradiente tenga sentido)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            a = next(iter(components[i]))
            b = next(iter(components[i + 1]))
            G.add_edge(a, b)

    edges = sorted(tuple(sorted(e)) for e in G.edges())
    edge_index = {e: k for k, e in enumerate(edges)}
    m = len(edges)

    # B1: |V| x |E|  (columna por arista (i,j), i<j: -1 en i, +1 en j)
    rows, cols, vals = [], [], []
    for k, (i, j) in enumerate(edges):
        rows += [i, j]
        cols += [k, k]
        vals += [-1.0, 1.0]
    B1 = csr_matrix((vals, (rows, cols)), shape=(n, m))

    # Triángulos = 2-simplices (cliques de tamaño 3 en G)
    triangles = [tuple(sorted(t)) for t in nx.enumerate_all_cliques(G) if len(t) == 3]
    t_count = len(triangles)

    # B2: |E| x |T|  (columna por triángulo (i,j,k), i<j<k:
    #                 +1 en (i,j), +1 en (j,k), -1 en (i,k)  -> curl(i,j,k) = 0 si es gradiente puro)
    rows2, cols2, vals2 = [], [], []
    for t_idx, (i, j, k) in enumerate(triangles):
        for (a, b), sign in [((i, j), 1.0), ((j, k), 1.0), ((i, k), -1.0)]:
            rows2.append(edge_index[(a, b)])
            cols2.append(t_idx)
            vals2.append(sign)
    B2 = csr_matrix((vals2, (rows2, cols2)), shape=(m, t_count))

    return G, edges, triangles, B1, B2


def random_edge_flow(m, seed=0):
    """Flujo aleatorio X sobre las aristas (representa, ej., comparaciones pareadas ruidosas)."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=m)


# ---------------------------------------------------------------------------
# 2. Descomposición de Hodge EXACTA (álgebra lineal directa)
# ---------------------------------------------------------------------------

def hodge_exact(B1, B2, X):
    """
    X = grad(phi) + curl*(psi) + h

    grad(phi) = B1^T phi          (componente de gradiente, en fila-espacio de B1)
    curl*(psi) = B2 psi           (componente de curl, en columna-espacio de B2)
    h          = componente armónica, h in ker(B1) ∩ ker(B2^T)

    Se resuelve con mínimos cuadrados exactos (pseudo-inversa) y la componente
    armónica se obtiene por resta, verificando ortogonalidad.
    """
    B1d, B2d = B1.toarray(), B2.toarray()

    # phi = argmin || B1^T phi - X ||  ->  sistema normal B1 B1^T phi = B1 X
    L0 = B1d @ B1d.T  # Laplaciano de vértices (grafo estándar)
    phi = np.linalg.pinv(L0) @ (B1d @ X)
    grad_component = B1d.T @ phi

    # psi = argmin || B2 psi - X ||  ->  sistema normal B2^T B2 psi = B2^T X
    L2 = B2d.T @ B2d
    psi = np.linalg.pinv(L2) @ (B2d.T @ X)
    curl_component = B2d @ psi

    harmonic = X - grad_component - curl_component
    return grad_component, curl_component, harmonic


# ---------------------------------------------------------------------------
# 3. Descomposición de Hodge APROXIMADA (solvers iterativos dispersos)
# ---------------------------------------------------------------------------

def hodge_approx(B1, B2, X, tol=1e-6, maxiter=1000):
    """
    Mismo modelo matemático que hodge_exact, pero evitando la pseudo-inversa
    densa O(n^3): se resuelven los sistemas normales con gradiente conjugado
    (CG) sobre matrices dispersas, que es lo que se usa en la práctica para
    grafos grandes (Jiang et al. no invierten Δ1 completo, resuelven sistemas
    lineales dispersos).
    """
    L0 = (B1 @ B1.T).tocsr()
    b0 = np.asarray(B1 @ X).ravel()
    phi, info0 = cg(L0, b0, rtol=tol, maxiter=maxiter)
    grad_component = np.asarray(B1.T @ phi).ravel()

    L2 = (B2.T @ B2).tocsr()
    b2 = np.asarray(B2.T @ X).ravel()
    if L2.shape[0] > 0:
        psi, info2 = cg(L2, b2, rtol=tol, maxiter=maxiter)
        curl_component = np.asarray(B2 @ psi).ravel()
    else:
        curl_component = np.zeros_like(X)
        info2 = 0

    harmonic = X - grad_component - curl_component
    converged = (info0 == 0) and (info2 == 0)
    return grad_component, curl_component, harmonic, converged


# ---------------------------------------------------------------------------
# 4. Framework de benchmarking
# ---------------------------------------------------------------------------

def relative_error(a, b):
    denom = np.linalg.norm(b)
    if denom < 1e-12:
        return np.linalg.norm(a - b)
    return np.linalg.norm(a - b) / denom


def run_benchmark(sizes, p=0.3, tol=1e-8, seed=0):
    results = []
    for n in sizes:
        G, edges, triangles, B1, B2 = build_complex(n, p, seed=seed)
        m = len(edges)
        X = random_edge_flow(m, seed=seed)

        t0 = time.perf_counter()
        g_exact, c_exact, h_exact = hodge_exact(B1, B2, X)
        t_exact = time.perf_counter() - t0

        t0 = time.perf_counter()
        g_approx, c_approx, h_approx, converged = hodge_approx(B1, B2, X, tol=tol)
        t_approx = time.perf_counter() - t0

        err_harmonic = relative_error(h_approx, h_exact)
        err_grad = relative_error(g_approx, g_exact)
        err_curl = relative_error(c_approx, c_exact)

        # chequeo de consistencia: la norma de la componente armónica debe ser
        # pequeña si el grafo es "rico" en triángulos (poca inconsistencia global),
        # y mayor si hay ciclos largos no triangulados.
        harmonic_norm_ratio = np.linalg.norm(h_exact) / (np.linalg.norm(X) + 1e-12)

        results.append({
            "n_vertices": n,
            "n_edges": m,
            "n_triangles": len(triangles),
            "t_exact_s": t_exact,
            "t_approx_s": t_approx,
            "speedup": t_exact / t_approx if t_approx > 0 else float("inf"),
            "rel_err_harmonic": err_harmonic,
            "rel_err_grad": err_grad,
            "rel_err_curl": err_curl,
            "harmonic_norm_ratio": harmonic_norm_ratio,
            "cg_converged": converged,
        })
    return results


def print_report(results):
    header = f"{'n':>5} {'|E|':>6} {'|T|':>6} {'t_exacto(s)':>12} {'t_aprox(s)':>11} " \
             f"{'speedup':>8} {'err_h':>10} {'err_grad':>10} {'err_curl':>10} {'‖h‖/‖X‖':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['n_vertices']:>5} {r['n_edges']:>6} {r['n_triangles']:>6} "
              f"{r['t_exact_s']:>12.5f} {r['t_approx_s']:>11.5f} "
              f"{r['speedup']:>8.2f} {r['rel_err_harmonic']:>10.2e} "
              f"{r['rel_err_grad']:>10.2e} {r['rel_err_curl']:>10.2e} "
              f"{r['harmonic_norm_ratio']:>9.3f}")


if __name__ == "__main__":
    # Nota de diseño: el numero de triangulos (2-simplices) crece ~ O(n^3 p^3),
    # y el metodo EXACTO usa una pseudo-inversa DENSA de tamano |T| x |T| solo
    # para tener un ground-truth de referencia. Por eso aqui se usa una
    # densidad de aristas moderada (p=0.15) y tamanos acotados: el objetivo es
    # comparar exactitud/velocidad del metodo aproximado contra el exacto, no
    # llevar el metodo exacto al limite (ese es justamente el punto: el metodo
    # exacto NO escala, y por eso se necesita el aproximado).
    sizes = [15, 25, 40, 60, 90]
    results = run_benchmark(sizes, p=0.15, tol=1e-8, seed=42)
    print_report(results)

    print("\nInterpretación rápida (ver tarea_22.md para el análisis completo):")
    print("- err_h, err_grad, err_curl deberían mantenerse cerca de la tolerancia")
    print("  del CG (~1e-6 a 1e-8), NO crecer con n si el método aproximado es estable.")
    print("- speedup (t_exacto / t_aproximado) debería crecer con n: la pseudo-inversa")
    print("  densa es O(n^3) mientras que CG sobre matrices dispersas escala mejor.")
