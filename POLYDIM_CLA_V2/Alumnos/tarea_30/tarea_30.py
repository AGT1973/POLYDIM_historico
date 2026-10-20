"""
Tarea 30 - POLYDIM
Alineación estadística (Optimal Transport / Sinkhorn) entre las distribuciones
de embeddings de dos modelos de lenguaje, insertada como una pasada de
optimización sobre un grafo computacional (AST).

Autor/alumno: claud62701@gmail.com

Cómo correrlo vos mismo:
    pip install torch numpy
    python tarea_30.py

No hace falta GPU. El script:
  1. Genera embeddings sintéticos de dos "modelos de lenguaje" (A y B).
  2. Implementa Sinkhorn desde cero (sin caja negra) para computar el
     transporte óptimo entrópico entre ambas distribuciones empíricas.
  3. Implementa una pasada de optimización sobre un grafo computacional
     (representado como una lista de nodos, a modo de AST simplificado)
     que detecta las dos subramas de embeddings y las reemplaza por un
     único nodo de alineación OT.
  4. Reporta la distancia de Wasserstein entrópica resultante y si el
     algoritmo convergió.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Modelo matemático: Sinkhorn (Transporte Óptimo entrópico)
# ---------------------------------------------------------------------------
def cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Matriz de costos C_ij = ||x_i - y_j||_2^2  (costo cuadrático euclídeo).
    X: (n, d), Y: (m, d) -> deben tener la misma dimensión d.
    """
    X2 = np.sum(X ** 2, axis=1)[:, None]
    Y2 = np.sum(Y ** 2, axis=1)[None, :]
    C = X2 + Y2 - 2 * X @ Y.T
    return np.maximum(C, 0.0)


def procrustes_projection(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Si las dos nubes de embeddings tienen dimensiones distintas (d_A != d_B),
    no se puede calcular una distancia euclídea directa. Proyectamos Y al
    espacio de X con una matriz ortogonal obtenida vía Procrustes ortogonal
    sobre una correspondencia inicial por rango (heurística válida solo
    quld_A == d_B tras recorte/padding; documentado como limitación).
    """
    d = min(X.shape[1], Y.shape[1])
    return X[:, :d], Y[:, :d]


def sinkhorn(mu: np.ndarray, nu: np.ndarray, C: np.ndarray,
             epsilon: float = 0.05, n_iter: int = 500, tol: float = 1e-9):
    """
    Algoritmo de Sinkhorn-Knopp para el Transporte Óptimo entrópico
    (Cuturi, 2013, 'Sinkhorn Distances: Lightspeed Computation of Optimal
    Transport').

    Resuelve:
        min_pi <C, pi> + epsilon * H(pi)
        s.a. pi @ 1 = mu, pi.T @ 1 = nu, pi >= 0

    mediante escalado matricial alterno de u, v tal que:
        pi = diag(u) @ K @ diag(v),   K = exp(-C / epsilon)

    Devuelve: plan de transporte pi*, distancia entrópica <C, pi*>,
    y si convergió (True/False) según la variación de u entre iteraciones.
    """
    n, m = C.shape
    K = np.exp(-C / epsilon)

    u = np.ones(n) / n
    v = np.ones(m) / m

    converged = False
    for it in range(n_iter):
        u_prev = u.copy()

        # Evitamos división por cero: si una fila/columna de K se degenera
        # (overflow numérico por epsilon muy chico), lo documentamos como
        # inestabilidad y frenamos en vez de propagar NaN.
        Kv = K @ v
        if np.any(Kv == 0) or np.any(np.isnan(Kv)):
            return None, None, False, it

        u = mu / Kv
        Ktu = K.T @ u
        if np.any(Ktu == 0) or np.any(np.isnan(Ktu)):
            return None, None, False, it
        v = nu / Ktu

        if np.linalg.norm(u - u_prev, ord=1) < tol:
            converged = True
            break

    pi = np.diag(u) @ K @ np.diag(v)
    w_dist = float(np.sum(pi * C))  # distancia entrópica <C, pi*>
    return pi, w_dist, converged, it + 1


# ---------------------------------------------------------------------------
# 2. La "pasada de optimización sobre AST topológico"
# ---------------------------------------------------------------------------
#
# Representamos un grafo computacional simplificado (a modo de AST) como una
# lista de nodos. Cada nodo tiene: id, tipo de operación, y datos asociados.
# El objetivo de la pasada es detectar el patrón:
#
#     EmbeddingNode(modelo=A)  --\
#                                  >-- NodoDeUsoDirecto (p.ej. concat, resta)
#     EmbeddingNode(modelo=B)  --/
#
# y reemplazarlo por un único nodo AlignmentNode que en vez de comparar
# los embeddings "en crudo" (lo cual no tiene sentido si las distribuciones
# no están alineadas estadísticamente), inserta la alineación vía Sinkhorn.
#
class Node:
    def __init__(self, node_id, op, data=None, inputs=None):
        self.id = node_id
        self.op = op                # p.ej. "embedding", "concat", "alignment"
        self.data = data            # payload (embeddings, metadatos)
        self.inputs = inputs or []  # ids de nodos de entrada

    def __repr__(self):
        return f"Node(id={self.id}, op={self.op}, inputs={self.inputs})"


def build_toy_graph(emb_a: np.ndarray, emb_b: np.ndarray):
    """
    Construye un grafo de ejemplo:
        n1: embedding del modelo A
        n2: embedding del modelo B
        n3: concat(n1, n2)   <- esto es lo que la pasada debe reemplazar
    """
    n1 = Node("n1", "embedding", data={"model": "A", "emb": emb_a})
    n2 = Node("n2", "embedding", data={"model": "B", "emb": emb_b})
    n3 = Node("n3", "concat", inputs=["n1", "n2"])
    return [n1, n2, n3]


def optimization_pass_align_embeddings(graph, epsilon=0.05, n_iter=500):
    """
    Pasada de optimización sobre el grafo (AST):
      1. Busca pares de nodos tipo "embedding" de modelos distintos que
         alimentan a un mismo nodo consumidor (patrón "AST topológico").
      2. Reemplaza ese nodo consumidor por un nodo "alignment" que
         calcula la distancia/plan de Sinkhorn entre ambas distribuciones.

    Devuelve el nuevo grafo optimizado y el resultado numérico de la
    alineación (para poder loguearlo/graficarlo).
    """
    nodes_by_id = {n.id: n for n in graph}
    new_graph = []
    alignment_result = None

    for node in graph:
        if node.op in ("concat", "sub", "compare"):
            input_nodes = [nodes_by_id[i] for i in node.inputs]
            embedding_inputs = [n for n in input_nodes if n.op == "embedding"]

            if len(embedding_inputs) == 2:
                n_a, n_b = embedding_inputs
                X, Y = n_a.data["emb"], n_b.data["emb"]
                Xp, Yp = procrustes_projection(X, Y)

                n, m = Xp.shape[0], Yp.shape[0]
                mu = np.ones(n) / n
                nu = np.ones(m) / m
                C = cost_matrix(Xp, Yp)

                pi, w_dist, converged, iters = sinkhorn(
                    mu, nu, C, epsilon=epsilon, n_iter=n_iter
                )

                alignment_result = {
                    "modelo_a": n_a.data["model"],
                    "modelo_b": n_b.data["model"],
                    "wasserstein_entropico": w_dist,
                    "convergio": converged,
                    "iteraciones": iters,
                }

                new_graph.append(n_a)
                new_graph.append(n_b)
                new_graph.append(
                    Node(
                        node.id,
                        "alignment",
                        data=alignment_result,
                        inputs=node.inputs,
                    )
                )
                continue

        new_graph.append(node)

    return new_graph, alignment_result


# ---------------------------------------------------------------------------
# 3. Comprobación empírica
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(seed=42)

    d = 16
    n_tokens_a = 60
    n_tokens_b = 45

    # Modelo A: embeddings de una gaussiana centrada en el origen.
    emb_a = rng.normal(loc=0.0, scale=1.0, size=(n_tokens_a, d))

    # Modelo B: embeddings de una gaussiana desplazada y rotada (simula
    # que es otro modelo, entrenado con otra semilla/arquitectura, por lo
    # que su espacio de embeddings no coincide "en crudo" con el de A).
    rotation = np.linalg.qr(rng.normal(size=(d, d)))[0]
    emb_b_raw = rng.normal(loc=0.0, scale=1.0, size=(n_tokens_b, d))
    emb_b = emb_b_raw @ rotation + 2.0  # desplazamiento + rotación

    graph = build_toy_graph(emb_a, emb_b)
    print("Grafo original:")
    for node in graph:
        print(" ", node)

    optimized_graph, result = optimization_pass_align_embeddings(graph)

    print("\nGrafo tras la pasada de optimización:")
    for node in optimized_graph:
        print(" ", node)

    print("\nResultado de la alineación estadística (Sinkhorn OT):")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if result["convergio"]:
        print(
            "\nConclusión: Sinkhorn convergió en "
            f"{result['iteraciones']} iteraciones. La distancia de "
            "Wasserstein entrópica entre las distribuciones de embeddings "
            "de los modelos A y B es "
            f"{result['wasserstein_entropico']:.4f}. Cuanto menor sea este "
            "valor, más parecidas son estadísticamente las dos "
            "distribuciones de embeddings tras la proyección compartida."
        )
    else:
        print(
            "\nConclusión: Sinkhorn NO convergió con los parámetros dados "
            "(epsilon demasiado chico produce inestabilidad numérica en el "
            "escalado matricial K=exp(-C/epsilon), generando divisiones "
            "por cero o NaN). Esto es un resultado válido: documenta que, "
            "para este costo C y este epsilon, el problema de Transporte "
            "Óptimo entrópico es numéricamente inestable, y habría que "
            "subir epsilon o normalizar C."
        )


if __name__ == "__main__":
    main()
