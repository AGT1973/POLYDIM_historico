"""
Tarea 37 - POLYDIM
Alineacion estadistica (no supervisada) de las distribuciones de embeddings de
DOS modelos de lenguaje distintos, usando Transporte Optimo Entropico (Sinkhorn,
via la libreria POT) combinado con el metodo de "Wasserstein Procrustes"
(Grave, Joulin, Berthet - AISTATS 2019).

Idea: no se asume una correspondencia palabra-a-palabra conocida entre los dos
espacios de embeddings. Se alterna entre:
  (1) Dada una rotacion ortogonal Q, resolver el plan de transporte optimo
      (Sinkhorn entropico, ot.sinkhorn) entre {Q x_i} y {y_j}.
  (2) Dado el plan de transporte pi, resolver el problema ortogonal de
      Procrustes (via SVD) que mejor alinea X rotado con Y bajo esa
      correspondencia blanda.

Se mide, en funcion del numero de iteraciones y del parametro de
regularizacion entropica (epsilon):
  - el costo de transporte (Wasserstein regularizado),
  - el residuo de Procrustes ||Q X^T - Y^T||_F asociado al acoplamiento actual,
  - la precision de recuperacion por vecino mas cercano (top-1 / top-5) sobre
    un conjunto de pares de referencia (ground truth sintetico), como metrica
    estandar y verificable de calidad de alineacion.

Requiere: numpy, scipy, pot  (pip install pot)
Correr:   python tarea_37.py
"""

import numpy as np
import ot  # Python Optimal Transport (POT)


# ---------------------------------------------------------------------------
# 1. Datos sinteticos: dos "modelos de lenguaje" con embeddings relacionados
#    por una rotacion ortogonal desconocida + ruido (para tener ground truth
#    verificable, ya que con modelos reales no conocemos la Q verdadera).
# ---------------------------------------------------------------------------
def make_synthetic_embeddings(n=300, d=50, noise_std=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))  # embeddings del "modelo A"
    Q_true, _ = np.linalg.qr(rng.normal(size=(d, d)))  # rotacion ortogonal oculta
    Y = X @ Q_true + noise_std * rng.normal(size=(n, d))  # embeddings del "modelo B"
    perm = rng.permutation(n)  # el orden de las filas de Y no revela la correspondencia
    Y = Y[perm]
    ground_truth_perm = np.argsort(perm)  # para evaluar retrieval: Y[perm][ground_truth_perm] == Y original
    return X, Y, ground_truth_perm, Q_true


# ---------------------------------------------------------------------------
# 2. Wasserstein Procrustes: alternar Sinkhorn (POT) <-> SVD-Procrustes
# ---------------------------------------------------------------------------
def sinkhorn_coupling(Xq, Y, epsilon):
    """Plan de transporte entropico entre las nubes Xq (=Q@X) e Y, masas uniformes."""
    n, m = Xq.shape[0], Y.shape[0]
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    C = ot.dist(Xq, Y, metric="sqeuclidean")
    C = C / C.max()  # normalizar para estabilidad numerica de Sinkhorn
    pi = ot.sinkhorn(a, b, C, reg=epsilon)
    cost = float(np.sum(pi * C))
    return pi, cost


def procrustes_from_coupling(X, Y, pi):
    """Dado el acoplamiento pi (n x m), resuelve Q* = argmin_Q ||Q X^T - Y_reordenado^T||_F
    via SVD del producto cruzado (Procrustes ortogonal clasico)."""
    # Y "esperado" para cada fila de X bajo el acoplamiento blando:
    Y_soft = pi @ Y * X.shape[0]  # reescalar porque pi tiene masas 1/n en filas
    M = X.T @ Y_soft  # d x d
    U, _, Vt = np.linalg.svd(M)
    Q = U @ Vt
    return Q


def wasserstein_procrustes(X, Y, epsilon=0.05, n_iters=15, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))  # inicializacion aleatoria ortogonal
    history = []
    for it in range(n_iters):
        Xq = X @ Q
        pi, cost = sinkhorn_coupling(Xq, Y, epsilon)
        Q = procrustes_from_coupling(X, Y, pi)
        residual = float(np.linalg.norm((X @ Q) - (pi @ Y) * X.shape[0], ord="fro"))
        history.append({"iter": it, "sinkhorn_cost": cost, "procrustes_residual": residual})
    return Q, pi, history


# ---------------------------------------------------------------------------
# 3. Metrica de evaluacion: precision de recuperacion por vecino mas cercano
# ---------------------------------------------------------------------------
def nearest_neighbor_accuracy(X, Y, Q, ground_truth_perm, k=5):
    Xq = X @ Q
    # distancias de cada fila de Xq a todas las filas de Y
    dists = ot.dist(Xq, Y, metric="sqeuclidean")
    order = np.argsort(dists, axis=1)
    top1 = np.mean(order[:, 0] == ground_truth_perm)
    topk = np.mean([ground_truth_perm[i] in order[i, :k] for i in range(len(ground_truth_perm))])
    return float(top1), float(topk)


# ---------------------------------------------------------------------------
# 4. Estudio: efecto de la regularizacion entropica epsilon en la alineacion
# ---------------------------------------------------------------------------
def epsilon_study(n=300, d=50, epsilons=(0.2, 0.1, 0.05, 0.02), n_iters=15, seed=0):
    X, Y, gt_perm, Q_true = make_synthetic_embeddings(n=n, d=d, seed=seed)
    print(f"{'epsilon':>8} | {'sinkhorn_cost_final':>20} | {'procrustes_res_final':>22} | "
          f"{'top1_acc':>9} | {'top5_acc':>9}")
    results = []
    for eps in epsilons:
        Q, pi, history = wasserstein_procrustes(X, Y, epsilon=eps, n_iters=n_iters, seed=seed)
        top1, top5 = nearest_neighbor_accuracy(X, Y, Q, gt_perm, k=5)
        last = history[-1]
        print(f"{eps:>8.3f} | {last['sinkhorn_cost']:>20.6f} | {last['procrustes_residual']:>22.6f} | "
              f"{top1:>9.3f} | {top5:>9.3f}")
        results.append({"epsilon": eps, "history": history, "top1": top1, "top5": top5})
    return results


if __name__ == "__main__":
    epsilon_study()
