"""
TAREA 29 - POLYDIM V2
Rol asignado: experto en Manifold Learning.

Pregunta de investigacion (reconstruida, ver tarea_29.md):
La 'homotopia aproximada' propuesta en la Seccion 9.1 del paper POLYDIM V2
(Persistent Homology + distancia de Wasserstein entre diagramas de persistencia),
es capaz de distinguir si dos trayectorias de composicion de rotores,
  path_A = R2 . R1   vs   path_B = R1 . R2
conmutan o no?

Ground truth algebraico: conmutan sii  ||R1 R2 - R2 R1||_F ~ 0
(cierto para SO(3) solo si R1, R2 comparten eje de rotacion, o alguno es la identidad).

Este script:
  1. Construye rotores (matrices de SO(3)) a partir de ejes/angulos.
  2. Calcula el defecto de conmutador algebraico (ground truth).
  3. Aplica ambas trayectorias a una nube de puntos X y obtiene Y_A, Y_B.
  4. Calcula la homologia persistente de grado 0 (via single-linkage sobre
     la matriz de distancias, que es exactamente el filtrado de Rips en grado 0)
     de Y_A y de Y_B, y la distancia de Wasserstein-1D entre esos diagramas.
  5. Compara ambos diagnosticos (algebraico vs. topologico) en dos regimenes:
     rotores genericos (no conmutan) y rotores coaxiales (si conmutan).
  6. Calcula ademas un test extrinseco (Wasserstein ambiente entre las nubes
     Y_A, Y_B directamente en R^3, sin pasar por invariantes de isometria)
     como alternativa que SI detecta la diferencia.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage


# ---------------------------------------------------------------------------
# 1. Rotores como matrices de SO(3) (equivalentes al sandwich de Clifford
#    R v R~ via la doble cobertura Spin(3) ~ cuaterniones unitarios).
# ---------------------------------------------------------------------------
def rotation_matrix(axis, theta):
    """Matriz de rotacion de Rodrigues para eje unitario `axis` y angulo `theta`."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    I = np.eye(3)
    return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# ---------------------------------------------------------------------------
# 2. Homologia persistente de grado 0 = dendrograma de single-linkage.
#    Esto es un hecho estandar de TDA (Carlsson, 2009): el complejo de Rips
#    filtrado por radio produce, en H0, exactamente los pares
#    (nacimiento=0, muerte=altura de fusion en single-linkage).
#    No se necesita ninguna libreria externa de TDA para esto.
# ---------------------------------------------------------------------------
def persistence_diagram_h0(points):
    """Devuelve las 'muertes' (deaths) del diagrama de persistencia H0
    (todos los nacimientos son 0 en un filtrado de Rips por vertices)."""
    D = pdist(points)
    Z = linkage(D, method="single")
    deaths = np.sort(Z[:, 2])  # alturas de fusion, ascendentes
    return deaths


def wasserstein_1d(a, b):
    """Distancia de Wasserstein-1 entre dos multiconjuntos 1D de igual tamano
    (tras rellenar con 0 si difieren en cardinalidad, convencion estandar de
    comparar contra la diagonal en TDA)."""
    n = max(len(a), len(b))
    a2 = np.pad(a, (0, n - len(a)))
    b2 = np.pad(b, (0, n - len(b)))
    return np.mean(np.abs(np.sort(a2) - np.sort(b2)))


def commutator_defect(R1, R2):
    return np.linalg.norm(R1 @ R2 - R2 @ R1, ord="fro")


def ambient_wasserstein(Y_A, Y_B):
    """Test extrinseco alternativo: Wasserstein entre las nubes de puntos
    en las coordenadas AMBIENTE (no invariante a isometrias), usando
    transporte optimo 1-1 por indice (asignacion conocida, ya que ambas
    provienen de aplicar dos caminos al MISMO x_i)."""
    return np.mean(np.linalg.norm(Y_A - Y_B, axis=1))


def run_case(name, axis1, theta1, axis2, theta2, X, epsilon=1e-6):
    R1 = rotation_matrix(np.array(axis1, dtype=float), theta1)
    R2 = rotation_matrix(np.array(axis2, dtype=float), theta2)

    delta_alg = commutator_defect(R1, R2)

    Y_A = (R2 @ R1 @ X.T).T   # path_A = R2 . R1
    Y_B = (R1 @ R2 @ X.T).T   # path_B = R1 . R2

    dgm_A = persistence_diagram_h0(Y_A)
    dgm_B = persistence_diagram_h0(Y_B)
    wd_topo = wasserstein_1d(dgm_A, dgm_B)

    wd_ambient = ambient_wasserstein(Y_A, Y_B)

    # matrices de distancias intrinsecas identicas? (prueba directa de
    # invariancia isometrica, mas fuerte que comparar solo el diagrama H0)
    dist_X = squareform(pdist(X))
    dist_YA = squareform(pdist(Y_A))
    dist_YB = squareform(pdist(Y_B))
    intrinsic_match = np.allclose(dist_YA, dist_X) and np.allclose(dist_YB, dist_X)

    print(f"--- Caso: {name} ---")
    print(f"  Defecto algebraico ||R1R2-R2R1||_F      = {delta_alg:.6e}"
          f"  -> {'NO conmutan' if delta_alg > epsilon else 'SI conmutan'} (ground truth)")
    print(f"  Wasserstein(diagrama H0(Y_A), H0(Y_B))  = {wd_topo:.6e}"
          f"  -> test del paper diria: {'SI conmutan' if wd_topo < epsilon else 'NO conmutan'}")
    print(f"  Matrices de distancia intrinsecas iguales a las de X: {intrinsic_match}")
    print(f"  Wasserstein AMBIENTE (Y_A vs Y_B, coords R^3)         = {wd_ambient:.6e}"
          f"  -> test alternativo diria: {'SI conmutan' if wd_ambient < epsilon else 'NO conmutan'}")
    print()
    return delta_alg, wd_topo, wd_ambient


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))  # nube de puntos generica en R^3

    print("=======================================================")
    print(" TAREA 29 - Test de homotopia aproximada (PH + Wasserstein)")
    print(" vs. conmutador algebraico real de rotores")
    print("=======================================================\n")

    # Caso 1: rotores genericos, ejes distintos -> NO deberian conmutar
    d1 = run_case(
        "rotores genericos (ejes distintos)",
        axis1=[1, 0, 0], theta1=0.9,
        axis2=[0, 1, 0], theta2=1.3,
        X=X,
    )

    # Caso 2: rotores coaxiales (mismo eje) -> SI conmutan
    d2 = run_case(
        "rotores coaxiales (mismo eje)",
        axis1=[0, 0, 1], theta1=0.9,
        axis2=[0, 0, 1], theta2=1.3,
        X=X,
    )

    print("=======================================================")
    print("CONCLUSION EMPIRICA")
    print("=======================================================")
    print(f"Caso generico  : delta_alg={d1[0]:.3e}  wd_topo={d1[1]:.3e}  wd_ambiente={d1[2]:.3e}")
    print(f"Caso coaxial   : delta_alg={d2[0]:.3e}  wd_topo={d2[1]:.3e}  wd_ambiente={d2[2]:.3e}")
    print()
    print("El defecto algebraico (ground truth) difiere claramente entre los dos casos")
    print("(no conmuta vs. conmuta), pero wd_topo es ~0 en AMBOS casos: el test de")
    print("homotopia aproximada basado en homologia persistente es CIEGO a la")
    print("no-conmutatividad de rotores, porque las rotaciones son isometrias y la")
    print("homologia persistente de Rips depende solo de distancias intrinsecas,")
    print("que una rotacion global no modifica. wd_ambiente, en cambio, SI distingue")
    print("ambos casos correctamente.")
