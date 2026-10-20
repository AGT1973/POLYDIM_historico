"""
Tarea 2 - Escuadron 1 (POLYDIM Swarm)
Rol solicitado: Tester de robustez de IA
Enunciado original: "Escribe codigo que simule la superposicion
garantizando conservacion del volumen determinante."
"""

import numpy as np

np.random.seed(42)


def matriz_ortogonal_aleatoria(n: int) -> np.ndarray:
    """
    Genera una matriz ortogonal Q (n x n) con det(Q) = +1 (rotacion pura).
    Estas matrices SI conservan volumen exactamente: |det(Q)| = 1.
    """
    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_composicion(n: int, n_ensayos: int = 1000) -> dict:
    """
    COMPOSICION (multiplicacion de matrices): Q1 @ Q2
    det(Q1 @ Q2) = det(Q1) * det(Q2) = 1 * 1 = 1
    """
    errores = []
    for _ in range(n_ensayos):
        Q1 = matriz_ortogonal_aleatoria(n)
        Q2 = matriz_ortogonal_aleatoria(n)
        C = Q1 @ Q2
        errores.append(abs(abs(np.linalg.det(C)) - 1.0))
    return {
        "operacion": "composicion (Q1 @ Q2)",
        "error_promedio_vs_volumen_1": float(np.mean(errores)),
        "error_maximo": float(np.max(errores)),
        "conserva_volumen": bool(np.max(errores) < 1e-9),
    }


def test_superposicion(n: int, n_ensayos: int = 1000) -> dict:
    """
    SUPERPOSICION (suma de matrices): Q1 + Q2
    No existe identidad algebraica que garantice det(Q1+Q2) = 1.
    """
    errores = []
    for _ in range(n_ensayos):
        Q1 = matriz_ortogonal_aleatoria(n)
        Q2 = matriz_ortogonal_aleatoria(n)
        S = Q1 + Q2
        errores.append(abs(abs(np.linalg.det(S)) - 1.0))
    return {
        "operacion": "superposicion (Q1 + Q2)",
        "error_promedio_vs_volumen_1": float(np.mean(errores)),
        "error_maximo": float(np.max(errores)),
        "conserva_volumen": bool(np.max(errores) < 1e-9),
    }


if __name__ == "__main__":
    n = 4
    resultado_composicion = test_composicion(n)
    resultado_superposicion = test_superposicion(n)

    print("=== TEST DE ROBUSTEZ: conservacion de volumen (determinante) ===")
    print(f"Dimension probada: R^{n}, ensayos por caso: 1000\n")
    print("Composicion (multiplicacion):", resultado_composicion)
    print("Superposicion (suma):        ", resultado_superposicion)

    if resultado_superposicion["conserva_volumen"]:
        print("\nConclusion: la superposicion conserva volumen.")
    else:
        print("\nConclusion: FALLA MATEMATICA detectada.")
        print("La superposicion (suma) de transformaciones que preservan")
        print("volumen NO conserva volumen. Solo la composicion lo hace.")
