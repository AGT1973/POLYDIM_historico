"""
Tarea 19 - Alineación Espectral y pertenencia a la Criba de Cobertura de un Sitio de Grothendieck
====================================================================================================

Ver tarea_19.md para el desarrollo matemático completo. Este script es la
verificación empírica del modelo: dado un subespacio latente estático
S subseteq R^3 y una familia cubridora {U_i} de un sitio (C, J) construido
sobre el retículo de subespacios de R^3, decide si S pertenece a la criba
generada por esa familia, usando el espectro (autovalores 0/1) del
operador de proyección ortogonal como criterio de alineación.

Requisitos: numpy (pip install numpy --break-system-packages)
Ejecución:  python3 tarea_19.py
"""

import numpy as np


def orthonormal_basis(vectors: list[list[float]]) -> np.ndarray:
    """Devuelve una base ortonormal (columnas) del subespacio generado por `vectors`."""
    M = np.array(vectors, dtype=float).T  # columnas = vectores generadores
    Q, R = np.linalg.qr(M)
    rank = np.linalg.matrix_rank(M)
    return Q[:, :rank]


def projection_matrix(U: np.ndarray) -> np.ndarray:
    """
    Operador de proyección ortogonal Pi_U = U U^T (U ya ortonormal).
    Su espectro es exactamente {1 (multiplicidad dim U), 0 (multiplicidad n - dim U)}:
    esta dicotomía espectral es el criterio de "alineación" que usamos para
    decidir pertenencia (factorización) de S a través de U.
    """
    return U @ U.T


def spectral_alignment_residual(S_basis: np.ndarray, U_basis: np.ndarray) -> float:
    """
    Mide qué tan bien S se alinea con el autoespacio de autovalor 1 de Pi_U.
    Residual = || (I - Pi_U) S_basis ||_F / (# vectores de S).

    Si el residual es ~0, S vive enteramente en el autoespacio propio de
    autovalor 1 de Pi_U, es decir S subseteq U (factorización estricta).
    Si el residual es grande, S tiene componente en el autoespacio de
    autovalor 0 (complemento ortogonal de U): no factoriza a través de U.
    """
    Pi_U = projection_matrix(U_basis)
    projected = Pi_U @ S_basis
    residual = S_basis - projected
    return float(np.linalg.norm(residual) / S_basis.shape[1])


def covering_sieve_membership(
    S_vectors: list[list[float]],
    covering_family: list[list[list[float]]],
    epsilon: float = 1e-9,
) -> tuple[bool, list[dict]]:
    """
    Decide si S pertenece a la criba de cobertura generada por `covering_family`
    sobre X = R^3.

    Definición operacional de la criba generada por una familia cubridora
    {U_i -> X}: es el conjunto de todos los subobjetos T -> X que factorizan
    a través de ALGUN U_i, es decir T subseteq U_i para algún i (cerradura
    por precomposición de una sieve generada por una familia de morfismos).

    Devuelve (pertenece_a_la_sieve, detalle_por_patch).
    """
    S_basis = orthonormal_basis(S_vectors)
    results = []
    for i, U_vectors in enumerate(covering_family):
        U_basis = orthonormal_basis(U_vectors)
        residual = spectral_alignment_residual(S_basis, U_basis)
        covered = residual < epsilon
        results.append({"patch": i, "residual": residual, "covered": covered})

    in_sieve = any(r["covered"] for r in results)
    return in_sieve, results


def is_valid_cover(covering_family: list[list[list[float]]], ambient_dim: int = 3) -> bool:
    """
    Chequeo de sanidad de la topología J: {U_i} es efectivamente una familia
    cubridora de X si la suma (join en el retículo de subespacios) de los U_i
    es todo X, es decir rank(union de generadores) == dim(X).
    Esto NO es lo mismo que "S pertenece a la sieve" (eso se decide patch a
    patch), pero es la condición para que la familia sea, en primer lugar,
    un cover legítimo del sitio.
    """
    joint = np.hstack([np.array(u, dtype=float).T for u in covering_family])
    return np.linalg.matrix_rank(joint) == ambient_dim


if __name__ == "__main__":
    # Sitio: X = R^3. Familia cubridora J-covering de X: los 3 planos coordenados.
    covering_family = [
        [[1, 0, 0], [0, 1, 0]],   # U_1 = plano xy
        [[0, 1, 0], [0, 0, 1]],   # U_2 = plano yz
        [[1, 0, 0], [0, 0, 1]],   # U_3 = plano xz
    ]

    print("¿La familia {U_i} es un cover válido de R^3? ->", is_valid_cover(covering_family))
    print()

    casos = {
        "S_en_plano_xy (subseteq U_1)": [[2.0, 3.0, 0.0]],
        "S_en_plano_xy_otro_vector (subseteq U_1)": [[1.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
        "S_diagonal_fuera_de_todos_los_U_i": [[1.0, 1.0, 1.0]],
        "S_eje_x (subseteq de los 3 U_i)": [[5.0, 0.0, 0.0]],
    }

    for nombre, S_vectors in casos.items():
        in_sieve, detalle = covering_sieve_membership(S_vectors, covering_family)
        print(f"--- {nombre} ---")
        print(f"  S = {S_vectors}")
        print(f"  ¿pertenece a la criba de cobertura?  {in_sieve}")
        for d in detalle:
            print(f"    patch U_{d['patch']}: residual={d['residual']:.3e}  covered={d['covered']}")
        print(f"  Interpretación topos: {'pullback no vacío (Ω=verdadero en algún patch)' if in_sieve else 'pullback al objeto inicial ∅ (Ω=falso en todos los patches)'}")
        print()
