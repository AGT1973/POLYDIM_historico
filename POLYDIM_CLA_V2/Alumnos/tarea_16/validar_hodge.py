"""
Validacion empirica de la descomposicion de Hodge combinatoria.

Corre dos casos:
  1. Un ARBOL (sin ciclos) -> se espera componente armonica ~ 0.
  2. Un grafo con 2 ciclos independientes -> se espera componente armonica
     no nula, con norma de divergencia ~0 (verificando B^T h = 0).

Como correrlo:
    python3 validar_hodge.py

Requiere: numpy, scipy  (pip install numpy scipy)
"""
import numpy as np
from hodge_reference import (
    build_incidence_matrix,
    hodge_decompose,
    first_betti_number,
)

rng = np.random.default_rng(16)


def caso_arbol():
    print("=" * 60)
    print("CASO 1: arbol (sin ciclos) -> b1 debe ser 0")
    n = 7
    # arbol en estrella + cadena, sin ciclos
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    B = build_incidence_matrix(n, edges)
    b1 = first_betti_number(n, edges, n_components=1)
    print(f"  nodos={n}, aristas={len(edges)}, b1 (teorico) = {b1}")

    phi = rng.normal(size=len(edges))
    res = hodge_decompose(phi, B)
    print(f"  ||componente armonica||   = {np.linalg.norm(res.harmonic_component):.3e}")
    print(f"  ||B^T h|| (debe ser ~0)   = {res.div_of_harmonic_norm:.3e}")
    print(f"  error de reconstruccion   = {res.reconstruction_error:.3e}")
    assert np.linalg.norm(res.harmonic_component) < 1e-8, "Se esperaba componente armonica nula en un arbol"
    print("  OK: componente armonica ~ 0, consistente con b1=0.\n")


def caso_con_ciclos():
    print("=" * 60)
    print("CASO 2: grafo con 2 ciclos independientes -> b1 = 2")
    n = 6
    # cadena base 0-1-2-3-4-5 (arbol, 5 aristas) + 2 aristas extra que cierran ciclos
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    edges += [(0, 2), (3, 5)]  # cierra 2 ciclos independientes
    B = build_incidence_matrix(n, edges)
    b1 = first_betti_number(n, edges, n_components=1)
    print(f"  nodos={n}, aristas={len(edges)}, b1 (teorico) = {b1}")

    phi = rng.normal(size=len(edges))
    res = hodge_decompose(phi, B)
    print(f"  energia total       ||phi||^2      = {np.linalg.norm(phi)**2:.4f}")
    print(f"  energia componente gradiente       = {res.energy_grad:.4f}")
    print(f"  energia componente armonica        = {res.energy_harmonic:.4f}")
    print(f"  ||B^T h|| (debe ser ~0)            = {res.div_of_harmonic_norm:.3e}")
    print(f"  error de reconstruccion            = {res.reconstruction_error:.3e}")
    assert res.div_of_harmonic_norm < 1e-8, "La componente armonica deberia tener divergencia nula"
    assert np.linalg.norm(res.harmonic_component) > 1e-6, "Con ciclos se espera componente armonica no nula"
    print("  OK: divergencia de h ~ 0, y h != 0 como corresponde a b1=2.\n")


if __name__ == "__main__":
    caso_arbol()
    caso_con_ciclos()
    print("=" * 60)
    print("Todos los chequeos pasaron.")
