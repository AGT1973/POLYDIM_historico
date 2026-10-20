"""
Tarea 47 - POLYDIM Escuadron 3
Convergencia de operadores monotonos sobre reticulos completos
(Teorema de Knaster-Tarski) aplicado a sistemas de restricciones
de pertenencia.

Alumno: ai.mpat.marta@gmail.com

Este script:
1. Implementa un sistema de restricciones de pertenencia generico
   sobre un universo finito U.
2. Itera con la cadena de Kleene X_0=vacio, X_(k+1)=T(X_k) hasta punto fijo.
3. Verifica la monotonia del operador.
4. Compara un caso monotono (converge, "colapso estable") contra un
   caso no monotono (oscila, no converge) para probar que la
   monotonia es la condicion necesaria del teorema.

No requiere dependencias externas.
"""

from __future__ import annotations
from typing import Callable, FrozenSet, Set


Universe = FrozenSet[int]
Rule = Callable[[int, Set[int]], bool]


class ConstraintSystem:
    """
    Sistema de restricciones de pertenencia sobre un universo finito U.

    T(X) = X U { u en U : rule(u, X) es verdadero }
    """

    def __init__(self, universe: Universe, rule: Rule):
        self.universe = universe
        self.rule = rule

    def T(self, X: Set[int]) -> Set[int]:
        """Un paso del operador de propagacion."""
        new_X = set(X)
        for u in self.universe:
            if self.rule(u, X):
                new_X.add(u)
        return new_X

    def is_monotone(self, samples: int = 200, seed: int = 0) -> bool:
        """
        Chequeo empirico de monotonia: X subset Y => T(X) subset T(Y),
        muestreando pares aleatorios de subconjuntos del universo.
        (Chequeo empirico, no una demostracion formal per-regla; sirve
        como sanity check antes de confiar en la convergencia teorica.)
        """
        import random
        rng = random.Random(seed)
        elems = list(self.universe)
        for _ in range(samples):
            X = {e for e in elems if rng.random() < 0.5}
            extra = {e for e in elems if e not in X and rng.random() < 0.5}
            Y = X | extra  # garantizado X subset Y
            TX, TY = self.T(X), self.T(Y)
            if not TX.issubset(TY):
                return False
        return True

    def kleene_iterate(self, max_steps: int | None = None):
        """
        Itera T desde el vacio hasta alcanzar un punto fijo (o hasta
        detectar un ciclo, en el caso no monotono).

        Devuelve: (lista_de_estados, punto_fijo_o_None, es_ciclico)
        """
        if max_steps is None:
            max_steps = len(self.universe) + 5  # cota teorica + margen

        estados = [set()]
        vistos = [frozenset()]

        for k in range(max_steps):
            X_actual = estados[-1]
            X_siguiente = self.T(X_actual)

            estados.append(X_siguiente)

            if X_siguiente == X_actual:
                # Punto fijo alcanzado: colapso estable.
                return estados, X_siguiente, False

            fz = frozenset(X_siguiente)
            if fz in vistos:
                # Ciclo detectado: el sistema oscila, no converge.
                return estados, None, True

            vistos.append(fz)

        # No convergio ni se detecto ciclo en max_steps (no deberia
        # pasar si el universo es finito y T es monotono).
        return estados, None, False


def hamming_distance(A: Set[int], B: Set[int]) -> int:
    """|A Delta B| : cardinalidad de la diferencia simetrica."""
    return len(A.symmetric_difference(B))


def demo_caso_monotono():
    print("=" * 70)
    print("CASO 1: sistema MONOTONO (solo agrega, nunca retira)")
    print("=" * 70)

    universo: Universe = frozenset(range(10))

    # Regla monotona: u entra si algun "vecino" u-1 o u+1 ya esta en X,
    # o si u es un elemento semilla (0). Nunca se retira nada de X.
    semillas = {0}

    def regla_monotona(u: int, X: Set[int]) -> bool:
        if u in semillas:
            return True
        return (u - 1) in X or (u + 1) in X

    sistema = ConstraintSystem(universo, regla_monotona)

    print(f"Universo: {sorted(universo)}")
    print(f"Monotono (chequeo empirico): {sistema.is_monotone()}")

    estados, punto_fijo, es_ciclico = sistema.kleene_iterate()

    print("\nCadena de Kleene (X_0, X_1, ...):")
    for i, X in enumerate(estados):
        print(f"  X_{i} = {sorted(X)}")

    print("\nDistancias de Hamming entre pasos sucesivos |X_(k+1) Delta X_k|:")
    distancias = [
        hamming_distance(estados[i], estados[i + 1])
        for i in range(len(estados) - 1)
    ]
    print(f"  {distancias}")

    if punto_fijo is not None:
        print(f"\n>> Colapso estable alcanzado en k*={len(estados) - 1} "
              f"iteraciones (cota teorica: |U|={len(universo)}).")
        print(f">> Punto fijo mu_T = {sorted(punto_fijo)}")
        print(f">> Verificacion de idempotencia T(mu_T) == mu_T: "
              f"{sistema.T(punto_fijo) == punto_fijo}")
    else:
        print("\n>> ERROR INESPERADO: un sistema monotono deberia converger.")


def demo_caso_no_monotono():
    print("\n" + "=" * 70)
    print("CASO 2: sistema NO MONOTONO (contraejemplo: puede retirar)")
    print("=" * 70)

    universo: Universe = frozenset(range(6))

    # Regla NO monotona: u entra si su "opuesto" (u+3 mod 6) NO esta en X.
    # Esto viola monotonia: agregar elementos a X puede hacer que un u
    # que antes calificaba, deje de calificar.
    def regla_no_monotona(u: int, X: Set[int]) -> bool:
        opuesto = (u + 3) % 6
        return opuesto not in X

    sistema = ConstraintSystem(universo, regla_no_monotona)

    print(f"Universo: {sorted(universo)}")
    print(f"Monotono (chequeo empirico): {sistema.is_monotone()}")

    estados, punto_fijo, es_ciclico = sistema.kleene_iterate(max_steps=12)

    print("\nCadena de Kleene (X_0, X_1, ...):")
    for i, X in enumerate(estados):
        print(f"  X_{i} = {sorted(X)}")

    distancias = [
        hamming_distance(estados[i], estados[i + 1])
        for i in range(len(estados) - 1)
    ]
    print("\nDistancias de Hamming entre pasos sucesivos |X_(k+1) Delta X_k|:")
    print(f"  {distancias}")

    if es_ciclico:
        print("\n>> El sistema NO colapsa de manera estable: entro en un "
              "ciclo (oscilacion) sin alcanzar un punto fijo.")
        print(">> Esto confirma que la monotonia es condicion necesaria "
              "para la convergencia garantizada por Knaster-Tarski.")
    elif punto_fijo is not None:
        print("\n>> (Este caso particular convergio de todas formas; "
              "no toda regla no monotona oscila, pero no hay garantia "
              "teorica de que lo haga. Probar con otras reglas.)")
    else:
        print("\n>> No convergio ni se detecto ciclo en el limite de pasos.")


if __name__ == "__main__":
    demo_caso_monotono()
    demo_caso_no_monotono()

    # --- Bloque opcional de graficado (requiere matplotlib) ---
    # import matplotlib.pyplot as plt
    # ... graficar las distancias de Hamming por iteracion para
    #     visualizar la convergencia vs. la oscilacion.
