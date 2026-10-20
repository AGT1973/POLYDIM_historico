"""
tarea_15 — Referencia en Python del modelo de "colapso estable" de un
sistema lógico de restricciones de pertenencia.

Reproduce el mismo modelo matemático que el arquetipo de VM en Rust/WASM
(ver src/main.rs), pero en Python puro para que sea trivial de correr,
inspeccionar y modificar.

Requisitos: Python 3.8+. Sin dependencias externas.

Cómo correrlo:
    python3 tarea_15_fixpoint_demo.py

Qué deberías ver: cuatro corridas del mismo sistema de reglas, cada una
con un orden de evaluación distinto, y una confirmación de que las
cuatro convergen exactamente al mismo estado final (el mínimo punto
fijo, en el sentido de Knaster-Tarski).

Si querés estresar el modelo, cambiá `init`, agregá reglas nuevas a
`RULES` (siempre y cuando sigan siendo monótonas: uniones, intersecciones
de conjuntos que solo crecen, o filtros por un predicado fijo), o
aumentá `random.shuffle` para probar más permutaciones de orden.
"""

import random


# ---------------------------------------------------------------------
# Universo finito sobre el que viven los conjuntos (elementos 0..11)
# ---------------------------------------------------------------------
UNIVERSE = list(range(12))


def rule_union(sets):
    """Unión de conjuntos. Trivialmente monótona: si crecen los
    argumentos, crece la unión."""
    out = set()
    for s in sets:
        out |= s
    return out


def rule_intersection(sets):
    """Intersección de DOS conjuntos que solo pueden crecer con el
    tiempo. Ojo: la intersección NO es monótona en general (si un
    conjunto se achica, la intersección puede reducirse), pero en este
    sistema los conjuntos nunca se achican (solo se les agregan
    elementos, nunca se les quitan), así que el operador
    (A, B) -> A ∩ B es monótono como función del PAR (A, B) bajo el
    orden producto, restringido a trayectorias crecientes. Esa es
    precisamente la hipótesis que garantiza el Teorema de
    Knaster-Tarski aquí."""
    a, b = sets
    return a & b


def rule_even_filter(sets):
    """Filtrar los elementos pares de un conjunto que solo crece.
    Un filtro por un predicado fijo es monótono: agrandar el conjunto
    de entrada nunca puede hacer perder elementos que ya cumplían el
    predicado."""
    (d,) = sets
    return {x for x in d if x % 2 == 0}


# Cada regla es (índice_destino, índices_fuente, combinador monótono).
# La actualización siempre es  estado[destino] |= combinador(fuentes),
# es decir: SOLO se agregan elementos, nunca se quitan.
RULES = [
    (2, [0, 1], rule_intersection),  # C ⊇ A ∩ B
    (3, [2, 0], rule_union),         # D ⊇ C ∪ A
    (1, [3], rule_even_filter),      # B ⊇ pares de D
]


def sweep(state, order):
    """Una pasada: aplica todas las reglas, en el orden dado, sobre el
    estado actual. Devuelve True si algo cambió."""
    changed = False
    for idx in order:
        target, sources, combinator = RULES[idx]
        add = combinator([state[s] for s in sources])
        if not add <= state[target]:
            state[target] = state[target] | add
            changed = True
    return changed


def run_to_fixpoint(init, order, max_sweeps=10_000):
    """Itera `sweep` hasta que el estado deje de cambiar.

    Termina siempre porque:
      1. El universo es finito (12 elementos) y hay 4 variables, así
         que la 'altura' del retículo producto es finita
         (12 * 4 = 48 elementos como máximo que se pueden agregar).
      2. Cada pasada que cambia algo agrega AL MENOS un elemento a
         AL MENOS una variable (nunca quita), así que la secuencia de
         estados es estrictamente creciente hasta el punto fijo.
      Esto es la construcción explícita del punto fijo mínimo del
      Teorema de Knaster-Tarski (vía la versión constructiva /
      Teorema de Kleene para retículos completos finitos).
    """
    state = [set(s) for s in init]
    sweeps = 0
    while True:
        changed = sweep(state, order)
        sweeps += 1
        if not changed:
            break
        if sweeps > max_sweeps:
            raise RuntimeError("no convergio dentro del limite de seguridad")
    return state, sweeps


def main():
    init = [{0, 1, 2}, set(), set(), set()]  # A={0,1,2}; B,C,D vacíos

    base_orders = [
        [0, 1, 2],
        [2, 1, 0],
        [1, 0, 2],
        [2, 0, 1],
    ]

    results = []
    for order in base_orders:
        order = list(order)
        random.shuffle(order)  # perturbación adicional del orden
        final_state, sweeps = run_to_fixpoint(init, order)
        print(f"orden {order} -> punto fijo {final_state}  ({sweeps} pasadas)")
        results.append(final_state)

    first = results[0]
    all_equal = all(r == first for r in results)
    print()
    print(
        "¿Todas las trayectorias colapsan al MISMO punto fijo, "
        f"sin importar el orden de evaluación? {all_equal}"
    )
    assert all_equal, (
        "El sistema NO fue orden-independiente: revisar que todas las "
        "reglas sean genuinamente monótonas."
    )


if __name__ == "__main__":
    main()
