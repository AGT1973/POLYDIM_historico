"""
TAREA 34 - POLYDIM V2
Rol asignado: matematico numerico.

Enunciado (banco maestro, Test 34):
"Actua como matematico numerico. Integra la libreria Z3 (SMT solver) para demostrar
como resolver restricciones sobre un subespacio latente R^3 de forma estatica,
validando si pertenece a la criba de cobertura del topos."

Modelo (ver tarea_34.md para el desarrollo completo):

- Sitio: objeto U = caja [-2,2]^3 en R^3 (un "subespacio semantico" del sitio C
  de la Seccion 2.1 del paper).
- Familia candidata a cubrir U (criba de cobertura): dos bolas U1, U2, cada una
  representando un "patch" de validez local (p. ej. dos reglas de negocio
  compatibles con distintas franjas del espacio latente).
- Un subespacio Q ⊆ U ("subespacio latente estatico" a validar, tal como pide
  el enunciado) puede estar:
    (a) enteramente dentro de la criba (VALIDO globalmente),
    (b) enteramente fuera (INVALIDO globalmente, colapsa al fibrado vacio 0), o
    (c) parcialmente dentro y parcialmente fuera (MIXTO: ningun valor de verdad
        clasico 0/1 describe a Q completo; el "valor de verdad" correcto es el
        propio subobjeto abierto Q ∩ (U1 ∪ U2) ⊊ Q, que es exactamente la
        semantica de Omega en un topos con logica de Heyting, Seccion 2.2).

Este script decide, para tres subespacios Q1, Q2, Q3 de ejemplo, en cual de los
tres regimenes caen, usando Z3 para las dos consultas de satisfacibilidad
necesarias por cada Q:
    phi_out(Q) := exists x en Q. x NO en (U1 union U2)   ("hay un punto de Q fuera de la criba")
    phi_in (Q) := exists x en Q. x en (U1 union U2)      ("hay un punto de Q dentro de la criba")

Tabla de decision:
    UNSAT(phi_out) y SAT(phi_in)   -> Q subconjunto de la criba (valido global)
    SAT(phi_out)   y UNSAT(phi_in) -> Q interseccion criba = vacio (invalido global)
    SAT(phi_out)   y SAT(phi_in)   -> Q parcialmente cubierto (mixto / intuicionista)
"""

import z3

x, y, z = z3.Reals("x y z")


def in_box(lo, hi):
    return z3.And(x >= lo[0], x <= hi[0], y >= lo[1], y <= hi[1], z >= lo[2], z <= hi[2])


def in_ball(center, radius):
    cx, cy, cz = center
    return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius ** 2


# --- Sitio y familia candidata de cobertura -------------------------------
U_LO, U_HI = (-2, -2, -2), (2, 2, 2)
U = in_box(U_LO, U_HI)

C1, R1 = (-1, -1, -1), 1.5
C2, R2 = (1, 1, 1), 1.5
U1 = in_ball(C1, R1)
U2 = in_ball(C2, R2)
SIEVE = z3.Or(U1, U2)


def check_global_coverage():
    """¿{U1, U2} cubre efectivamente a U? (axioma de cobertura de Grothendieck:
    la union de la familia debe ser TODO el objeto U)."""
    s = z3.Solver()
    s.add(U, z3.Not(SIEVE))
    result = s.check()
    witness = s.model() if result == z3.sat else None
    return result, witness


def classify_subspace(name, Q):
    s_out = z3.Solver()
    s_out.add(Q, z3.Not(SIEVE))
    r_out = s_out.check()

    s_in = z3.Solver()
    s_in.add(Q, SIEVE)
    r_in = s_in.check()

    if r_out == z3.unsat and r_in == z3.sat:
        veredicto = "VALIDO GLOBAL (Q subconjunto de la criba)"
    elif r_out == z3.sat and r_in == z3.unsat:
        veredicto = "INVALIDO GLOBAL (Q interseccion criba = vacio, colapsa a 0)"
    elif r_out == z3.sat and r_in == z3.sat:
        veredicto = "MIXTO (subobjeto propio: ni valido ni invalido globalmente)"
    else:
        veredicto = "Q VACIO (caso degenerado, revisar definicion de Q)"

    print(f"[{name}] existe punto FUERA de la criba: {r_out}   "
          f"existe punto DENTRO de la criba: {r_in}")
    print(f"        -> {veredicto}\n")
    return veredicto


if __name__ == "__main__":
    print("=" * 70)
    print("1) ¿La familia {U1, U2} cubre efectivamente el sitio U (axioma de")
    print("   cobertura de Grothendieck: union de la familia = U)?")
    print("=" * 70)
    result, witness = check_global_coverage()
    if result == z3.sat:
        print(f"NO es un cubrimiento valido de U. Z3 encontro un punto de U fuera")
        print(f"de U1 union U2 (un estado que colapsaria al fibrado vacio 0): {witness}\n")
    else:
        print("SI es un cubrimiento valido de U (todo punto de U cae en U1 o U2).\n")

    print("=" * 70)
    print("2) Clasificacion de tres subespacios latentes ESTATICOS Q1, Q2, Q3")
    print("   respecto de la criba de cobertura {U1, U2}")
    print("=" * 70)

    Q1 = in_box((-1.2, -1.2, -1.2), (-0.8, -0.8, -0.8))   # centrado en el centro de U1
    Q2 = in_box((1.8, -2.0, 1.8), (2.0, -1.8, 2.0))        # esquina lejana de ambos centros
    Q3 = in_box((0.3, -1.1, -1.1), (0.7, -0.9, -0.9))      # a caballo del borde esferico de U1

    classify_subspace("Q1 (caja cerca del centro de U1)", Q1)
    classify_subspace("Q2 (esquina lejana de U)", Q2)
    classify_subspace("Q3 (caja que cruza el borde de U1)", Q3)
