"""
TAREA 10 - Enjambre POLYDIM
Restricciones O-Minimales via SMT (Z3) sobre subespacio latente R^3.

Objetivo:
  1) Modelar una "criba de cobertura" (covering sieve) sobre una region
     acotada de R^3 como una union finita de conjuntos semialgebraicos
     (definidos por polinomios, decidibles por o-minimalidad / Tarski-Seidenberg).
  2) Verificar, de forma ESTATICA (sin muestreo, sin ejecutar tensores),
     si un estado concreto p in R^3 pertenece a la criba.
  3) Verificar, de forma ESTATICA y EXHAUSTIVA (cuantificador universal),
     si la criba efectivamente CUBRE toda la region acotada declarada,
     o si existe un "agujero" (contraejemplo) donde ningun abierto aplica.
"""

from z3 import Reals, Solver, Or, And, Not, sat, unsat


# 1. Variables del subespacio latente R^3
x, y, z = Reals("x y z")

# 2. Region acotada de referencia: caja [-1, 1]^3
region = And(x >= -1, x <= 1, y >= -1, y <= 1, z >= -1, z <= 1)

# 3. Conjuntos semialgebraicos (abiertos U_i de la criba)
U1 = x**2 + y**2 + z**2 <= 0.25        # bola pequena (radio 0.5)
U2 = x**2 + y**2 + z**2 >= 0.75        # complemento de una bola mayor
U3 = (x - y) ** 2 <= 0.01              # franja delgada |x - y| <= 0.1

sieve = [U1, U2, U3]


def criba_contiene(punto):
    """Chequeo ESTATICO (SMT) de pertenencia de un estado a la criba: se
    resuelve como restriccion logica (punto == p AND OR(U_i)), no como
    aritmetica de punto flotante."""
    px, py, pz = punto
    s = Solver()
    s.add(x == px, y == py, z == pz)
    s.add(Or(*sieve))
    return s.check() == sat


def criba_cubre_la_region():
    """Verificacion FORMAL y EXHAUSTIVA: la criba cubre 'region' si y solo
    si NO existe (x,y,z) en region tal que ningun U_i se cumple. Se prueba
    la UNSAT de la negacion (refutacion), forma canonica de verificacion
    en teorias de cuerpos reales cerrados, decidibles por Tarski-Seidenberg.

    Con esta configuracion existe deliberadamente un hueco real en la
    region anular 0.25 < norma^2 < 0.75 fuera de la franja U3, para forzar
    a Z3 a encontrar un contraejemplo real en vez de una cobertura trivial.
    """
    s = Solver()
    s.add(And(region, Not(Or(*sieve))))
    resultado = s.check()
    if resultado == unsat:
        return True, None
    modelo = s.model()
    return False, (modelo[x], modelo[y], modelo[z])


def main():
    estado_a = (0.1, 0.1, 0.1)
    estado_b = (0.5, -0.5, 0.0)
    pertenece_a = criba_contiene(estado_a)
    pertenece_b = criba_contiene(estado_b)
    cubre, contraejemplo = criba_cubre_la_region()
    
    print("--- RESULTADOS SMT (Z3) ---")
    print(f"Estado A {estado_a} pertenece a la criba: {pertenece_a}")
    print(f"Estado B {estado_b} pertenece a la criba: {pertenece_b}")
    print(f"Cobertura total de la region: {cubre}")
    if not cubre:
        print(f"Contraejemplo (Hueco Topologico) hallado por Z3: {contraejemplo}")
        
    return {
        "estado_a_pertenece": pertenece_a,
        "estado_b_pertenece": pertenece_b,
        "cobertura_total": cubre,
        "contraejemplo": None if cubre else tuple(str(c) for c in contraejemplo),
    }


if __name__ == "__main__":
    main()
