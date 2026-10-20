"""
tarea_53.py
Resolucion de restricciones simbolicas sobre la inicializacion de pesos de una red
usando el SMT solver Z3.

Este script intenta encontrar (o demostrar la infactibilidad de) una matriz de pesos
W que satisfaga simultaneamente tres restricciones tipicas de esquemas de
inicializacion de redes neuronales:

  1. Preservacion de varianza (estilo Xavier/Glorot, Glorot & Bengio 2010):
     la energia total de la matriz (suma de cuadrados) iguala un valor objetivo
     determinado por fan_in y fan_out.
  2. Cuasi-ortogonalidad de columnas (estilo Saxe, McClelland & Ganguli 2013,
     "dynamical isometry"): W^T W se aproxima a la identidad dentro de una
     tolerancia epsilon.
  3. Cota por entrada: cada peso esta acotado en magnitud por un valor b.

Se corre para dos tamanos de matriz (2x2 y 3x3) para mostrar como escala
(o deja de escalar) el costo del solver SMT no lineal (nlsat) de Z3 al
aumentar la dimension -- esto es lo que el enunciado pide "demostrar".

Dependencias:
    pip install z3-solver

Uso:
    python tarea_53.py
"""

from z3 import Real, Solver, sat, unsat, Sum


def build_weight_csp(n_rows, n_cols, fan_in, fan_out, eps_orth=0.15, bound=1.5, timeout_ms=20000):
    """
    Construye y resuelve un CSP en Z3 para una matriz de pesos W de tamano (n_rows x n_cols).

    Restricciones:
      - Energia total (suma de cuadrados) == target_var * n_rows * n_cols
        target_var = 2 / (fan_in + fan_out)  (heuristica Glorot/Xavier)
      - Para cada columna i: | ||W_i||^2 - 1 | <= eps_orth   (norma unitaria aproximada)
      - Para cada par de columnas i != j: |<W_i, W_j>| <= eps_orth  (cuasi-ortogonalidad)
      - |W[r][c]| <= bound para todo r, c

    Devuelve (resultado, modelo_o_None, matriz_de_variables_simbolicas).
    """
    W = [[Real(f"w_{r}_{c}") for c in range(n_cols)] for r in range(n_rows)]

    s = Solver()
    s.set("timeout", timeout_ms)

    target_var = 2.0 / (fan_in + fan_out)
    total_energy = Sum([W[r][c] * W[r][c] for r in range(n_rows) for c in range(n_cols)])
    s.add(total_energy == target_var * n_rows * n_cols)

    for r in range(n_rows):
        for c in range(n_cols):
            s.add(W[r][c] <= bound, W[r][c] >= -bound)

    for i in range(n_cols):
        norm_i = Sum([W[r][i] * W[r][i] for r in range(n_rows)])
        s.add(norm_i - 1 <= eps_orth, 1 - norm_i <= eps_orth)
        for j in range(i + 1, n_cols):
            dot_ij = Sum([W[r][i] * W[r][j] for r in range(n_rows)])
            s.add(dot_ij <= eps_orth, dot_ij >= -eps_orth)

    result = s.check()
    model = s.model() if result == sat else None
    return result, model, W


def print_result(label, result, model, W):
    print(f"--- {label} ---")
    print(f"Estado del solver: {result}")
    if result == sat:
        for row in W:
            print([model.eval(w, model_completion=True) for w in row])
    elif result == unsat:
        print("El sistema de restricciones es infactible (unsat): no existe W que las satisfaga tal cual estan planteadas.")
    else:
        print("El solver no pudo decidir dentro del timeout (unknown) -- indicio de explosion combinatoria del nlsat al crecer la dimension.")
    print()


if __name__ == "__main__":
    # Caso 1: matriz pequena 2x2, fan_in = fan_out = 2
    r1, m1, w1 = build_weight_csp(n_rows=2, n_cols=2, fan_in=2, fan_out=2, timeout_ms=10000)
    print_result("Caso 2x2 (fan_in=2, fan_out=2)", r1, m1, w1)

    # Caso 2: matriz 3x3, fan_in = fan_out = 3 (para observar el escalado del costo del solver)
    r2, m2, w2 = build_weight_csp(n_rows=3, n_cols=3, fan_in=3, fan_out=3, timeout_ms=10000)
    print_result("Caso 3x3 (fan_in=3, fan_out=3)", r2, m2, w2)
