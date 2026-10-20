"""
Tarea 17 - POLYDIM
Rol asignado: especialista en NVIDIA Triton / kernels GPU
Nucleo matematico: Producto Geometrico de un Algebra de Clifford truncada

Pregunta que investigamos (reconstruida a partir del enunciado incompleto del banco):
    Sea Cl(3,0) el algebra geometrica de R^3 (multivectores de dimension 8,
    grados 0..3: escalar, vector, bivector, trivector/pseudoescalar).
    Sea P_<=2 la proyeccion que anula la componente de grado 3 (pseudoescalar).
    Sea a (x)_T b := P_<=2(a (x) b) el "producto geometrico truncado"
    (lo que haria en la practica un kernel que descarta el grado mas alto
    despues de cada multiplicacion, por ahorro de memoria/computo).

    Pregunta: es (x)_T asociativo? Es decir, para todo a,b,c:
        (a (x)_T b) (x)_T c   ==   a (x)_T (b (x)_T c) ?

Resultado esperado (a confirmar corriendo este script vos mismo, ver abajo):
    NO en general. El producto SIN truncar (x) es asociativo (es un teorema:
    Cl(3,0) se construye como cociente del algebra tensorial, que es asociativa,
    por un ideal bilatero -- ver Hestenes & Sobczyk, "Clifford Algebra to
    Geometric Calculus", 1984; o Dorst, Fontijne & Mann, "Geometric Algebra
    for Computer Science", 2007). Pero la proyeccion P_<=2 es una operacion
    con perdida que en general NO conmuta con el reagrupamiento del producto,
    de la misma forma que redondear despues de cada suma parcial puede romper
    la asociatividad exacta de la suma de reales en una calculadora de punto
    flotante, aunque la suma "de fondo" sea asociativa.

Este script:
    1) Implementa el producto geometrico completo de Cl(3,0) via bitmasks
       (algoritmo estandar de "canonical reordering sign", ver p.ej. Fontijne,
       "Efficient Implementation of Geometric Algebra", 2007).
    2) Se auto-verifica contra identidades conocidas (e1*e1=1, e1*e2=e12,
       e2*e1=-e12, e123^2=-1) -- si estas aserciones fallan, el resto del
       script no vale, por eso van primero.
    3) Confirma numericamente que el producto SIN truncar es asociativo
       (control negativo).
    4) Da un contraejemplo EXACTO, a mano, de no-asociatividad truncada.
    5) Corre un muestreo Monte Carlo para cuantificar el tamano tipico del
       error de asociatividad bajo truncamiento (metrica: norma L2 de los
       8 coeficientes del multivector "asociador").

Como correrlo vos mismo (esto es la parte que te toca a vos, no yo):
    pip install numpy
    python3 tarea_17.py

No hace falta ninguna libreria mas alla de numpy. No se necesita GPU ni
Triton instalado para esta verificacion matematica -- Triton entra recien
cuando este mismo producto se lleva a un kernel real (ver nota final).
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1) Producto geometrico de Cl(3,0) via bitmasks
# ---------------------------------------------------------------------------
# Orden de la base (indice 0..7 <-> bitmask 0..7):
#   0: 1        (escalar,   grado 0)
#   1: e1       (vector,    grado 1)
#   2: e2       (vector,    grado 1)
#   3: e12      (bivector,  grado 2)
#   4: e3       (vector,    grado 1)
#   5: e13      (bivector,  grado 2)
#   6: e23      (bivector,  grado 2)
#   7: e123     (trivector, grado 3)
# (el bitmask de cada blade indica que vectores e1,e2,e3 -bits 0,1,2- participan)

N_BASIS = 8
GRADE_OF = [bin(i).count("1") for i in range(N_BASIS)]  # popcount = grado


def canonical_reordering_sign(a: int, b: int) -> int:
    """Signo por reordenar (bubble-sort) los indices de la blade b hasta
    intercalarlos con los de la blade a, en base Euclidea (e_i^2=+1).
    Algoritmo estandar en implementaciones de algebra geometrica."""
    a_shift = a >> 1
    total = 0
    while a_shift != 0:
        total += bin(a_shift & b).count("1")
        a_shift >>= 1
    return -1 if (total % 2) else 1


def blade_product(a: int, b: int):
    """Producto geometrico de dos blades (bitmasks). Metrica Euclidea (+1,+1,+1),
    por lo que los indices repetidos simplemente se cancelan (e_i^2=+1) sin
    signo adicional mas alla del de reordenamiento."""
    sign = canonical_reordering_sign(a, b)
    result_blade = a ^ b
    return sign, result_blade


# Tabla de multiplicacion precomputada: MUL_SIGN[i][j], MUL_BLADE[i][j]
MUL_SIGN = np.zeros((N_BASIS, N_BASIS), dtype=np.int64)
MUL_BLADE = np.zeros((N_BASIS, N_BASIS), dtype=np.int64)
for i in range(N_BASIS):
    for j in range(N_BASIS):
        s, blade = blade_product(i, j)
        MUL_SIGN[i, j] = s
        MUL_BLADE[i, j] = blade


def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Producto geometrico completo (SIN truncar) de dos multivectores
    representados como arrays de 8 coeficientes reales."""
    result = np.zeros(N_BASIS, dtype=np.float64)
    for i in range(N_BASIS):
        ai = a[i]
        if ai == 0.0:
            continue
        for j in range(N_BASIS):
            bj = b[j]
            if bj == 0.0:
                continue
            result[MUL_BLADE[i, j]] += MUL_SIGN[i, j] * ai * bj
    return result


def project_leq2(m: np.ndarray) -> np.ndarray:
    """Proyeccion P_<=2: anula la componente de grado 3 (indice 7, e123)."""
    out = m.copy()
    out[7] = 0.0
    return out


def truncated_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Producto geometrico truncado: a (x)_T b := P_<=2(a (x) b)."""
    return project_leq2(geometric_product(a, b))


def mv(**kwargs) -> np.ndarray:
    """Constructor de multivectores por nombre de blade, p.ej. mv(e1=1, e3=1)."""
    names = {"1": 0, "e1": 1, "e2": 2, "e12": 3, "e3": 4, "e13": 5, "e23": 6, "e123": 7}
    out = np.zeros(N_BASIS, dtype=np.float64)
    for k, v in kwargs.items():
        out[names[k]] = v
    return out


def l2(m: np.ndarray) -> float:
    return float(np.linalg.norm(m))


# ---------------------------------------------------------------------------
# 2) Auto-verificacion contra identidades conocidas de Cl(3,0)
#    (si algo de esto falla, el resto del script es invalido -- por eso
#    se corta con un assert antes de seguir)
# ---------------------------------------------------------------------------
def self_check():
    e1, e2, e3 = mv(e1=1), mv(e2=1), mv(e3=1)
    one = mv(**{"1": 1})
    e12 = mv(e12=1)
    e123 = mv(e123=1)

    assert np.allclose(geometric_product(e1, e1), one), "e1*e1 deberia ser 1"
    assert np.allclose(geometric_product(e1, e2), e12), "e1*e2 deberia ser e12"
    assert np.allclose(geometric_product(e2, e1), -e12), "e2*e1 deberia ser -e12"
    assert np.allclose(geometric_product(e12, e12), -one), "e12^2 deberia ser -1"
    assert np.allclose(geometric_product(e123, e123), -one), "e123^2 deberia ser -1"

    print("[OK] Identidades basicas de Cl(3,0) verificadas "
          "(e1^2=1, e1e2=e12, e2e1=-e12, e12^2=-1, e123^2=-1).")


# ---------------------------------------------------------------------------
# 3) Control negativo: el producto SIN truncar es asociativo (numericamente,
#    hasta el error de redondeo de punto flotante)
# ---------------------------------------------------------------------------
def check_full_product_is_associative(n_trials=200, seed=0):
    rng = np.random.default_rng(seed)
    max_err = 0.0
    for _ in range(n_trials):
        a, b, c = rng.normal(size=8), rng.normal(size=8), rng.normal(size=8)
        left = geometric_product(geometric_product(a, b), c)
        right = geometric_product(a, geometric_product(b, c))
        max_err = max(max_err, l2(left - right))
    print(f"[control] Producto SIN truncar: error de asociatividad maximo "
          f"sobre {n_trials} pruebas = {max_err:.2e} (deberia ser ~1e-12, "
          f"solo error de redondeo).")
    return max_err


# ---------------------------------------------------------------------------
# 4) Contraejemplo EXACTO de no-asociatividad truncada
#    A = e1 + e3 (vector con componente fuera del plano e12)
#    B = e12     (bivector)
#    C = e2      (vector, contenido en el plano de B)
# ---------------------------------------------------------------------------
def exact_counterexample():
    A = mv(e1=1, e3=1)
    B = mv(e12=1)
    C = mv(e2=1)

    left = truncated_product(truncated_product(A, B), C)
    right = truncated_product(A, truncated_product(B, C))
    associator = left - right

    print("[contraejemplo exacto]")
    print(f"  (A (x)_T B) (x)_T C = {left}")
    print(f"  A (x)_T (B (x)_T C) = {right}")
    print(f"  asociador (diferencia) = {associator}  "
          f"(componente 5 = coeficiente de e13)")
    print(f"  norma L2 del asociador = {l2(associator):.6f}  "
          f"(se espera exactamente 1.0, o sea el asociador es +-e13)")
    return associator


# ---------------------------------------------------------------------------
# 5) Muestreo Monte Carlo: tamano tipico del error de asociatividad truncada
# ---------------------------------------------------------------------------
def monte_carlo_truncated_associativity(n_trials=2000, seed=1):
    rng = np.random.default_rng(seed)
    errors = np.zeros(n_trials)
    for k in range(n_trials):
        a, b, c = rng.normal(size=8), rng.normal(size=8), rng.normal(size=8)
        left = truncated_product(truncated_product(a, b), c)
        right = truncated_product(a, truncated_product(b, c))
        errors[k] = l2(left - right)

    frac_nonzero = float(np.mean(errors > 1e-9))
    print(f"[monte carlo] n={n_trials} multivectores aleatorios (normal estandar, "
          f"las 8 componentes):")
    print(f"  fraccion de tripletas con asociador != 0 : {frac_nonzero:.3f}")
    print(f"  error medio                              : {errors.mean():.4f}")
    print(f"  error maximo                              : {errors.max():.4f}")
    print(f"  desvio estandar                           : {errors.std():.4f}")
    return errors


if __name__ == "__main__":
    self_check()
    print()
    check_full_product_is_associative()
    print()
    exact_counterexample()
    print()
    monte_carlo_truncated_associativity()
