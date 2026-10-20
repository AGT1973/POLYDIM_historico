"""
Tarea 28 - Verificacion empirica del producto geometrico en Cl(3,0)
y de la (no-)clausura de distintos truncamientos.

Representacion: un multivector es un array de 8 floats indexado por
una bitmask sobre {e1, e2, e3}:
    indice 0 = 1 (escalar)
    indice 1 = e1
    indice 2 = e2
    indice 3 = e1^e2
    indice 4 = e3
    indice 5 = e1^e3
    indice 6 = e2^e3
    indice 7 = e1^e2^e3 (pseudoescalar)

Se usa el algoritmo estandar de "bit-blade" para algebra geometrica
(ver Dorst, Fontijne, Mann - "Geometric Algebra for Computer Science",
cap. 21) para computar signo e indice resultante del producto de dos
blades base, valido para metrica euclidiana (e_i^2 = +1).
"""

import random

N = 8  # 2^3 componentes para Cl(3,0)


def popcount(x: int) -> int:
    return bin(x).count("1")


def grade(x: int) -> int:
    return popcount(x)


def blade_sign_and_index(a: int, b: int):
    """Signo e indice del producto geometrico entre dos blades base a, b."""
    cnt = 0
    a1 = a >> 1
    while a1:
        cnt += popcount(a1 & b)
        a1 >>= 1
    sign = -1 if (cnt % 2) else 1
    return sign, a ^ b


def geometric_product(A, B):
    """Producto geometrico completo entre dos multivectores densos (len 8)."""
    C = [0.0] * N
    for a in range(N):
        if A[a] == 0.0:
            continue
        for b in range(N):
            if B[b] == 0.0:
                continue
            sign, idx = blade_sign_and_index(a, b)
            C[idx] += sign * A[a] * B[b]
    return C


def zero_vec():
    return [0.0] * N


def basis(i, coeff=1.0):
    v = zero_vec()
    v[i] = coeff
    return v


def add(A, B):
    return [x + y for x, y in zip(A, B)]


# --------------------------------------------------------------------------
# 1) Sanity check de asociatividad: (e1 e2) e3 == e1 (e2 e3)
# --------------------------------------------------------------------------
def check_associativity():
    lhs = geometric_product(geometric_product(basis(1), basis(2)), basis(4))
    rhs = geometric_product(basis(1), geometric_product(basis(2), basis(4)))
    assert lhs == rhs, f"Asociatividad violada: {lhs} != {rhs}"
    print("[OK] Asociatividad del producto geometrico verificada: (e1 e2) e3 == e1 (e2 e3) =", lhs)


# --------------------------------------------------------------------------
# 2) Contraejemplo: el truncamiento "grado <= 2" (descartar solo el
#    pseudoescalar, indice 7) NO es cerrado bajo el producto geometrico.
# --------------------------------------------------------------------------
def check_grade_leq_2_not_closed():
    e12 = basis(3)  # e1^e2
    e3 = basis(4)
    prod = geometric_product(e12, e3)
    leaked_idx = [i for i, c in enumerate(prod) if c != 0.0]
    assert leaked_idx == [7], f"Se esperaba fuga solo en el pseudoescalar, se obtuvo {leaked_idx}"
    print(f"[OK] e12 * e3 = {prod} -> produce componente en indice 7 (grado 3, pseudoescalar).")
    print("     Esto confirma que el subespacio grado<=2 (indices 0..6) NO es cerrado bajo el producto geometrico.")


# --------------------------------------------------------------------------
# 3) La subalgebra PAR (grados 0 y 2: indices {0, 3, 5, 6}) SI es cerrada.
#    Se verifica empiricamente con muestreo aleatorio y ademas se da el
#    argumento de paridad que lo demuestra en general.
# --------------------------------------------------------------------------
EVEN_INDICES = [0, 3, 5, 6]
ODD_OR_TRIVECTOR_INDICES = [1, 2, 4, 7]


def random_even_multivector(rng):
    v = zero_vec()
    for i in EVEN_INDICES:
        v[i] = rng.uniform(-5, 5)
    return v


def check_even_subalgebra_closed(trials=2000, seed=42):
    rng = random.Random(seed)
    max_leak = 0.0
    for _ in range(trials):
        A = random_even_multivector(rng)
        B = random_even_multivector(rng)
        C = geometric_product(A, B)
        leak = max(abs(C[i]) for i in ODD_OR_TRIVECTOR_INDICES)
        max_leak = max(max_leak, leak)
    assert max_leak < 1e-9, f"Fuga inesperada fuera de la subalgebra par: {max_leak}"
    print(f"[OK] {trials} productos aleatorios en la subalgebra par -> fuga maxima fuera de {{0,3,5,6}}: {max_leak:.3e}")
    print("     Argumento general: popcount(a xor b) tiene la misma paridad que popcount(a)+popcount(b);")
    print("     si ambos son pares, el resultado tiene popcount par (grado 0 o 2). QED (para n=3).")


if __name__ == "__main__":
    check_associativity()
    check_grade_leq_2_not_closed()
    check_even_subalgebra_closed()
    print("\nTodas las verificaciones pasaron.")
