"""
Álgebra Geométrica Cl(3,0) construida desde cero (sin librerías externas de GA)
para verificar, con matemática real, qué es y qué hace un ROTOR de Clifford.

Representación: cada "blade" (elemento base) se codifica como un bitmask de 3 bits.
  0b000 = 1        (escalar)
  0b001 = e1
  0b010 = e2
  0b100 = e3
  0b011 = e1^e2 = e12
  0b101 = e1^e3 = e13
  0b110 = e2^e3 = e23
  0b111 = e1^e2^e3 = e123 (pseudoescalar)

Un multivector es un dict {bitmask: coeficiente}.
Métrica euclídea: e_i^2 = +1 para todo i (esto es Cl(3,0)).

Nota para el grupo: este material NO sigue el formato "worker/JSON silencioso"
de swarm_context_200.txt. El código de abajo fue ejecutado en un intérprete real
antes de subirlo; ver tarea_10_rotores_clifford.md para la explicación completa,
el diagnóstico de qué afirmaciones de POLYDIM se sostienen y cuáles no, y los
15 pilares matemáticos del paper explicados con rigor.
"""

import numpy as np
from itertools import product

# ---------- 1. Producto geométrico de dos "blades" (núcleo del álgebra) ----------

def blade_sign(a: int, b: int) -> int:
    """Signo por reordenamiento canónico al multiplicar blade a * blade b."""
    a >>= 1
    sign = 1
    while a:
        sign *= (-1) ** bin(a & b).count("1")
        a >>= 1
    return sign

def blade_multiply(a: int, b: int):
    """Devuelve (blade_resultante, signo) del producto geométrico de dos blades base."""
    sign = blade_sign(a, b)
    # bits comunes a & b se contraen vía e_i^2 = +1 (métrica euclídea) -> no cambian el signo
    result = a ^ b
    return result, sign

def mv_multiply(A: dict, B: dict) -> dict:
    """Producto geométrico de dos multivectores generales."""
    result = {}
    for ba, ca in A.items():
        for bb, cb in B.items():
            blade, sign = blade_multiply(ba, bb)
            result[blade] = result.get(blade, 0.0) + sign * ca * cb
    return {k: v for k, v in result.items() if abs(v) > 1e-12}

def mv_reverse(A: dict) -> dict:
    """Reversión (~A): invierte el orden de los vectores en cada blade.
    Para un k-blade, el signo es (-1)^(k(k-1)/2)."""
    def grade(bitmask):
        return bin(bitmask).count("1")
    out = {}
    for b, c in A.items():
        k = grade(b)
        sign = (-1) ** (k * (k - 1) // 2)
        out[b] = sign * c
    return out

def mv_add(A: dict, B: dict, scalar_b=1.0) -> dict:
    out = dict(A)
    for b, c in B.items():
        out[b] = out.get(b, 0.0) + scalar_b * c
    return {k: v for k, v in out.items() if abs(v) > 1e-12}

def mv_scale(A: dict, s: float) -> dict:
    return {k: v * s for k, v in A.items()}

def vec(x, y, z) -> dict:
    """Construye un vector (1-blade) e1,e2,e3."""
    return {1: x, 2: y, 4: z}

def mv_norm(A: dict) -> float:
    """Norma para un vector puro: sqrt(v * ~v) tomando la parte escalar."""
    prod = mv_multiply(A, mv_reverse(A))
    return np.sqrt(prod.get(0, 0.0))

# ---------- 2. El ROTOR: R = exp(-theta/2 * B), B = bivector unitario del plano de rotación ----------

def bivector_e12(theta_half):
    """Para un bivector B con B^2 = -1 (como e12, e13, e23 en Cl(3,0)), la exponencial
    tiene forma cerrada: exp(theta * B) = cos(theta) + sin(theta) * B (idéntico a Euler,
    pero con B en vez de i)."""
    return {0: np.cos(theta_half), 3: np.sin(theta_half)}  # 3 = e12

def rotor_e12(theta):
    """Rotor que rota en el plano e1-e2 un ángulo theta."""
    half = theta / 2.0
    return {0: np.cos(half), 3: -np.sin(half)}  # R = exp(-theta/2 * e12)

def apply_rotor(R: dict, v: dict) -> dict:
    """Rotación sandwich: v' = R v R~ (esto es la operación real, no una metáfora)."""
    Rr = mv_reverse(R)
    return mv_multiply(mv_multiply(R, v), Rr)

# ---------- 3. VERIFICACIONES (afirmaciones que SÍ se pueden probar) ----------

if __name__ == "__main__":
    print("=== TEST 1: El rotor preserva la norma del vector (esto es real y demostrable) ===")
    v = vec(1.0, 0.0, 0.0)
    for theta_deg in [0, 30, 90, 137, 350]:
        theta = np.radians(theta_deg)
        R = rotor_e12(theta)
        v_rot = apply_rotor(R, v)
        norm_before = mv_norm(v)
        norm_after = mv_norm(v_rot)
        print(f"  theta={theta_deg:>4}°  |v|={norm_before:.6f}  |Rv R~|={norm_after:.6f}  "
              f"preservada={np.isclose(norm_before, norm_after)}")

    print("\n=== TEST 2: El rotor de Clifford coincide EXACTAMENTE con la matriz de rotación estándar ===")
    theta = np.radians(53.0)
    R = rotor_e12(theta)
    v = vec(0.7, 0.3, 0.0)
    v_rot_clifford = apply_rotor(R, v)

    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    v_arr = np.array([0.7, 0.3, 0.0])
    v_rot_matrix = rot_matrix @ v_arr

    v_rot_clifford_arr = np.array([v_rot_clifford.get(1, 0.0), v_rot_clifford.get(2, 0.0), v_rot_clifford.get(4, 0.0)])
    print(f"  Rotor de Clifford: {v_rot_clifford_arr}")
    print(f"  Matriz R_z clásica: {v_rot_matrix}")
    print(f"  Coinciden: {np.allclose(v_rot_clifford_arr, v_rot_matrix)}")

    print("\n=== TEST 3: Composición de rotores = rotor del ángulo compuesto (esto SÍ está probado) ===")
    theta1, theta2 = np.radians(40), np.radians(65)
    R1 = rotor_e12(theta1)
    R2 = rotor_e12(theta2)
    R_composed = mv_multiply(R2, R1)
    R_direct = rotor_e12(theta1 + theta2)
    print(f"  R2*R1 = {R_composed}")
    print(f"  R(theta1+theta2) = {R_direct}")
    match = all(np.isclose(R_composed.get(k, 0.0), R_direct.get(k, 0.0)) for k in set(R_composed) | set(R_direct))
    print(f"  Coinciden: {match}")

    print("\n=== TEST 4: R * R~ = 1 (condición de que el rotor es una rotación pura, no un escalado) ===")
    theta = np.radians(77)
    R = rotor_e12(theta)
    Rr = mv_reverse(R)
    identity_check = mv_multiply(R, Rr)
    print(f"  R * R~ = {identity_check}  (debe ser {{0: 1.0}})")

    print("\n=== TEST 5: ¿Qué pasa en dimensión alta? (Cl(4,0): plano e1-e2 rotando, e3,e4 fijos) ===")
    def vec4(x, y, z, w):
        return {1: x, 2: y, 4: z, 8: w}

    v4 = vec4(1.0, 0.0, 2.0, -1.0)
    theta = np.radians(90)
    R4 = rotor_e12(theta)
    v4_rot = apply_rotor(R4, v4)
    print(f"  v4 original: e1=1.0, e2=0.0, e3=2.0, e4=-1.0")
    print(f"  v4 rotado (plano e1-e2, 90°): {v4_rot}")
    print("  -> e3 y e4 quedan INTACTOS (2.0 y -1.0): el rotor de Clifford SÍ rota un plano específico")
    print("     sin afectar los demás ejes, sin singularidades. ESTO es lo que realmente se puede probar.")
    print("     Lo que el paper POLYDIM agrega ('evita el gimbal lock de alta dimensionalidad como concepto")
    print("     general del sistema') NO tiene una definición formal de qué sería 'gimbal lock' en Cl(n>3),")
    print("     así que no es una proposición demostrable ni falsable tal como está escrita.")
