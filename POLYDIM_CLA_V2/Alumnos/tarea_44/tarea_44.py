"""
Tarea 44 - POLYDIM V2
Rol: Tester de robustez de IA.

Verifica empiricamente:
  (A) Conservacion del volumen (determinante) de un conjunto de vectores
      latentes al aplicarles un Rotor de Spin (Clifford / cuaternion unitario).
  (B) Si la asociatividad de la composicion de rotores ("restriccion
      monoidal estricta") se sostiene de forma exacta o solo aproximada
      en aritmetica de punto flotante (float64).

No requiere librerias externas de algebra geometrica: el producto de
Hamilton (cuaterniones) se implementa a mano con NumPy para que el
razonamiento algebraico sea visible en el codigo, sin caja negra.
"""

import numpy as np


class Quaternion:
    """Cuaternion q = w + xi + yj + zk. Representa un rotor cuando ||q|| = 1."""

    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w, x, y, z):
        self.w, self.x, self.y, self.z = w, x, y, z

    def as_array(self):
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return float(np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2))

    def normalized(self):
        n = self.norm()
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def mul(self, other):
        """Producto de Hamilton q1 * q2 (NO conmutativo, SI asociativo por axioma)."""
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def __repr__(self):
        return f"Quaternion({self.w:.6f}, {self.x:.6f}, {self.y:.6f}, {self.z:.6f})"


def rotor_from_axis_angle(axis, theta):
    """Construye un rotor unitario R = cos(theta/2) + sin(theta/2) * B_hat."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = theta / 2.0
    s = np.sin(half)
    return Quaternion(np.cos(half), s * axis[0], s * axis[1], s * axis[2])


def vector_to_pure_quaternion(v):
    return Quaternion(0.0, v[0], v[1], v[2])


def rotate(q, v):
    """Producto sandwich: v' = q * v * q^-1 (q unitario => q^-1 = conjugado)."""
    vq = vector_to_pure_quaternion(v)
    result = q.mul(vq).mul(q.conjugate())
    return np.array([result.x, result.y, result.z], dtype=np.float64)


def compose(q2, q1):
    """Composicion de rotores: aplicar primero q1, despues q2 <=> cuaternion q2*q1."""
    return q2.mul(q1)


# ---------------------------------------------------------------------------
# TEST A: conservacion del volumen (determinante) bajo la accion del rotor
# ---------------------------------------------------------------------------
def test_volumen(n_trials=200, seed=42):
    rng = np.random.default_rng(seed)
    max_error = 0.0
    errores = []

    for _ in range(n_trials):
        # Tres vectores latentes aleatorios -> forman un paralelepipedo
        v1, v2, v3 = rng.normal(size=(3, 3))
        vol_original = np.linalg.det(np.column_stack([v1, v2, v3]))

        # Rotor aleatorio
        axis = rng.normal(size=3)
        theta = rng.uniform(0, 2 * np.pi)
        q = rotor_from_axis_angle(axis, theta)

        v1p, v2p, v3p = rotate(q, v1), rotate(q, v2), rotate(q, v3)
        vol_rotado = np.linalg.det(np.column_stack([v1p, v2p, v3p]))

        error = abs(vol_rotado - vol_original)
        errores.append(error)
        max_error = max(max_error, error)

    return {
        "n_trials": n_trials,
        "error_maximo": max_error,
        "error_promedio": float(np.mean(errores)),
    }


# ---------------------------------------------------------------------------
# TEST B: asociatividad estricta de la composicion de rotores (restriccion
# monoidal). Comparamos aplicar rotores en secuencia vs. componer primero
# los cuaterniones y aplicar una sola vez.
# ---------------------------------------------------------------------------
def test_asociatividad(n_trials=200, n_rotores_compuestos=5, seed=7):
    rng = np.random.default_rng(seed)
    max_residuo = 0.0
    residuos = []

    for _ in range(n_trials):
        v = rng.normal(size=3)

        rotores = []
        for _ in range(n_rotores_compuestos):
            axis = rng.normal(size=3)
            theta = rng.uniform(0, 2 * np.pi)
            rotores.append(rotor_from_axis_angle(axis, theta))

        # Camino 1: aplicar rotores uno por uno en secuencia sobre el vector
        v_secuencial = v.copy()
        for q in rotores:
            v_secuencial = rotate(q, v_secuencial)

        # Camino 2: componer todos los cuaterniones primero (producto de
        # Hamilton acumulado), y aplicar el rotor compuesto una sola vez.
        q_compuesto = rotores[0]
        for q in rotores[1:]:
            q_compuesto = compose(q, q_compuesto)
        v_compuesto = rotate(q_compuesto, v)

        residuo = float(np.linalg.norm(v_secuencial - v_compuesto))
        residuos.append(residuo)
        max_residuo = max(max_residuo, residuo)

    return {
        "n_trials": n_trials,
        "n_rotores_compuestos": n_rotores_compuestos,
        "residuo_maximo": max_residuo,
        "residuo_promedio": float(np.mean(residuos)),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("TEST A: Conservacion del volumen (determinante) bajo rotor")
    print("=" * 70)
    resultado_a = test_volumen()
    for k, v in resultado_a.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 70)
    print("TEST B: Asociatividad estricta de la composicion de rotores")
    print("=" * 70)
    resultado_b = test_asociatividad()
    for k, v in resultado_b.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 70)
    print("INTERPRETACION (no reemplaza tu propia lectura de los numeros)")
    print("=" * 70)
    print(
        "- El error del Test A deberia ser muy pequeno (orden 1e-14 o menor): "
        "confirma el teorema de que los rotores preservan el determinante."
    )
    print(
        "- El residuo del Test B NO deberia ser exactamente 0.0 en la mayoria "
        "de las corridas: eso evidencia que la asociatividad, exacta a nivel "
        "algebraico, no es bit-exacta en punto flotante."
    )
    print(
        "- Probá variar 'n_rotores_compuestos' (por ejemplo a 20 o 50) y "
        "fijate si el residuo_maximo crece: esa es la pregunta abierta que "
        "te toca comprobar a vos."
    )
