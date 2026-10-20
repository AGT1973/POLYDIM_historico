"""
Tarea 2 (Escuadrón 1) - "Simular la superposición garantizando conservación
del volumen determinante"

Qué es esto realmente:
Un ROTOR (en álgebra de Clifford) es una rotación pura. Las rotaciones nunca
cambian el volumen de lo que rotan, solo su orientación. En 3D, los rotores
son matemáticamente equivalentes a los cuaterniones unitarios, así que los
usamos aquí (más simples de programar que bivectores de Clifford, mismo
resultado).

Criterio de "volumen conservado":
Si R es la matriz 3x3 que representa la rotación, debe cumplirse:
    det(R) = 1        (no hay compresión ni expansión)
    R^T R = I          (es una transformación ortogonal, ángulos preservados)
"""

import numpy as np


def quaternion_from_axis_angle(axis, theta):
    """Construye el cuaternión unitario (rotor) para rotar 'theta' radianes
    alrededor de 'axis'."""
    axis = axis / np.linalg.norm(axis)
    w = np.cos(theta / 2)
    xyz = axis * np.sin(theta / 2)
    return np.array([w, *xyz])


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def rotate_vector(v, q):
    """Aplica la rotación q al vector v mediante q * v * q^{-1}
    (la 'sandwich formula', el equivalente cuaterniónico de un rotor
    de Clifford aplicado por conjugación)."""
    qv = np.array([0.0, *v])
    q_conj = quat_conjugate(q)
    rotated = quat_mult(quat_mult(q, qv), q_conj)
    return rotated[1:]


def rotation_matrix_from_quaternion(q):
    """Construye la matriz 3x3 rotando la base canónica."""
    e1, e2, e3 = np.eye(3)
    cols = [rotate_vector(e, q) for e in (e1, e2, e3)]
    return np.column_stack(cols)


def demo():
    rng = np.random.default_rng(0)

    print("=== Rotores aleatorios: el volumen SIEMPRE se conserva ===")
    for i in range(5):
        axis = rng.normal(size=3)
        theta = rng.uniform(0, 2 * np.pi)
        q = quaternion_from_axis_angle(axis, theta)
        R = rotation_matrix_from_quaternion(q)
        det = np.linalg.det(R)
        ortho_error = np.linalg.norm(R.T @ R - np.eye(3))
        print(f"  prueba {i}: det(R) = {det:.6f} | error ortogonalidad = {ortho_error:.2e}")

    print()
    print("=== Contraejemplo: una mezcla NO-rotor (matriz densa cualquiera) ===")
    M = rng.normal(size=(3, 3))
    print(f"  det(M) = {np.linalg.det(M):.6f}  (no es 1, distorsiona volumen)")


if __name__ == "__main__":
    demo()
