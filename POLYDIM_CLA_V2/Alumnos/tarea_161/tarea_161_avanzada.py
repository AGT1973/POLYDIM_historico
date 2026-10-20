import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import numpy as np
from scipy.linalg import svd

# Función para crear un rotor de Clifford
def create_rotor(angle, bivector):
    B = bivector * angle
    R = jnp.exp(-B / 2)
    return R

# Función para aplicar el rotor a un vector
def apply_rotor(R, v):
    return R @ v @ jnp.linalg.inv(R)

# Función para truncar un tensor usando SVD
def truncate_svd(T, k):
    U, S, Vt = svd(T)
    S_truncated = S[:k]
    U_truncated = U[:, :k]
    Vt_truncated = Vt[:k, :]
    T_truncated = U_truncated @ jnp.diag(S_truncated) @ Vt_truncated
    return T_truncated

# Ejemplo de uso
def main():
    # Definir un vector y un bivector
    v = jnp.array([1.0, 0.0, 0.0])
    bivector = jnp.array([[0, 1, 0],
                          [-1, 0, 0],
                          [0, 0, 0]])

    # Crear un rotor
    angle = jnp.pi / 4
    R = create_rotor(angle, bivector)

    # Aplicar el rotor al vector
    v_transformed = apply_rotor(R, v)

    # Verificar la invariancia isométrica
    norm_v = jnp.linalg.norm(v)
    norm_v_transformed = jnp.linalg.norm(v_transformed)

    print(f"Norma original: {norm_v}")
    print(f"Norma transformada: {norm_v_transformed}")

    # Crear un tensor de bajo rango
    T = jnp.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

    # Truncar el tensor usando SVD
    k = 2
    T_truncated = truncate_svd(T, k)

    print(f"Tensor original: \n{T}")
    print(f"Tensor truncado: \n{T_truncated}")

if __name__ == "__main__":
    main()