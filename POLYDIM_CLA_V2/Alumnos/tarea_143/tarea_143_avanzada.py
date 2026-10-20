import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
from scipy.linalg import svd

# Función para crear un rotor de Clifford
def create_rotor(theta, bivector):
    rotor = jnp.exp(0.5 * theta * bivector)
    return rotor

# Función para aplicar un rotor de Clifford a un vector
def apply_rotor(rotor, vector):
    return rotor * vector * rotor.T

# Función para truncar el rango de un tensor usando SVD
def truncate_svd(tensor, rank):
    U, S, Vh = svd(tensor)
    truncated_tensor = U[:, :rank] @ jnp.diag(S[:rank]) @ Vh[:rank, :]
    return truncated_tensor

# Función para calcular la norma L2 de un tensor
def l2_norm(tensor):
    return jnp.sqrt(jnp.sum(tensor ** 2))

# Función para calcular la invariancia isométrica
def is_isometric(T, x, y):
    return jnp.isclose(l2_norm(T(x) - T(y)), l2_norm(x - y))

# Ejemplo de uso
theta = 1.0  # Ángulo de rotación
bivector = jnp.array([[0, -1], [1, 0]])  # Bivector unitario

# Crear un rotor de Clifford
rotor = create_rotor(theta, bivector)

# Definir una transformación isométrica
def isometric_transform(vector):
    return apply_rotor(rotor, vector)

# Definir vectores de prueba
x = jnp.array([1.0, 0.0])
y = jnp.array([0.0, 1.0])

# Verificar invariancia isométrica
isometric_result = is_isometric(isometric_transform, x, y)
print(f"Invariancia isométrica: {isometric_result}")

# Ejemplo de truncamiento SVD
tensor = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
truncated_tensor = truncate_svd(tensor, rank=2)
print(f"Tensor truncado: {truncated_tensor}")