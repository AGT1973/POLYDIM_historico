import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import grad, jit, vmap
from scipy.linalg import expm

# Definir una red neuronal simple en JAX
def simple_nn(x):
    w1 = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    b1 = jnp.array([0.1, 0.1])
    w2 = jnp.array([[1.0], [1.0]])
    b2 = jnp.array([0.1])
    h1 = jax.nn.relu(jnp.dot(x, w1) + b1)
    return jnp.dot(h1, w2) + b2

# Definir una transformación de rotación en 2D (SO(2))
def so2_rotation(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                      [jnp.sin(theta), jnp.cos(theta)]])

# Definir una capa de pooling invariante (media)
def invariant_pooling(x):
    return jnp.mean(x, axis=0)

# Función para validar la invariancia
def validate_invariance(x, theta):
    x_rotated = jnp.dot(so2_rotation(theta), x)
    y1 = invariant_pooling(simple_nn(x))
    y2 = invariant_pooling(simple_nn(x_rotated))
    return jnp.allclose(y1, y2, atol=1e-6)

# Ejecutar la validación
x = jnp.array([1.0, 1.0])
theta = jnp.pi / 4  # 45 grados
invariance_result = validate_invariance(x, theta)

print(f"Invariancia validada: {invariance_result}")