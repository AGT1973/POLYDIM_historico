import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import numpy as np
import scipy.linalg
import torch

# Función para generar una matriz de rotación 2D
def rotation_matrix(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                      [jnp.sin(theta), jnp.cos(theta)]])

# Función para aplicar una isometría a un punto
def apply_isometry(x, R, t):
    return R @ x + t

# Capa neuronal equivariante a isometrías
def equivariant_layer(x, W, b):
    return jnp.dot(W, x) + b

# Función de pérdida
def loss_fn(params, x, y):
    W, b = params
    y_pred = equivariant_layer(x, W, b)
    return jnp.mean((y_pred - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = grad(loss_fn)

# Parámetros iniciales
W = jnp.array([[1.0, 0.0], [0.0, 1.0]])
b = jnp.array([0.0, 0.0])
params = (W, b)

# Datos de entrenamiento
x = jnp.array([1.0, 0.0])
y = jnp.array([1.0, 0.0])

# Rotación y traslación
theta = jnp.pi / 4
R = rotation_matrix(theta)
t = jnp.array([0.5, 0.5])
x_transformed = apply_isometry(x, R, t)
y_transformed = apply_isometry(y, R, t)

# Validación empírica
loss_before = loss_fn(params, x, y)
loss_after = loss_fn(params, x_transformed, y_transformed)

# Gradiente antes y después de la transformación
grad_before = grad_loss(params, x, y)
grad_after = grad_loss(params, x_transformed, y_transformed)

print(f"Loss before transformation: {loss_before}")
print(f"Loss after transformation: {loss_after}")
print(f"Gradient before transformation: {grad_before}")
print(f"Gradient after transformation: {grad_after}")