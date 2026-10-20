import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.linalg import svd

# Configuración de JAX para optimización
jax.config.update("jax_enable_x64", True)

# Función para generar un rotor Clifford aleatorio
def random_rotor(dim):
    theta = np.random.uniform(0, 2 * np.pi)
    v = np.random.randn(dim)
    v = v / np.linalg.norm(v)
    R = np.cos(theta / 2) + np.sin(theta / 2) * v
    return R

# Función para aplicar un rotor a un vector
def apply_rotor(v, R):
    return R * v * ~R

# Función para truncar SVD
def truncate_svd(A, k):
    U, S, Vh = svd(A)
    A_truncated = U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :]
    return A_truncated

# Función para aplicar una capa de red neuronal con rotor y truncamiento SVD
def rotor_layer(x, R, W, b, k):
    # Aplicar rotor
    x_rotated = apply_rotor(x, R)
    # Linealidad y bias
    x_linear = jnp.dot(W, x_rotated) + b
    # Truncamiento SVD
    x_truncated = truncate_svd(x_linear, k)
    return x_truncated

# Función de pérdida (ejemplo: MSE)
def loss_fn(params, x, y_true):
    R, W, b, k = params
    y_pred = rotor_layer(x, R, W, b, k)
    return jnp.mean((y_pred - y_true) ** 2)

# Derivada de la función de pérdida
grad_loss_fn = jax.grad(loss_fn)

# Ejemplo de uso
dim = 10
k = 5
x = jnp.ones(dim)
y_true = jnp.ones(dim)
R = random_rotor(dim)
W = jnp.eye(dim)
b = jnp.zeros(dim)

params = (R, W, b, k)
loss = loss_fn(params, x, y_true)
print(f"Loss: {loss}")

# Optimización usando gradiente descendente
learning_rate = 0.01
for _ in range(100):
    grads = grad_loss_fn(params, x, y_true)
    params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)

# Verificar la invariancia isométrica
x_rotated = apply_rotor(x, R)
x_rotated_norm = jnp.linalg.norm(x_rotated)
x_norm = jnp.linalg.norm(x)
print(f"Norm of x: {x_norm}")
print(f"Norm of x rotated: {x_rotated_norm}")