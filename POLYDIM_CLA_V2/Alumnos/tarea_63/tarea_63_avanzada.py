import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
import scipy.optimize
import torch

# Definición del grupo de Lie SO(2) (rotaciones en 2D)
def so2_action(theta, x):
    """Aplica una rotación en 2D a un vector x."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    R = jnp.array([[c, -s], [s, c]])
    return R @ x

# Definición de una función invariante a SO(2)
def invariant_function(x):
    """Función invariante a rotaciones en 2D."""
    return jnp.linalg.norm(x)  # Norma Euclidiana

# Verificación de la invariancia
def check_invariance(theta, x):
    rotated_x = so2_action(theta, x)
    return invariant_function(rotated_x) == invariant_function(x)

# Función de red neuronal simple invariante a SO(2)
def invariant_neural_network(x, params):
    """Red neuronal invariante a rotaciones en 2D."""
    w1, b1, w2, b2 = params
    h1 = jnp.linalg.norm(x)  # Capa invariante a rotaciones
    h2 = jnp.tanh(jnp.dot(w1, h1) + b1)
    return jnp.dot(w2, h2) + b2

# Función de pérdida
def loss(params, x, y):
    """Función de pérdida para la red neuronal."""
    y_pred = invariant_neural_network(x, params)
    return jnp.mean((y_pred - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = jit(grad(loss))

# Optimización de la red neuronal
def optimize_neural_network(x, y, params, num_steps=1000, learning_rate=0.01):
    for i in range(num_steps):
        grads = grad_loss(params, x, y)
        params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)
        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss(params, x, y):.4f}")
    return params

# Datos de ejemplo
np.random.seed(42)
x = np.random.randn(100, 2)
y = np.linalg.norm(x, axis=1)  # Etiquetas son la norma Euclidiana

# Inicialización de parámetros
w1 = np.random.randn(1)
b1 = np.random.randn(1)
w2 = np.random.randn(1)
b2 = np.random.randn(1)
params = (w1, b1, w2, b2)

# Optimización
optimized_params = optimize_neural_network(x, y, params)

# Verificación de la invariancia
theta = np.pi / 4  # Rotación de 45 grados
x_rotated = vmap(so2_action, in_axes=(None, 0))(theta, x)
y_pred = invariant_neural_network(x, optimized_params)
y_pred_rotated = invariant_neural_network(x_rotated, optimized_params)

# Cálculo del error
error = jnp.mean(jnp.abs(y_pred - y_pred_rotated))
print(f"Error de invariancia: {error:.4f}")