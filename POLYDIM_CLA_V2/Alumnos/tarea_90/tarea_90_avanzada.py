import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import torch
import scipy.optimize

# Definición de la función de pérdida
def loss_fn(params, x, y):
    predictions = model(params, x)
    return jnp.mean((predictions - y) ** 2)

# Definición del modelo (ejemplo: una red neuronal simple)
def model(params, x):
    return jnp.dot(x, params)

# Generación de datos de ejemplo
x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = jnp.array([1.0, 2.0, 3.0])

# Inicialización de los parámetros
params = jnp.array([0.1, 0.2])

# Cálculo del gradiente
grad_loss = grad(loss_fn)(params, x, y)

# Normalización del gradiente
grad_loss_norm = grad_loss / jnp.linalg.norm(grad_loss)

# Función de optimización usando JAX
@jit
def update_params(params, grad_loss_norm, learning_rate):
    return params - learning_rate * grad_loss_norm

# Parámetros de optimización
learning_rate = 0.01
num_iterations = 100

# Optimización
for i in range(num_iterations):
    grad_loss = grad(loss_fn)(params, x, y)
    grad_loss_norm = grad_loss / jnp.linalg.norm(grad_loss)
    params = update_params(params, grad_loss_norm, learning_rate)
    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss_fn(params, x, y)}")

# Validación final
final_loss = loss_fn(params, x, y)
print(f"Final Loss: {final_loss}")