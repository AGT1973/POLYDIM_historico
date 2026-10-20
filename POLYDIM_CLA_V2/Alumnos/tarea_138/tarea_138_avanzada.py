import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.lax import scan
import numpy as np
from scipy.linalg import svd

# Definición de la función de pérdida
def loss_fn(params, x, y):
    predictions = model(params, x)
    return jnp.mean((predictions - y) ** 2)

# Modelo simple de red neuronal
def model(params, x):
    return jnp.dot(params, x)

# Proyección en el espacio tangente
def project_tangent_space(grad, params):
    return grad - jnp.dot(jnp.dot(grad, params.T), params)

# Truncación SVD
def truncate_svd(matrix, rank):
    U, S, Vh = svd(matrix)
    return jnp.dot(U[:, :rank], jnp.dot(jnp.diag(S[:rank]), Vh[:rank, :]))

# Optimización con fibrado tangente y truncación SVD
@jit
def optimize(params, x, y, learning_rate, rank):
    grad = grad(loss_fn)(params, x, y)
    tangent_grad = project_tangent_space(grad, params)
    updated_params = params - learning_rate * tangent_grad
    truncated_params = truncate_svd(updated_params, rank)
    return truncated_params

# Datos de ejemplo
x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = jnp.array([1.0, 2.0, 3.0])

# Inicialización de pesos
params = jnp.array([[0.1, 0.2], [0.3, 0.4]])

# Parámetros de optimización
learning_rate = 0.01
rank = 1
num_steps = 100

# Optimización
for i in range(num_steps):
    params = optimize(params, x, y, learning_rate, rank)
    if i % 10 == 0:
        print(f"Step {i}, Loss: {loss_fn(params, x, y)}")

print("Final params:", params)