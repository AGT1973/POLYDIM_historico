import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
import scipy.linalg

# Definición de la función de pérdida
def loss(params, x, y):
    predictions = jnp.dot(params, x)
    return jnp.mean((predictions - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = grad(loss)

# Truncamiento SVD periódico
def truncate_svd(matrix, rank):
    U, s, Vh = jnp.linalg.svd(matrix, full_matrices=False)
    s_truncated = jnp.concatenate([s[:rank], jnp.zeros(s.shape[0] - rank)])
    return jnp.dot(U, jnp.dot(jnp.diag(s_truncated), Vh))

# Regularización de gradientes
def regularize_gradients(params, x, y, rank):
    gradients = grad_loss(params, x, y)
    truncated_gradients = truncate_svd(gradients, rank)
    return truncated_gradients

# Inicialización de parámetros y datos
params = jnp.array(np.random.randn(10, 1))
x = jnp.array(np.random.randn(100, 10))
y = jnp.array(np.random.randn(100, 1))

# Optimización con regularización de gradientes
@jit
def optimize(params, x, y, rank, learning_rate, num_steps):
    for i in range(num_steps):
        gradients = regularize_gradients(params, x, y, rank)
        params -= learning_rate * gradients
    return params

# Parámetros de optimización
rank = 5
learning_rate = 0.01
num_steps = 1000

# Ejecución de la optimización
optimized_params = optimize(params, x, y, rank, learning_rate, num_steps)
print("Optimized Parameters:", optimized_params)