import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
from scipy.linalg import svd
import numpy as np

# Función para aplicar SVD y truncar
def truncate_svd(matrix, rank):
    U, S, Vt = svd(matrix)
    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]
    return U, S, Vt

# Función de forward pass con estabilización SVD
@jit
def forward_pass(params, x):
    W1, b1, W2, b2 = params
    h1 = jnp.tanh(jnp.dot(x, W1) + b1)
    h1_truncated = truncate_svd(h1, rank=10)  # Truncar a rango 10
    h1 = jnp.dot(h1_truncated[0], jnp.diag(h1_truncated[1])) @ h1_truncated[2]
    y = jnp.dot(h1, W2) + b2
    return y

# Función de pérdida
def loss(params, x, y_true):
    y_pred = forward_pass(params, x)
    return jnp.mean((y_pred - y_true) ** 2)

# Gradiente de la función de pérdida
grad_loss = jit(grad(loss))

# Inicialización de parámetros
input_dim = 10
hidden_dim = 50
output_dim = 1

W1 = jnp.array(np.random.randn(input_dim, hidden_dim))
b1 = jnp.array(np.random.randn(hidden_dim))
W2 = jnp.array(np.random.randn(hidden_dim, output_dim))
b2 = jnp.array(np.random.randn(output_dim))

params = (W1, b1, W2, b2)

# Datos de ejemplo
x = jnp.array(np.random.randn(100, input_dim))
y_true = jnp.array(np.random.randn(100, output_dim))

# Entrenamiento con JAX
learning_rate = 0.01
num_steps = 1000

for step in range(num_steps):
    grads = grad_loss(params, x, y_true)
    params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss(params, x, y_true):.4f}")

# Validación final
print(f"Final Loss: {loss(params, x, y_true):.4f}")