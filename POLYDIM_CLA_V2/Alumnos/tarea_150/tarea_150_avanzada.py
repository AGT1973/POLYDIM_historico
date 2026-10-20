import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from scipy.linalg import svd
import torch

# Definir la función de pérdida
def loss_fn(params, inputs, targets):
    predictions = forward_pass(params, inputs)
    return jnp.mean((predictions - targets) ** 2)

# Definir la función de paso forward
def forward_pass(params, inputs):
    for W, b in params:
        inputs = jnp.dot(inputs, W) + b
        inputs = jnp.tanh(inputs)
    return inputs

# Definir la función de truncamiento SVD
def truncate_svd(matrix, rank):
    U, S, V = svd(matrix, full_matrices=False)
    S_truncated = S[:rank]
    U_truncated = U[:, :rank]
    V_truncated = V[:rank, :]
    return jnp.dot(U_truncated, jnp.dot(jnp.diag(S_truncated), V_truncated))

# Definir la función de optimización
@jit
def optimize(params, inputs, targets, learning_rate, rank):
    grads = grad(loss_fn)(params, inputs, targets)
    new_params = []
    for W, b, grad_W, grad_b in zip(params, params, grads, grads):
        W_new = W - learning_rate * grad_W
        b_new = b - learning_rate * grad_b
        W_new = truncate_svd(W_new, rank)
        new_params.append((W_new, b_new))
    return new_params

# Generar datos de ejemplo
inputs = jnp.array([[1.0, 2.0], [3.0, 4.0]])
targets = jnp.array([[0.5, 1.0], [1.0, 0.5]])

# Inicializar parámetros
params = [
    (jnp.array([[0.1, 0.2], [0.3, 0.4]]), jnp.array([0.1, 0.2])),
    (jnp.array([[0.5, 0.6], [0.7, 0.8]]), jnp.array([0.3, 0.4]))
]

# Parámetros de optimización
learning_rate = 0.01
rank = 2
num_iterations = 100

# Optimización
for i in range(num_iterations):
    params = optimize(params, inputs, targets, learning_rate, rank)
    if i % 10 == 0:
        loss = loss_fn(params, inputs, targets)
        print(f"Iteration {i}, Loss: {loss:.4f}")

# Verificar la estabilidad del rango
for W, b in params:
    print(f"Rank of W: {jnp.linalg.matrix_rank(W)}")