import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
from scipy.linalg import svd

# Definición de la red neuronal
def neural_network(x, params):
    for W, b in params:
        x = jnp.dot(x, W) + b
        x = jax.nn.relu(x)
    return x

# Función de pérdida
def loss(params, x, y):
    predictions = neural_network(x, params)
    return jnp.mean((predictions - y) ** 2)

# Truncamiento SVD
def truncate_svd(W, k=10):
    U, S, Vh = svd(W, full_matrices=False)
    S_truncated = S[:k]
    U_truncated = U[:, :k]
    Vh_truncated = Vh[:k, :]
    return U_truncated @ jnp.diag(S_truncated) @ Vh_truncated

# Gradiente de la función de pérdida
@jit
def compute_gradients(params, x, y):
    return grad(loss)(params, x, y)

# Actualización de los parámetros con truncamiento SVD
def update_params(params, gradients, learning_rate=0.01, k=10):
    new_params = []
    for (W, b), (dW, db) in zip(params, gradients):
        W_new = W - learning_rate * dW
        W_new = truncate_svd(W_new, k)
        b_new = b - learning_rate * db
        new_params.append((W_new, b_new))
    return new_params

# Datos de ejemplo
x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
y = jnp.array([1.0, 0.0])

# Inicialización de parámetros
params = [
    (jnp.array([[0.1, 0.2], [0.3, 0.4]]), jnp.array([0.5, 0.6])),
    (jnp.array([[0.7, 0.8], [0.9, 1.0]]), jnp.array([1.1, 1.2])),
]

# Entrenamiento
for i in range(1000):
    gradients = compute_gradients(params, x, y)
    params = update_params(params, gradients)

# Evaluación final
final_loss = loss(params, x, y)
print(f"Final Loss: {final_loss}")