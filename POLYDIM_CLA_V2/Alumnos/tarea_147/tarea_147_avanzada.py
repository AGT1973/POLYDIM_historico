import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.linalg import svd
import torch
from scipy.linalg import expm

# Definición de la red neuronal
def neural_net(params, x):
    w1, b1, w2, b2 = params
    h1 = jnp.relu(jnp.dot(x, w1) + b1)
    return jnp.dot(h1, w2) + b2

# Función de costo
def loss(params, x, y):
    y_pred = neural_net(params, x)
    return jnp.mean((y_pred - y) ** 2)

# Gradiente del costo
grad_loss = grad(loss)

# Truncamiento SVD
def truncate_svd(matrix, k):
    u, s, vh = svd(matrix)
    s_truncated = jnp.concatenate([s[:k], jnp.zeros(s.shape[0] - k)])
    return u @ jnp.diag(s_truncated) @ vh

# Optimización
def optimize(params, x, y, lr=0.01, num_steps=1000, k=10):
    for _ in range(num_steps):
        grads = grad_loss(params, x, y)
        params = jax.tree_map(lambda p, g: p - lr * g, params, grads)
        
        # Truncamiento SVD periódico
        w1, b1, w2, b2 = params
        w1 = truncate_svd(w1, k)
        w2 = truncate_svd(w2, k)
        params = (w1, b1, w2, b2)
    
    return params

# Datos de ejemplo
x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = jnp.array([[1.0], [1.0], [2.0]])

# Inicialización de parámetros
w1 = jnp.array([[0.1, 0.2], [0.3, 0.4]])
b1 = jnp.array([0.1, 0.2])
w2 = jnp.array([[0.1, 0.2]])
b2 = jnp.array([0.1])
params = (w1, b1, w2, b2)

# Optimización
params = optimize(params, x, y)

# Evaluación de la red neuronal
y_pred = neural_net(params, x)
print("Predicciones:", y_pred)
print("Pérdida final:", loss(params, x, y))