import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.linalg import svd

# Definir la función de pérdida (ejemplo: MSE)
def loss_fn(params, x, y):
    preds = jnp.dot(x, params)
    return jnp.mean((preds - y) ** 2)

# Gradiente de la función de pérdida
grad_loss_fn = jax.grad(loss_fn)

# Función de truncamiento SVD
def truncated_svd(W, k):
    U, S, Vt = svd(W)
    S_truncated = S[:k]
    U_truncated = U[:, :k]
    Vt_truncated = Vt[:k, :]
    return U_truncated @ np.diag(S_truncated) @ Vt_truncated

# Descenso del gradiente en el fibrado tangente
def tangent_bundle_descent(params, x, y, alpha=0.1, k=10, num_steps=100):
    for t in range(num_steps):
        grad = grad_loss_fn(params, x, y)
        # Aplicar la métrica riemanniana (ejemplo: identidad)
        grad_metric = grad
        # Actualizar los parámetros
        params -= alpha * grad_metric
        # Truncar el tensor de parámetros periódicamente
        if t % 10 == 0:
            params = truncated_svd(params, k)
    return params

# Datos de ejemplo
x = jnp.array([[1, 2], [2, 3], [3, 4]])
y = jnp.array([3, 5, 7])

# Inicializar parámetros
params = jnp.array([[0.5, 0.5], [0.5, 0.5]])

# Realizar el descenso del gradiente
params_optimized = tangent_bundle_descent(params, x, y)

print("Parámetros optimizados:", params_optimized)