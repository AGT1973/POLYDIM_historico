import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import torch
from scipy.linalg import svd

# Función para generar una matriz de rotación aleatoria en SO(n)
def random_rotation(n):
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(0), (n, n)))
    return Q

# Función para aplicar una transformación lineal isométrica
def isometric_transform(x, W):
    return jnp.dot(W, x)

# Función para truncar una matriz usando SVD
def truncated_svd(A, k):
    U, S, Vh = svd(A)
    return U[:, :k] @ jnp.diag(S[:k]) @ Vh[:k, :]

# Función de pérdida simple (MSE)
def loss_fn(y_true, y_pred):
    return jnp.mean((y_true - y_pred) ** 2)

# Generar datos de ejemplo
n = 10
k = 5
X = jnp.random.randn(n, 100)
W = random_rotation(n)
y_true = jnp.random.randn(n, 100)

# Aplicar transformación isométrica
y_pred = vmap(isometric_transform, (0, None))(X, W)

# Truncar la matriz de transformación periódicamente
W_truncated = truncated_svd(W, k)

# Aplicar transformación isométrica con matriz truncada
y_pred_truncated = vmap(isometric_transform, (0, None))(X, W_truncated)

# Calcular la pérdida antes y después del truncamiento
loss_before = loss_fn(y_true, y_pred)
loss_after = loss_fn(y_true, y_pred_truncated)

print(f"Pérdida antes del truncamiento SVD: {loss_before:.4f}")
print(f"Pérdida después del truncamiento SVD: {loss_after:.4f}")