import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import scipy.linalg

# Definición de la función de pérdida
def loss_fn(params, x, y):
    y_pred = model(params, x)
    return jnp.mean((y_pred - y) ** 2)

# Definición del modelo simple
def model(params, x):
    return jnp.dot(params, x)

# Función de truncamiento SVD
def truncate_svd(W, k):
    U, S, V = jnp.linalg.svd(W, full_matrices=False)
    S_k = jnp.diag(S[:k])
    U_k = U[:, :k]
    V_k = V[:k, :]
    return jnp.dot(U_k, jnp.dot(S_k, V_k))

# Función de optimización con truncamiento SVD
@jit
def optimize_step(params, x, y, k):
    grads = grad(loss_fn)(params, x, y)
    params -= 0.01 * grads  # Paso de gradiente descendente
    params = truncate_svd(params, k)  # Truncamiento SVD
    return params

# Datos de ejemplo
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

# Inicialización de parámetros
params = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Número de iteraciones y rango truncado
num_iterations = 100
k = 2

# Bucle de optimización
for i in range(num_iterations):
    params = optimize_step(params, x, y, k)
    if i % 10 == 0:
        print(f"Iteración {i}, Pérdida: {loss_fn(params, x, y)}")

# Resultados finales
print("Parámetros finales:")
print(params)
print("Pérdida final:")
print(loss_fn(params, x, y))