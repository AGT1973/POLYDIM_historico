import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
from torch import einsum
from scipy.linalg import svd

# Función para aplicar la invariancia isométrica
def isometric_mapping(x):
    # Ejemplo de una transformación isométrica: rotación en 2D
    theta = jnp.pi / 4
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                   [jnp.sin(theta), jnp.cos(theta)]])
    return R @ x

# Función para truncar el rango de un tensor usando SVD
def truncate_svd(tensor, rank):
    U, S, Vh = svd(tensor)
    S_truncated = jnp.concatenate([S[:rank], jnp.zeros(len(S) - rank)])
    return U @ jnp.diag(S_truncated) @ Vh

# Función de pérdida (ejemplo: MSE)
def loss_fn(params, inputs, targets):
    predictions = isometric_mapping(inputs @ params)
    return jnp.mean((predictions - targets) ** 2)

# Función de entrenamiento con truncamiento SVD periódico
@jit
def train_step(params, inputs, targets, rank):
    # Calcular el gradiente de la función de pérdida
    grads = grad(loss_fn)(params, inputs, targets)
    
    # Aplicar truncamiento SVD al gradiente
    truncated_grads = truncate_svd(grads, rank)
    
    # Actualizar los parámetros
    learning_rate = 0.01
    params = params - learning_rate * truncated_grads
    
    return params

# Datos de ejemplo
inputs = jnp.array([[1.0, 0.5], [0.5, 1.0], [0.0, 1.0]])
targets = jnp.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])

# Inicialización de los parámetros
params = jnp.array([[0.5, 0.5], [0.5, 0.5]])

# Número de iteraciones de entrenamiento
num_iterations = 100
rank = 1  # Rango truncado

# Entrenamiento
for i in range(num_iterations):
    params = train_step(params, inputs, targets, rank)
    if i % 10 == 0:
        loss = loss_fn(params, inputs, targets)
        print(f"Iteration {i}, Loss: {loss:.4f}")

# Resultados finales
print("Final Parameters:")
print(params)