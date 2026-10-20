import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import scipy.linalg
import torch

# Función para realizar la descomposición SVD y truncar
def truncated_svd(matrix, rank):
    U, S, Vt = jnp.linalg.svd(matrix, full_matrices=False)
    U_trunc = U[:, :rank]
    S_trunc = S[:rank]
    Vt_trunc = Vt[:rank, :]
    return U_trunc @ (S_trunc[:, None] * Vt_trunc)

# Función para aplicar una transformación isométrica
def isometric_transform(matrix):
    Q, R = jnp.linalg.qr(matrix)
    return Q

# Función de pérdida de ejemplo
def loss_function(params, x, y):
    # Aplicar transformación isométrica
    params_isometric = isometric_transform(params)
    # Calcular la salida de la red neuronal
    output = jnp.dot(params_isometric, x)
    # Calcular la pérdida (por ejemplo, MSE)
    loss = jnp.mean((output - y) ** 2)
    return loss

# Gradiente de la función de pérdida
grad_loss = grad(loss_function)

# Función de entrenamiento
@jit
def train_step(params, x, y, learning_rate, rank):
    # Calcular el gradiente
    grads = grad_loss(params, x, y)
    # Aplicar truncamiento SVD a los gradientes
    grads_trunc = truncated_svd(grads, rank)
    # Actualizar los parámetros
    params = params - learning_rate * grads_trunc
    return params

# Datos de ejemplo
x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = jnp.array([1.0, 2.0, 3.0])

# Inicialización de parámetros
params = jnp.array([[0.1, 0.2], [0.3, 0.4]])

# Hiperparámetros
learning_rate = 0.01
rank = 1
num_steps = 100

# Entrenamiento
for step in range(num_steps):
    params = train_step(params, x, y, learning_rate, rank)
    if step % 10 == 0:
        loss = loss_function(params, x, y)
        print(f"Step {step}: Loss = {loss:.4f}")

# Resultados finales
print("Final Parameters:")
print(params)