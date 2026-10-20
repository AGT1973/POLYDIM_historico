import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import scipy.linalg
import torch

# Función para realizar la descomposición SVD y truncar
def truncated_svd(A, k):
    U, S, V = jnp.linalg.svd(A, full_matrices=False)
    S_k = S[:k]
    U_k = U[:, :k]
    V_k = V[:k, :]
    return U_k, S_k, V_k

# Función para reconstruir el tensor truncado
def reconstruct_tensor(U_k, S_k, V_k):
    return jnp.dot(U_k, jnp.dot(jnp.diag(S_k), V_k))

# Función de pérdida de ejemplo
def loss_function(W, X, y):
    y_pred = jnp.dot(X, W)
    return jnp.mean((y - y_pred) ** 2)

# Función de entrenamiento con truncamiento SVD periódico
@jit
def train_step(W, X, y, k, step_size):
    # Calcular el gradiente
    grad_loss = grad(loss_function)(W, X, y)
    
    # Aplicar la descomposición SVD y truncar
    U_k, S_k, V_k = truncated_svd(grad_loss, k)
    grad_loss_truncated = reconstruct_tensor(U_k, S_k, V_k)
    
    # Actualizar los pesos
    W = W - step_size * grad_loss_truncated
    return W

# Datos de ejemplo
X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = jnp.array([7.0, 8.0, 9.0])

# Inicializar los pesos
W = jnp.array([[0.1], [0.2]])

# Hiperparámetros
k = 1  # Rango de truncamiento
step_size = 0.01
num_steps = 100

# Entrenamiento
for i in range(num_steps):
    W = train_step(W, X, y, k, step_size)
    if i % 10 == 0:
        print(f"Step {i}, Loss: {loss_function(W, X, y)}")

print("Final Weights:", W)