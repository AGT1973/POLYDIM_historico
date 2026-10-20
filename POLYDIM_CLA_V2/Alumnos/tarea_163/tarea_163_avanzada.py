import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
from scipy.linalg import svd

# Definir la función de homotopía
def homotopy(f0, f1, t):
    return (1 - t) * f0 + t * f1

# Definir la función de red neuronal
def neural_network(x, params):
    W1, b1, W2, b2 = params
    h1 = jax.nn.relu(jnp.dot(x, W1) + b1)
    return jnp.dot(h1, W2) + b2

# Definir la función de pérdida
def loss_fn(params, x, y):
    y_pred = neural_network(x, params)
    return jnp.mean((y_pred - y) ** 2)

# Definir la función de truncación SVD
def truncate_svd(W, k):
    U, S, V = svd(W)
    S_truncated = S[:k]
    U_truncated = U[:, :k]
    V_truncated = V[:k, :]
    return U_truncated @ jnp.diag(S_truncated) @ V_truncated

# Definir la función de entrenamiento
@jit
def train_step(params, x, y, k):
    # Calcular el gradiente de la función de pérdida
    grads = grad(loss_fn)(params, x, y)
    
    # Aplicar la truncación SVD a los gradientes
    W1_grad, b1_grad, W2_grad, b2_grad = grads
    W1_grad_truncated = truncate_svd(W1_grad, k)
    W2_grad_truncated = truncate_svd(W2_grad, k)
    
    # Actualizar los parámetros
    eta = 0.01  # Tasa de aprendizaje
    W1 = params[0] - eta * W1_grad_truncated
    b1 = params[1] - eta * b1_grad
    W2 = params[2] - eta * W2_grad_truncated
    b2 = params[3] - eta * b2_grad
    
    return W1, b1, W2, b2

# Generar datos de ejemplo
x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = jnp.array([1.0, 0.0, 1.0])

# Inicializar parámetros de la red neuronal
W1 = jnp.array([[0.1, 0.2], [0.3, 0.4]])
b1 = jnp.array([0.1, 0.2])
W2 = jnp.array([[0.5, 0.6], [0.7, 0.8]])
b2 = jnp.array([0.3, 0.4])
params = (W1, b1, W2, b2)

# Entrenar la red neuronal
k = 1  # Rango de truncación SVD
for i in range(100):
    params = train_step(params, x, y, k)
    if i % 10 == 0:
        loss = loss_fn(params, x, y)
        print(f"Step {i}, Loss: {loss:.4f}")

# Imprimir los parámetros finales
print("Final parameters:")
print("W1:", params[0])
print("b1:", params[1])
print("W2:", params[2])
print("b2:", params[3])