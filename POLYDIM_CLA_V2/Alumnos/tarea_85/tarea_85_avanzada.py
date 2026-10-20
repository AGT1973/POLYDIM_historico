import jax
import jax.numpy as jnp
import jax.random as random
import torch
from scipy.linalg import svd

# Función para truncar el rango de un tensor usando SVD
def truncate_svd(A, rank):
    U, S, Vh = svd(A, full_matrices=False)
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]
    return U @ (S[:, None] * Vh)

# Función para aplicar la red neuronal con invariancia isométrica
@jax.jit
def isometric_network(x, W1, W2, b1, b2, rank):
    # Capa 1: Aplicación lineal con truncamiento SVD
    z1 = jnp.dot(x, truncate_svd(W1, rank)) + b1
    a1 = jax.nn.relu(z1)
    
    # Capa 2: Aplicación lineal con truncamiento SVD
    z2 = jnp.dot(a1, truncate_svd(W2, rank)) + b2
    a2 = jax.nn.relu(z2)
    
    return a2

# Función de pérdida
@jax.jit
def loss_fn(params, x, y):
    W1, W2, b1, b2, rank = params
    y_pred = isometric_network(x, W1, W2, b1, b2, rank)
    loss = jnp.mean((y_pred - y) ** 2)
    return loss

# Gradiente de la función de pérdida
grad_loss = jax.grad(loss_fn)

# Datos de ejemplo
key = random.PRNGKey(0)
x = random.normal(key, (100, 10))
y = random.normal(key, (100, 5))

# Inicialización de parámetros
W1 = random.normal(key, (10, 20))
W2 = random.normal(key, (20, 5))
b1 = jnp.zeros(20)
b2 = jnp.zeros(5)
rank = 10

params = (W1, W2, b1, b2, rank)

# Entrenamiento
learning_rate = 0.01
num_steps = 1000

for step in range(num_steps):
    grads = grad_loss(params, x, y)
    W1, W2, b1, b2, rank = params
    W1 -= learning_rate * grads[0]
    W2 -= learning_rate * grads[1]
    b1 -= learning_rate * grads[2]
    b2 -= learning_rate * grads[3]
    params = (W1, W2, b1, b2, rank)
    
    if step % 100 == 0:
        loss = loss_fn(params, x, y)
        print(f"Step {step}, Loss: {loss:.4f}")

# Validación final
final_loss = loss_fn(params, x, y)
print(f"Final Loss: {final_loss:.4f}")