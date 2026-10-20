import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
from scipy.linalg import expm

# Definición de la capa neuronal en SO(3) (grupo de rotaciones en R^3)
def so3_layer(x, W, b):
    """
    x: Input tensor of shape (batch_size, 3)
    W: Weight matrix of shape (3, 3)
    b: Bias vector of shape (3,)
    """
    # Ensure W is a rotation matrix (orthogonal with det=1)
    U, _, V = jnp.linalg.svd(W)
    W = U @ V.T
    if jnp.linalg.det(W) < 0:
        W = -W
    return jnp.dot(x, W) + b

# Función de pérdida con regularización de gradientes
def loss_fn(params, x, y_true):
    W, b = params
    y_pred = so3_layer(x, W, b)
    mse_loss = jnp.mean((y_pred - y_true) ** 2)
    grad_loss = jnp.mean(jnp.linalg.norm(grad(so3_layer)(x, W, b), axis=1))
    return mse_loss + 0.1 * grad_loss

# Generación de datos de ejemplo
x = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
y_true = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

# Inicialización de parámetros
W = jnp.eye(3)
b = jnp.zeros(3)
params = (W, b)

# Optimización usando JAX
@jit
def update(params, x, y_true, lr=0.01):
    grads = grad(loss_fn)(params, x, y_true)
    return [(p - lr * g) for p, g in zip(params, grads)]

# Entrenamiento
for i in range(1000):
    params = update(params, x, y_true)
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss_fn(params, x, y_true):.4f}")

# Validación
W_opt, b_opt = params
y_pred = so3_layer(x, W_opt, b_opt)
print("Optimized W:\n", W_opt)
print("Optimized b:\n", b_opt)
print("Predicted y:\n", y_pred)
print("True y:\n", y_true)