import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
from jax.lax import svd
from scipy.linalg import qr
import numpy as np

# Función para aplicar la descomposición SVD y truncar los valores singulares
def truncated_svd(A, k):
    U, S, V = svd(A, full_matrices=False)
    S = S[:k]
    U = U[:, :k]
    V = V[:k, :]
    return U, S, V

# Función para aplicar la descomposición QR seguida de SVD
def qr_svd_truncate(A, k):
    Q, R = qr(A)
    U, S, V = truncated_svd(R, k)
    return Q @ U, S, V

# Definición de una capa de red neuronal simple
def neural_layer(x, W, b):
    return jnp.dot(W, x) + b

# Función de pérdida
def loss_fn(params, x, y):
    W, b = params
    y_pred = neural_layer(x, W, b)
    return jnp.mean((y_pred - y) ** 2)

# Gradiente de la función de pérdida
grad_loss_fn = grad(loss_fn)

# Función de actualización de parámetros con regularización SVD
@jit
def update_params(params, x, y, learning_rate, k):
    W, b = params
    grad_W, grad_b = grad_loss_fn(params, x, y)
    
    # Aplicar descomposición SVD y truncar
    U, S, V = qr_svd_truncate(grad_W, k)
    W = W - learning_rate * (U @ np.diag(S) @ V)
    b = b - learning_rate * grad_b
    
    return (W, b)

# Datos de entrada y salida de ejemplo
x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
y = jnp.array([[1.0], [2.0]])

# Inicialización de parámetros
W = jnp.array([[0.1, 0.2], [0.3, 0.4]])
b = jnp.array([0.1])

# Hiperparámetros
learning_rate = 0.01
k = 1
num_epochs = 100

# Entrenamiento
for epoch in range(num_epochs):
    params = (W, b)
    W, b = update_params(params, x, y, learning_rate, k)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss_fn(params, x, y)}")

print("Final Weights:", W)
print("Final Bias:", b)