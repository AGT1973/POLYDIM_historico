import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import numpy as np
import scipy.linalg as spla
import torch

# Función para crear un rotor en el álgebra de Clifford
def create_rotor(theta, bivector):
    return jnp.exp(0.5 * theta * bivector)

# Función para aplicar la rotación a un vector
def apply_rotor(rotor, vector):
    return rotor * vector * jnp.conj(rotor)

# Función para truncar la matriz W usando SVD
def truncate_svd(W, k):
    U, S, Vh = spla.svd(W)
    S_truncated = jnp.zeros_like(S)
    S_truncated[:k] = S[:k]
    return U @ jnp.diag(S_truncated) @ Vh

# Capa de la red neuronal con rotación y truncamiento SVD
@jit
def neural_layer(W, rotor, x, k):
    x_transformed = W @ x
    x_rotated = apply_rotor(rotor, x_transformed)
    W_truncated = truncate_svd(W, k)
    return W_truncated @ x_rotated

# Función de pérdida (ejemplo: MSE)
def loss_fn(params, x, y):
    W, rotor, k = params
    y_pred = neural_layer(W, rotor, x, k)
    return jnp.mean((y_pred - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = grad(loss_fn)

# Ejemplo de entrenamiento
def train(params, x, y, learning_rate=0.01, num_steps=1000):
    for i in range(num_steps):
        grads = grad_loss(params, x, y)
        W, rotor, k = params
        W -= learning_rate * grads[0]
        params = (W, rotor, k)
        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss_fn(params, x, y)}")
    return params

# Datos de ejemplo
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([0.5, 1.0, 1.5])

# Inicialización de parámetros
W = jnp.array(np.random.randn(3, 3))
theta = jnp.array(0.5)
bivector = jnp.array([0, 1, 0])
rotor = create_rotor(theta, bivector)
k = 2
params = (W, rotor, k)

# Entrenamiento
trained_params = train(params, x, y)

# Validación
x_test = jnp.array([4.0, 5.0, 6.0])
y_test = jnp.array([2.0, 2.5, 3.0])
y_pred = neural_layer(*trained_params, x_test, k)
print(f"Predicted: {y_pred}, Ground Truth: {y_test}")