import jax
import jax.numpy as jnp
import numpy as np
import scipy
import torch
from jax import vmap, jit, grad

# Definición del rotor de Clifford
def rotor(theta, bivector):
    return jnp.exp(0.5 * theta * bivector)

# Transformación de datos por el rotor
def transform(x, rotor):
    return rotor @ x @ rotor.conj()

# Capa de red neuronal con rotor de Clifford
def cliffor_layer(x, theta, bivector, bias):
    rotor_ = rotor(theta, bivector)
    transformed_x = transform(x, rotor_)
    return jnp.tanh(transformed_x + bias)

# Función de pérdida
def loss_fn(params, x, y_true):
    theta, bivector, bias = params
    y_pred = cliffor_layer(x, theta, bivector, bias)
    return jnp.mean((y_pred - y_true) ** 2)

# Gradiente de la función de pérdida
grad_loss_fn = jit(grad(loss_fn))

# Generación de datos de ejemplo
np.random.seed(42)
x_data = np.random.randn(100, 3)
y_true = np.sin(x_data.sum(axis=1))

# Inicialización de parámetros
theta = jnp.array(0.1)
bivector = jnp.array([0.5, -0.3, 0.2])
bias = jnp.array(0.0)

params = (theta, bivector, bias)

# Optimización usando gradiente descendente
learning_rate = 0.01
num_iterations = 1000

for i in range(num_iterations):
    grads = grad_loss_fn(params, x_data, y_true)
    params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss_fn(params, x_data, y_true)}")

# Validación final
theta_opt, bivector_opt, bias_opt = params
y_pred = cliffor_layer(x_data, theta_opt, bivector_opt, bias_opt)
print("Final Loss:", loss_fn(params, x_data, y_true))