import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.optimize import minimize
from jax import grad, jit, vmap

# Definición de la red neuronal simple
def simple_neural_network(params, x):
    W1, b1, W2, b2 = params
    h = jnp.tanh(jnp.dot(x, W1) + b1)
    return jnp.dot(h, W2) + b2

# Función de pérdida
def loss_fn(params, x, y):
    y_pred = simple_neural_network(params, x)
    return jnp.mean((y_pred - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = grad(loss_fn)

# Inicialización de parámetros
np.random.seed(42)
W1 = np.random.randn(5, 10)
b1 = np.random.randn(10)
W2 = np.random.randn(10, 1)
b2 = np.random.randn(1)
params = (W1, b1, W2, b2)

# Datos de entrenamiento
x_train = np.random.randn(100, 5)
y_train = np.random.randn(100, 1)

# Función de optimización usando JAX
@jit
def optimize(params, x, y, learning_rate=0.01):
    grads = grad_loss(params, x, y)
    new_params = [p - learning_rate * g for p, g in zip(params, grads)]
    return new_params

# Entrenamiento de la red neuronal
for i in range(1000):
    params = optimize(params, x_train, y_train)
    if i % 100 == 0:
        loss = loss_fn(params, x_train, y_train)
        print(f'Iteración {i}, Pérdida: {loss:.4f}')

# Validación empírica de la estabilización de gradientes
def validate_gradient_stability(params, x, y, num_steps=100):
    gradients = []
    for _ in range(num_steps):
        grads = grad_loss(params, x, y)
        gradients.append(grads)
        params = optimize(params, x, y)
    
    gradients = np.array(gradients)
    gradient_norms = np.linalg.norm(gradients, axis=1)
    return gradient_norms

# Validación empírica
gradient_norms = validate_gradient_stability(params, x_train, y_train)
print(f'Normas de gradientes a lo largo del entrenamiento: {gradient_norms}')