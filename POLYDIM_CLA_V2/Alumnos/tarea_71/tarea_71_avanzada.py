import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
from scipy.linalg import expm

# Definir el rotor de Clifford
def rotor(theta, p, q):
    # Generadores del álgebra de Clifford
    generators = [jnp.eye(p + q)]
    for i in range(p + q):
        for j in range(i + 1, p + q):
            generators.append(jnp.kron(jnp.eye(2**(p + q - 2)), jnp.array([[0, -1], [1, 0]])))
    
    # Suma ponderada de generadores
    rotor_matrix = jnp.zeros((2**(p + q), 2**(p + q)))
    for i in range(len(generators)):
        rotor_matrix += theta[i] * generators[i]
    
    # Exponencial de la matriz
    return expm(rotor_matrix)

# Función de transformación lineal con rotor
def linear_transform_with_rotor(W, b, x, theta, p, q):
    R = rotor(theta, p, q)
    return R @ (W @ x + b) @ R.T

# Función de pérdida
def loss(params, x, y_true):
    W, b, theta = params
    y_pred = linear_transform_with_rotor(W, b, x, theta, p, q)
    return jnp.mean((y_pred - y_true) ** 2)

# Gradiente de la función de pérdida
grad_loss = jit(grad(loss))

# Ejemplo de datos
p, q = 2, 0  # Dimensiones del espacio
W = jnp.array([[0.5, 0.1], [0.1, 0.5]])
b = jnp.array([0.1, 0.1])
theta = jnp.array([0.1, 0.1])
x = jnp.array([[1.0, 2.0], [2.0, 1.0]])
y_true = jnp.array([[1.5, 1.5], [1.5, 1.5]])

# Parámetros iniciales
params = (W, b, theta)

# Actualización de parámetros usando gradiente descendente
learning_rate = 0.01
num_steps = 100

for i in range(num_steps):
    grads = grad_loss(params, x, y_true)
    W, b, theta = params
    W -= learning_rate * grads[0]
    b -= learning_rate * grads[1]
    theta -= learning_rate * grads[2]
    params = (W, b, theta)
    if i % 10 == 0:
        print(f"Step {i}, Loss: {loss(params, x, y_true)}")

# Resultados finales
print("Final Parameters:")
print(f"W: {W}")
print(f"b: {b}")
print(f"theta: {theta}")
print(f"Final Loss: {loss(params, x, y_true)}")