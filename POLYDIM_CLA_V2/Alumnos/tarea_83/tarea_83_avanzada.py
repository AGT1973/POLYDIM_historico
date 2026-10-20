import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import scipy.optimize as opt
import torch

# Función para crear un rotor de Clifford
def create_rotor(theta):
    return jnp.array([jnp.cos(theta / 2), jnp.sin(theta / 2)])

# Función para aplicar un rotor a un vector
def apply_rotor(rotor, v):
    return rotor * v * rotor.conj()

# Función de pérdida
def loss(params, x, y):
    rotor = create_rotor(params[0])
    x_transformed = vmap(apply_rotor, (None, 0))(rotor, x)
    y_pred = jnp.sum(x_transformed, axis=1)  # Ejemplo simple de sumatoria
    return jnp.mean((y_pred - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = grad(loss)

# Datos de ejemplo
x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = jnp.array([1.0, 1.0, 2.0])

# Optimización de los parámetros
def optimize(params):
    return loss(params, x, y)

initial_params = jnp.array([0.1])
result = opt.minimize(optimize, initial_params, method='BFGS', jac=grad_loss)

# Resultados
print("Optimized parameters:", result.x)
print("Loss:", result.fun)