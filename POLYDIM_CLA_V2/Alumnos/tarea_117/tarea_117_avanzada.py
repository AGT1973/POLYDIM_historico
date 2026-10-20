import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from scipy.linalg import svd

# Definición de la capa de transformación de Lie
def lie_transformation(x, theta):
    # Generamos una matriz de rotación 2D para simplicidad
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                   [jnp.sin(theta), jnp.cos(theta)]])
    return jnp.dot(R, x)

# Definición de la capa de invariancia
def invariant_layer(x, theta):
    transformed_x = lie_transformation(x, theta)
    # Promediamos los resultados de las transformaciones
    return jnp.mean(transformed_x, axis=0)

# Función de pérdida
def loss_function(params, x, y):
    theta = params['theta']
    predictions = invariant_layer(x, theta)
    return jnp.mean((predictions - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = grad(loss_function)

# Optimización usando JAX
@jit
def update(params, x, y, learning_rate):
    gradients = grad_loss(params, x, y)
    return {k: params[k] - learning_rate * gradients[k] for k in params}

# Datos de ejemplo
x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
y = jnp.array([1.0, 1.0])

# Inicialización de parámetros
params = {'theta': jnp.array(0.0)}

# Entrenamiento
learning_rate = 0.01
num_iterations = 1000

for i in range(num_iterations):
    params = update(params, x, y, learning_rate)
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss_function(params, x, y)}")

# Verificación de invariancia
theta_test = jnp.array(jnp.pi / 4)
transformed_x = lie_transformation(x, theta_test)
predictions = invariant_layer(transformed_x, params['theta'])
print(f"Predictions after transformation: {predictions}")