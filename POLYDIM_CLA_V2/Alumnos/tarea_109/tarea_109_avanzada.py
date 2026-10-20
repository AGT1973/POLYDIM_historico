import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import scipy.spatial.distance as dist

# Definir la red neuronal simple usando JAX
def mlp(params, x):
    w1, b1, w2, b2 = params
    hidden = jnp.dot(x, w1) + b1
    hidden = jax.nn.relu(hidden)
    output = jnp.dot(hidden, w2) + b2
    return output

# Función de pérdida
def loss_fn(params, x, y):
    predictions = mlp(params, x)
    return jnp.mean((predictions - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = jit(grad(loss_fn))

# Generar datos de ejemplo
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (100, 2))
y = jax.random.normal(key, (100, 1))

# Inicializar parámetros de la red neuronal
w1 = jax.random.normal(key, (2, 10))
b1 = jax.random.normal(key, (10,))
w2 = jax.random.normal(key, (10, 1))
b2 = jax.random.normal(key, (1,))
params = (w1, b1, w2, b2)

# Entrenar la red neuronal
learning_rate = 0.01
num_epochs = 1000
for epoch in range(num_epochs):
    gradients = grad_loss(params, x, y)
    params = jax.tree_map(lambda p, g: p - learning_rate * g, params, gradients)

# Función para calcular la distancia isométrica
def isometric_distance(x, y, params):
    x_mapped = mlp(params, x)
    y_mapped = mlp(params, y)
    return jnp.linalg.norm(x_mapped - y_mapped), jnp.linalg.norm(x - y)

# Validar la invariancia isométrica
distances_input = vmap(lambda x, y: jnp.linalg.norm(x - y))(x, x)
distances_output = vmap(lambda x, y: jnp.linalg.norm(mlp(params, x) - mlp(params, y)))(x, x)

# Calcular el error de invariancia isométrica
isometric_error = jnp.mean(jnp.abs(distances_output - distances_input))

print(f"Error de invariancia isométrica: {isometric_error}")