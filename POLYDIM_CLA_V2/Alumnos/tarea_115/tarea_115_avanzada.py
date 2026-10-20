import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import scipy.spatial

# Definir una red neuronal simple usando JAX
def neural_network(params, x):
    for W, b in params:
        x = jnp.dot(x, W) + b
        x = jax.nn.relu(x)
    return x

# Generar datos de entrada
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (100, 10))

# Definir una isometría simple (rotación)
def isometry(x):
    # Generar una matriz de rotación aleatoria
    rotation_matrix = scipy.stats.ortho_group.rvs(10)
    return jnp.dot(x, rotation_matrix)

# Inicializar parámetros de la red neuronal
params = [(jax.random.normal(key, (10, 10)), jax.random.normal(key, (10,))),
          (jax.random.normal(key, (10, 5)), jax.random.normal(key, (5,)))]

# Calcular las activaciones para los datos originales y transformados
activations_original = vmap(neural_network, in_axes=(None, 0))(params, x)
activations_transformed = vmap(neural_network, in_axes=(None, 0))(params, isometry(x))

# Calcular la distancia euclidiana entre las activaciones
distances_original = scipy.spatial.distance.pdist(activations_original, 'euclidean')
distances_transformed = scipy.spatial.distance.pdist(activations_transformed, 'euclidean')

# Comparar las distancias
distance_difference = jnp.linalg.norm(distances_original - distances_transformed)

print(f"Distancia entre las distancias de activaciones: {distance_difference:.6f}")