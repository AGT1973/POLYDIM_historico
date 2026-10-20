import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import torch
from scipy.spatial.transform import Rotation as R

# Definir una función radial invariante a rotaciones
def radial_function(x):
    return jnp.exp(-jnp.linalg.norm(x))

# Definir una red neuronal simple con capa radial
def neural_network(x):
    return radial_function(x)

# Generar datos de ejemplo
x = jnp.array([1.0, 2.0, 3.0])

# Definir una rotación aleatoria en SO(3)
def random_rotation():
    return R.random().as_matrix()

# Aplicar la rotación a los datos
def apply_rotation(x, rotation_matrix):
    return jnp.dot(rotation_matrix, x)

# Función para comprobar la invariancia
def check_invariance(x, neural_network, n_samples=100):
    original_output = neural_network(x)
    invariance_errors = []
    for _ in range(n_samples):
        rotation_matrix = random_rotation()
        rotated_x = apply_rotation(x, rotation_matrix)
        rotated_output = neural_network(rotated_x)
        invariance_errors.append(jnp.abs(original_output - rotated_output))
    return jnp.array(invariance_errors)

# Compilar la función de comprobación con JAX
check_invariance_jit = jit(check_invariance)

# Ejecutar la comprobación
invariance_errors = check_invariance_jit(x, neural_network)

# Imprimir los resultados
print("Errores de invariancia (deben ser cercanos a cero):")
print(invariance_errors)

# Verificar la media de los errores
mean_error = jnp.mean(invariance_errors)
print(f"Media de los errores de invariancia: {mean_error}")