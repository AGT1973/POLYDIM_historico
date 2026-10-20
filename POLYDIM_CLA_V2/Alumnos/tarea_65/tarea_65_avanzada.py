import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import numpy as np
import scipy.spatial.distance as distance

# Definir una función para calcular la distancia Hausdorff
def hausdorff_distance(A, B):
    dist_matrix = distance.cdist(A, B, 'euclidean')
    max_A_to_B = np.max(np.min(dist_matrix, axis=1))
    max_B_to_A = np.max(np.min(dist_matrix, axis=0))
    return max(max_A_to_B, max_B_to_A)

# Definir una función para aplicar un rotor de Clifford
def apply_clifford_rotor(H, theta, B):
    # Rotor en la Álgebra de Clifford
    R = jnp.exp(0.5 * theta * B)
    # Aplicar el rotor a H
    H_rotated = R * H * R.inv()
    return H_rotated

# Definir una función para calcular la invariancia topológica
def invariance_metric(H, theta, B):
    # Aplicar el rotor
    H_rotated = apply_clifford_rotor(H, theta, B)
    # Calcular la distancia Hausdorff
    d_H = hausdorff_distance(H, H_rotated)
    # Calcular el diámetro de H
    diam_H = np.max(distance.pdist(H, 'euclidean'))
    # Calcular la métrica de invariancia topológica
    invariance = 1 - (d_H / diam_H)
    return invariance

# Ejemplo de uso
# Generar datos de entrada
X = np.random.rand(100, 2)

# Definir una red neuronal simple (ejemplo con una capa lineal)
def simple_neural_network(X):
    W = np.random.rand(2, 2)
    b = np.random.rand(2)
    H = jnp.dot(X, W) + b
    return H

# Calcular la representación intermedia
H = simple_neural_network(X)

# Definir el ángulo de rotación y el bivector
theta = 1.0
B = jnp.array([[0, 1], [-1, 0]])

# Calcular la invariancia topológica
invariance = invariance_metric(H, theta, B)
print(f"Invariancia topológica: {invariance}")