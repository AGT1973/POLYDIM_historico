import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.linalg import svd

# Función para aplicar una transformación lineal con truncamiento SVD
def linear_transform_with_truncation(X, W, truncation_rank):
    # Aplicar la transformación lineal
    Y = jnp.dot(X, W)
    
    # Realizar la descomposición SVD
    U, S, Vt = svd(Y)
    
    # Truncar el rango
    S_truncated = S[:truncation_rank]
    U_truncated = U[:, :truncation_rank]
    Vt_truncated = Vt[:truncation_rank, :]
    
    # Reconstruir la matriz truncada
    Y_truncated = jnp.dot(U_truncated, jnp.dot(jnp.diag(S_truncated), Vt_truncated))
    
    return Y_truncated

# Función para aplicar una transformación no lineal con rotores Clifford
def nonlinear_transform_with_clifford(X, W, theta):
    # Aplicar la transformación lineal
    Y = jnp.dot(X, W)
    
    # Aplicar la rotación Clifford
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    Y_rotated = jnp.dot(Y, R)
    
    return Y_rotated

# Función para validar la invariancia topológica
def validate_topological_invariance(X, W, theta, truncation_rank):
    # Aplicar la transformación lineal con truncamiento SVD
    Y_truncated = linear_transform_with_truncation(X, W, truncation_rank)
    
    # Aplicar la transformación no lineal con rotores Clifford
    Y_rotated = nonlinear_transform_with_clifford(X, W, theta)
    
    # Calcular las descomposiciones SVD
    U_truncated, S_truncated, Vt_truncated = svd(Y_truncated)
    U_rotated, S_rotated, Vt_rotated = svd(Y_rotated)
    
    # Comparar los espectros singulares
    spectral_distance = jnp.linalg.norm(S_truncated - S_rotated)
    
    return spectral_distance

# Parámetros
X = jnp.array(np.random.randn(100, 10))  # Datos de entrada
W = jnp.array(np.random.randn(10, 10))    # Matriz de peso
theta = jnp.pi / 4                        # Ángulo de rotación
truncation_rank = 5                       # Rango de truncación

# Validar la invariancia topológica
spectral_distance = validate_topological_invariance(X, W, theta, truncation_rank)
print(f"Spectral Distance: {spectral_distance}")