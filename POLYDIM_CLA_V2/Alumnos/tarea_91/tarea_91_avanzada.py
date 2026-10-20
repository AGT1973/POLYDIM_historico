import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
from scipy.spatial.transform import Rotation as R

# Definición de la capa invariante a isometrías
def invariant_layer(x, W, b, pool_type='max'):
    # W debe ser una matriz diagonal
    W_diagonal = jnp.diag(W)
    # Aplicar la transformación lineal
    z = jnp.dot(W_diagonal, x) + b
    # Aplicar la función de activación (ReLU)
    z_activated = jnp.maximum(z, 0)
    # Aplicar el pooling global
    if pool_type == 'max':
        y = jnp.max(z_activated, axis=0)
    elif pool_type == 'mean':
        y = jnp.mean(z_activated, axis=0)
    else:
        raise ValueError("Invalid pooling type. Use 'max' or 'mean'.")
    return y

# Función para generar una isometría aleatoria
def random_isometry(dim):
    # Generar una matriz de rotación aleatoria
    rotation = R.random().as_matrix()[:dim, :dim]
    # Generar un vector de traslación aleatorio
    translation = jnp.array(torch.randn(dim))
    return rotation, translation

# Función para aplicar una isometría a un vector
def apply_isometry(x, rotation, translation):
    return jnp.dot(rotation, x) + translation

# Función para verificar la invariancia
def check_invariance(x, W, b, pool_type='max'):
    # Generar una isometría aleatoria
    rotation, translation = random_isometry(x.shape[0])
    # Aplicar la isometría a x
    x_transformed = apply_isometry(x, rotation, translation)
    # Calcular la salida antes y después de la isometría
    y_original = invariant_layer(x, W, b, pool_type)
    y_transformed = invariant_layer(x_transformed, W, b, pool_type)
    # Verificar si las salidas son iguales
    return jnp.allclose(y_original, y_transformed, atol=1e-6)

# Parámetros de la capa
dim = 3
W = jnp.array([1.0, 2.0, 3.0])  # Diagonal de la matriz de pesos
b = jnp.array([0.5, -0.5, 0.0])  # Vector de sesgos
x = jnp.array([1.0, 2.0, 3.0])  # Entrada de prueba

# Verificar la invariancia
invariant = check_invariance(x, W, b, pool_type='max')
print(f"La capa es invariante a isometrías: {invariant}")