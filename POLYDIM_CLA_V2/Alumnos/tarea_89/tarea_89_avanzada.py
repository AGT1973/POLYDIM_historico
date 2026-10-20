import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.linalg import expm
from jax import vmap, jit, grad

# Definición del rotor de Clifford
def rotor(v, theta):
    """
    Genera un rotor de Clifford para una rotación de ángulo theta en el plano definido por v.
    """
    v = v / jnp.linalg.norm(v)
    R = jnp.cos(theta / 2) - jnp.sin(theta / 2) * v
    return R

# Aplicación del rotor a un vector
def apply_rotor(R, v):
    """
    Aplica el rotor R al vector v.
    """
    return R * v * jnp.conj(R)

# Función de pérdida (ejemplo: MSE)
def loss_fn(params, x, y):
    """
    Función de pérdida para la red neuronal.
    """
    predictions = predict(params, x)
    return jnp.mean((predictions - y) ** 2)

# Predicción de la red neuronal (ejemplo: regresión lineal)
def predict(params, x):
    """
    Predicción de la red neuronal.
    """
    W, b = params
    return jnp.dot(x, W) + b

# Generación de datos sintéticos
def generate_data(n_samples, n_features):
    """
    Genera datos sintéticos.
    """
    X = jnp.random.randn(n_samples, n_features)
    y = jnp.random.randn(n_samples, 1)
    return X, y

# Inicialización de parámetros de la red neuronal
def init_params(n_features):
    """
    Inicializa los parámetros de la red neuronal.
    """
    W = jax.random.normal(jax.random.PRNGKey(0), (n_features, 1))
    b = jax.random.normal(jax.random.PRNGKey(1), (1,))
    return (W, b)

# Validación de la invariancia isométrica
def validate_invariance(params, X, y, theta):
    """
    Valida la invariancia isométrica aplicando rotores de Clifford.
    """
    n_samples, n_features = X.shape
    R = rotor(jnp.array([1, 0, 0]), theta)
    X_rotated = vmap(lambda x: apply_rotor(R, x))(X)
    
    loss_original = loss_fn(params, X, y)
    loss_rotated = loss_fn(params, X_rotated, y)
    
    print(f"Loss Original: {loss_original}")
    print(f"Loss Rotated: {loss_rotated}")

# Configuración
n_samples = 1000
n_features = 3
theta = jnp.pi / 4

# Generación de datos y inicialización de parámetros
X, y = generate_data(n_samples, n_features)
params = init_params(n_features)

# Validación
validate_invariance(params, X, y, theta)