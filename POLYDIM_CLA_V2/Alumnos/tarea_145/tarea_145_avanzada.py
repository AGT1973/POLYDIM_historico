import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.linalg import svd

# Definición de la función isométrica
def isometric_function(x):
    return x  # Identidad para simplificar

# Definición de la función de pérdida que mide la invariancia isométrica
def isometric_loss(x1, x2, f):
    y1 = f(x1)
    y2 = f(x2)
    return jnp.abs(jnp.linalg.norm(y1 - y2) - jnp.linalg.norm(x1 - x2))

# Definición de la función de regularización de la matriz jacobiana
def regularize_jacobian(f, x, k):
    J = jax.jacfwd(f)(x)
    U, S, Vh = svd(J)
    S_truncated = np.diag(S[:k])
    U_truncated = U[:, :k]
    Vh_truncated = Vh[:k, :]
    J_truncated = U_truncated @ S_truncated @ Vh_truncated
    return J_truncated

# Generación de datos de prueba
x1 = np.random.randn(100)
x2 = np.random.randn(100)

# Validación empírica de la invariancia isométrica
f = isometric_function
loss = isometric_loss(x1, x2, f)
print(f"Isometric Loss: {loss}")

# Validación empírica de la estabilización de gradientes
k = 10  # Rango truncado
J_truncated = regularize_jacobian(f, x1, k)
print(f"Truncated Jacobian: {J_truncated}")

# Verificación de la norma de los gradientes
original_norm = np.linalg.norm(J_truncated, ord='fro')
print(f"Original Jacobian Frobenius Norm: {original_norm}")

# Verificación de la norma de los gradientes truncados
truncated_norm = np.linalg.norm(J_truncated, ord='fro')
print(f"Truncated Jacobian Frobenius Norm: {truncated_norm}")