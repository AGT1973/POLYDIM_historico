import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.linalg import eigh

# Definir la red neuronal usando JAX
def neural_network(x, params):
    for W, b in params:
        x = jax.nn.relu(jnp.dot(x, W) + b)
    return x

# Generar datos de entrada y perturbaciones
np.random.seed(0)
x = np.random.randn(100, 10)  # 100 muestras, 10 características
epsilon = 1e-3
eta = np.random.randn(100, 10)

# Definir los parámetros de la red
params = [
    (np.random.randn(10, 50), np.random.randn(50)),
    (np.random.randn(50, 20), np.random.randn(20)),
    (np.random.randn(20, 10), np.random.randn(10))
]

# Calcular las salidas de la red
z = neural_network(x, params)
z_perturbed = neural_network(x + epsilon * eta, params)

# Calcular la métrica inducida
def metric(z):
    dz = jax.jacfwd(neural_network)(x, params)
    return jnp.einsum('ij,ij->i', dz, dz)

g_z = metric(z)
g_z_perturbed = metric(z_perturbed)

# Calcular el gradient de la métrica inducida
grad_g_z = jax.grad(lambda x: metric(neural_network(x, params)))(x)
grad_g_z_perturbed = jax.grad(lambda x: metric(neural_network(x + epsilon * eta, params)))(x)

# Calcular la norma del gradient
norm_grad_g_z = jnp.linalg.norm(grad_g_z, axis=1)
norm_grad_g_z_perturbed = jnp.linalg.norm(grad_g_z_perturbed, axis=1)

# Imprimir resultados
print("Norma del gradient de la métrica inducida en z:", norm_grad_g_z)
print("Norma del gradient de la métrica inducida en z_perturbed:", norm_grad_g_z_perturbed)

# Validar la invariancia isométrica
is_isometric = np.allclose(norm_grad_g_z, norm_grad_g_z_perturbed, atol=1e-3)
print("Is Isometric Invariant:", is_isometric)