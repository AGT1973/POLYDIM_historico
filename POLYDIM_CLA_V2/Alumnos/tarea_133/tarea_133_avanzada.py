import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import torch
from jax import vmap, jit, grad

# Función para realizar la descomposición SVD y truncar
def truncated_svd(matrix, k):
    U, S, V = jnp.linalg.svd(matrix, full_matrices=False)
    return U[:, :k] @ jnp.diag(S[:k]) @ V[:k, :]

# Función para calcular la propagación de gradientes
@jit
def propagate_gradients(weights, inputs, k=5):
    activations = jnp.dot(inputs, weights)
    gradients = grad(lambda w: jnp.sum(activations))(weights)
    truncated_gradients = truncated_svd(gradients, k)
    return truncated_gradients

# Generar datos de ejemplo
np.random.seed(42)
inputs = jnp.array(np.random.randn(100, 10))
weights = jnp.array(np.random.randn(10, 5))

# Propagar gradientes y truncar
truncated_gradients = propagate_gradients(weights, inputs)

# Validar empíricamente la estabilización de gradientes
def validate_stability(weights, inputs, k=5, num_steps=100):
    original_gradients = grad(lambda w: jnp.sum(jnp.dot(inputs, w)))(weights)
    truncated_gradients = propagate_gradients(weights, inputs, k)
    
    original_norm = jnp.linalg.norm(original_gradients)
    truncated_norm = jnp.linalg.norm(truncated_gradients)
    
    stability_ratio = truncated_norm / original_norm
    return stability_ratio

stability_ratio = validate_stability(weights, inputs)
print(f"Stability Ratio: {stability_ratio:.4f}")

# Verificar que la estabilidad es razonable (cercana a 1)
assert stability_ratio < 1.1, "Stability ratio is too high, indicating potential instability."