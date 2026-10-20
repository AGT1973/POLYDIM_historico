import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import scipy.linalg
import torch

# Función para realizar la descomposición SVD y truncar
def truncated_svd(A, k):
    U, S, Vh = jnp.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ jnp.diag(S[:k]) @ Vh[:k, :]

# Función para aplicar un rotor de Clifford
def apply_clifford_rotor(x, rotor):
    return rotor @ x @ rotor.T

# Función para componer capas de la red neuronal
def compose_layers(layer1, layer2, k):
    # Composición tensorial
    composed_layer = jnp.einsum('ij,jk->ik', layer1, layer2)
    # Truncación SVD
    return truncated_svd(composed_layer, k)

# Función de pérdida (ejemplo)
def loss_fn(params, x, y):
    predictions = jnp.dot(x, params)
    return jnp.mean((predictions - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = grad(loss_fn)

# Ejemplo de validación empírica
def validate_topology():
    # Generar datos de ejemplo
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([1.0, 2.0])
    params = jnp.array([[0.5, 0.5], [0.5, 0.5]])

    # Aplicar rotor de Clifford
    rotor = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    x_rotated = vmap(apply_clifford_rotor, in_axes=(0, None))(x, rotor)

    # Componer capas con truncamiento SVD
    layer1 = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    layer2 = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    composed_layer = compose_layers(layer1, layer2, k=1)

    # Calcular gradientes
    gradients = grad_loss(params, x_rotated, y)

    # Mostrar resultados
    print("Rotated Input:", x_rotated)
    print("Composed Layer:", composed_layer)
    print("Gradients:", gradients)

# Ejecutar la validación
validate_topology()