import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import torch

# Definición de un rotor de Clifford
def rotor(theta, bivector):
    return jnp.exp(0.5 * theta * bivector)

# Transformación de gradientes usando un rotor
def transform_gradient(gradient, rotor):
    return rotor @ gradient @ jnp.linalg.inv(rotor)

# Función de pérdida de ejemplo
def loss_function(params, x, y):
    predictions = jnp.dot(x, params)
    return jnp.mean((predictions - y)**2)

# Función principal para la validación empírica
@jit
def validate_transformed_gradients(params, x, y, theta, bivector):
    # Calcular el gradiente sin transformación
    grad_fn = grad(loss_function)
    gradient = grad_fn(params, x, y)
    
    # Crear el rotor
    rotor_value = rotor(theta, bivector)
    
    # Transformar el gradiente
    transformed_gradient = transform_gradient(gradient, rotor_value)
    
    # Verificar la invariancia isométrica
    original_norm = jnp.linalg.norm(gradient)
    transformed_norm = jnp.linalg.norm(transformed_gradient)
    
    return original_norm, transformed_norm

# Parámetros de ejemplo
params = jnp.array([1.0, 2.0, 3.0])
x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = jnp.array([1.0, 2.0])
theta = 1.0
bivector = jnp.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

# Validar la transformación de gradientes
original_norm, transformed_norm = validate_transformed_gradients(params, x, y, theta, bivector)

print(f"Norma original del gradiente: {original_norm}")
print(f"Norma transformada del gradiente: {transformed_norm}")

# Verificar la invariancia isométrica
assert jnp.allclose(original_norm, transformed_norm), "La invariancia isométrica no se mantiene"