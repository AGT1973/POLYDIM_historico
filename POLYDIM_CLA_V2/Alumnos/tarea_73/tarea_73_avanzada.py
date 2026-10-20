import jax
import jax.numpy as jnp
import scipy
import torch

# Definir una red neuronal simple usando JAX
def simple_nn(x, params):
    W1, b1, W2, b2 = params
    hidden = jax.nn.relu(jnp.dot(x, W1) + b1)
    output = jnp.dot(hidden, W2) + b2
    return output

# Definir una isometría (rotación)
def isometry(x, theta):
    # Rotación 2D (ejemplo)
    c, s = jnp.cos(theta), jnp.sin(theta)
    R = jnp.array([[c, -s], [s, c]])
    return jnp.dot(R, x)

# Inicializar parámetros de la red neuronal
key = jax.random.PRNGKey(0)
W1 = jax.random.normal(key, (2, 10))
b1 = jax.random.normal(key, (10,))
W2 = jax.random.normal(key, (10, 2))
b2 = jax.random.normal(key, (2,))
params = (W1, b1, W2, b2)

# Definir la función de validación
def validate_isometric_invariance(x, theta, params):
    # Aplicar la isometría a la entrada
    x_rotated = isometry(x, theta)
    # Predecir con la red neuronal
    y1 = simple_nn(x_rotated, params)
    y2 = isometry(simple_nn(x, params), theta)
    # Calcular la diferencia
    diff = jnp.linalg.norm(y1 - y2)
    return diff

# Ejemplo de validación
x = jnp.array([1.0, 2.0])
theta = jnp.pi / 4  # 45 grados

# Validar la invariancia isométrica
diff = validate_isometric_invariance(x, theta, params)
print(f"Diferencia entre f(T(x)) y T(f(x)): {diff:.6f}")