import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import numpy as np
from scipy.linalg import svd

# Definición de la función de pérdida
def loss_fn(params, x, y):
    predictions = model(params, x)
    return jnp.mean((predictions - y) ** 2)

# Definición del modelo (ejemplo: una red neuronal simple)
def model(params, x):
    return jnp.dot(params, x)

# Función para truncar la SVD
def truncated_svd(G, k):
    U, S, Vt = svd(G)
    return U[:, :k] @ jnp.diag(S[:k]) @ Vt[:k, :]

# Función de entrenamiento con regularización del fibrado tangente
@jit
def train_step(params, x, y, learning_rate, k):
    # Calcular el gradiente
    grads = grad(loss_fn)(params, x, y)
    
    # Truncar la SVD del gradiente
    truncated_grads = truncated_svd(grads, k)
    
    # Actualizar los parámetros
    params = params - learning_rate * truncated_grads
    return params

# Datos de ejemplo
x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
y = jnp.array([1.0, 2.0])

# Inicialización de los parámetros
params = jnp.array([0.1, 0.2])

# Hiperparámetros
learning_rate = 0.01
k = 1  # Rango de truncación

# Entrenamiento
for i in range(100):
    params = train_step(params, x, y, learning_rate, k)
    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss_fn(params, x, y)}")

print("Final parameters:", params)