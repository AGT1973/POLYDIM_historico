import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import torch
import scipy
import numpy as np

# Definición de la función de pérdida (ejemplo: MSE)
def loss_function(w, x, y):
    y_pred = jnp.dot(x, w)
    return jnp.mean((y - y_pred) ** 2)

# Generación de datos de ejemplo
np.random.seed(0)
x = np.random.randn(100, 10)
y = np.random.randn(100)

# Conversión a JAX arrays
x = jnp.array(x)
y = jnp.array(y)

# Inicialización de los pesos
w = jnp.zeros(10)

# Cálculo del gradiente de la función de pérdida
grad_loss = grad(loss_function, argnums=0)

# Función para calcular la norma del gradiente
def norm_gradient(w, x, y):
    g = grad_loss(w, x, y)
    return jnp.linalg.norm(g)

# Vectorización y JIT compilation para eficiencia
norm_gradient_vmap = vmap(norm_gradient, in_axes=(None, 0, 0))
norm_gradient_jit = jit(norm_gradient_vmap)

# Cálculo de la norma del gradiente a lo largo del entrenamiento
def compute_norms(w, x, y, num_steps):
    norms = []
    for _ in range(num_steps):
        g = grad_loss(w, x, y)
        w = w - 0.01 * g  # Simple SGD update
        norms.append(norm_gradient(w, x, y))
    return jnp.array(norms)

# Ejecución del cálculo
num_steps = 100
norms = compute_norms(w, x, y, num_steps)

# Impresión de los resultados
print("Normas de los gradientes a lo largo del entrenamiento:")
print(norms)

# Cálculo de la varianza de las normas de los gradientes
variance = jnp.var(norms)
print("Varianza de las normas de los gradientes:", variance)