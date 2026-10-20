import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
from scipy.stats import ortho_group

# Función para generar una matriz ortogonal aleatoria
def generate_orthogonal_matrix(dim):
    return ortho_group.rvs(dim)

# Función de red neuronal con capa invariante a transformaciones de grupo de Lie
def invariant_neural_network(x, W, b):
    # Aplicar transformación ortogonal
    x_transformed = jnp.dot(W, x)
    # Capa lineal
    return jnp.dot(W, x_transformed) + b

# Función de pérdida
def loss_fn(params, x, y):
    W, b = params
    predictions = invariant_neural_network(x, W, b)
    return jnp.mean((predictions - y) ** 2)

# Gradiente de la función de pérdida
grad_loss_fn = grad(loss_fn)

# Generar datos de ejemplo
dim = 10
num_samples = 100
X = jnp.array(torch.randn(num_samples, dim))
Y = jnp.array(torch.randn(num_samples, dim))

# Inicializar parámetros
W = generate_orthogonal_matrix(dim)
b = jnp.zeros(dim)

# Optimización usando JAX
@jit
def update(params, x, y, lr=0.01):
    gradients = grad_loss_fn(params, x, y)
    return [(p - lr * g) for p, g in zip(params, gradients)]

# Entrenamiento
num_epochs = 100
params = (W, b)
for epoch in range(num_epochs):
    params = update(params, X, Y)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss_fn(params, X, Y)}")

# Validación de la invariancia isométrica
def check_isometric_invariance(W):
    # Generar un vector aleatorio
    x = jnp.array(torch.randn(dim))
    # Aplicar transformación ortogonal
    x_transformed = jnp.dot(W, x)
    # Comprobar si la norma se mantiene
    return jnp.allclose(jnp.linalg.norm(x), jnp.linalg.norm(x_transformed))

# Verificar invariancia isométrica
invariance_check = check_isometric_invariance(params[0])
print(f"Invariancia isométrica: {invariance_check}")