import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import scipy.linalg

# Definición de la red neuronal simple
def neural_network(params, x):
    W1, b1, W2, b2 = params
    h1 = jax.nn.relu(jnp.dot(x, W1) + b1)
    return jnp.dot(h1, W2) + b2

# Función de pérdida
def loss(params, x, y):
    y_pred = neural_network(params, x)
    return jnp.mean((y_pred - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = jit(grad(loss))

# Truncamiento SVD
def truncate_svd(matrix, rank):
    U, S, Vh = jnp.linalg.svd(matrix, full_matrices=False)
    S_truncated = jnp.diag(S[:rank])
    U_truncated = U[:, :rank]
    Vh_truncated = Vh[:rank, :]
    return jnp.dot(U_truncated, jnp.dot(S_truncated, Vh_truncated))

# Inicialización de parámetros
def initialize_params(input_dim, hidden_dim, output_dim):
    W1 = jax.random.normal(jax.random.PRNGKey(0), (input_dim, hidden_dim))
    b1 = jax.random.normal(jax.random.PRNGKey(1), (hidden_dim,))
    W2 = jax.random.normal(jax.random.PRNGKey(2), (hidden_dim, output_dim))
    b2 = jax.random.normal(jax.random.PRNGKey(3), (output_dim,))
    return (W1, b1, W2, b2)

# Validación empírica
def validate_stability(params, x, y, rank):
    # Calcular gradientes
    gradients = grad_loss(params, x, y)
    
    # Aplicar truncamiento SVD a los gradientes
    W1_grad, b1_grad, W2_grad, b2_grad = gradients
    W1_truncated = truncate_svd(W1_grad, rank)
    W2_truncated = truncate_svd(W2_grad, rank)
    
    # Verificar invariancia isométrica
    norm_before = jnp.linalg.norm(W1_grad) + jnp.linalg.norm(W2_grad)
    norm_after = jnp.linalg.norm(W1_truncated) + jnp.linalg.norm(W2_truncated)
    
    return norm_before, norm_after

# Parámetros de entrada
input_dim = 10
hidden_dim = 50
output_dim = 1
rank = 5

# Datos de ejemplo
x = jax.random.normal(jax.random.PRNGKey(4), (100, input_dim))
y = jax.random.normal(jax.random.PRNGKey(5), (100, output_dim))

# Inicializar parámetros
params = initialize_params(input_dim, hidden_dim, output_dim)

# Validar estabilidad
norm_before, norm_after = validate_stability(params, x, y, rank)
print(f"Norma del gradiente antes del truncamiento SVD: {norm_before}")
print(f"Norma del gradiente después del truncamiento SVD: {norm_after}")