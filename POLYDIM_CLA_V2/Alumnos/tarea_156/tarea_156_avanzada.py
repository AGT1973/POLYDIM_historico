import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import torch
import scipy.linalg

# Definición de la función de pérdida
def loss_fn(params, x, y):
    predictions = model(params, x)
    return jnp.mean((predictions - y) ** 2)

# Modelo simplificado (por ejemplo, una red neuronal lineal)
def model(params, x):
    return jnp.dot(x, params)

# Cálculo del gradiente usando JAX
grad_loss = jax.grad(loss_fn)

# Truncamiento SVD periódico
def truncated_svd(matrix, k):
    u, s, vh = jsla.svd(matrix, full_matrices=False)
    return u[:, :k] @ jnp.diag(s[:k]) @ vh[:k, :]

# Optimización con fibrado tangente y métrica riemanniana
def optimize(params, x, y, alpha, k, num_steps):
    for i in range(num_steps):
        # Cálculo del gradiente
        grad = grad_loss(params, x, y)
        
        # Cálculo de la métrica riemanniana (ejemplo simple)
        g = jnp.eye(len(params))  # Identidad para simplicidad
        g_inv = jnp.linalg.inv(g)
        
        # Actualización de parámetros
        params = params - alpha * g_inv @ grad
        
        # Truncamiento SVD periódico
        if i % 10 == 0:  # Truncar cada 10 iteraciones
            params = truncated_svd(params, k)
        
        # Imprimir el error de pérdida
        loss = loss_fn(params, x, y)
        print(f"Iteration {i}, Loss: {loss:.6f}")
    
    return params

# Datos de ejemplo
x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = jnp.array([3.0, 5.0, 7.0])

# Inicialización de parámetros
params = jnp.array([1.0, 1.0])

# Parámetros de optimización
alpha = 0.01
k = 1
num_steps = 100

# Ejecutar la optimización
optimized_params = optimize(params, x, y, alpha, k, num_steps)