import jax.numpy as jnp
from jax import vmap, jit, grad
import scipy.linalg
import torch

# Definición de la función de rotación utilizando rotores de Clifford
def rotor_clifford(theta, dim):
    I = jnp.eye(dim)
    R = jnp.exp(0.5 * theta * I)
    return R

# Función de transformación lineal usando rotores
def linear_transform(x, theta, dim):
    R = rotor_clifford(theta, dim)
    return R @ x

# Función de truncamiento SVD
def truncate_svd(A, k):
    U, S, Vh = scipy.linalg.svd(A)
    S_truncated = S[:k]
    U_truncated = U[:, :k]
    Vh_truncated = Vh[:k, :]
    A_truncated = U_truncated @ (S_truncated[:, None] * Vh_truncated)
    return A_truncated

# Función de entrenamiento con truncamiento SVD periódico
@jit
def train_step(x, y, theta, dim, k, learning_rate):
    # Transformación lineal
    y_pred = vmap(linear_transform, (0, None, None))(x, theta, dim)
    
    # Cálculo del error
    loss = jnp.mean((y_pred - y) ** 2)
    
    # Gradiente del error
    grad_theta = grad(loss, theta)
    
    # Actualización de los parámetros
    theta = theta - learning_rate * grad_theta
    
    # Truncamiento SVD periódico
    theta = truncate_svd(theta, k)
    
    return theta, loss

# Ejemplo de uso
if __name__ == "__main__":
    # Parámetros
    dim = 10
    k = 5
    learning_rate = 0.01
    num_steps = 100
    
    # Datos de entrenamiento (ejemplo)
    x = jnp.random.randn(100, dim)
    y = jnp.random.randn(100, dim)
    
    # Inicialización de los parámetros
    theta = jnp.zeros((dim, dim))
    
    # Entrenamiento
    for step in range(num_steps):
        theta, loss = train_step(x, y, theta, dim, k, learning_rate)
        print(f"Step {step + 1}, Loss: {loss:.4f}")