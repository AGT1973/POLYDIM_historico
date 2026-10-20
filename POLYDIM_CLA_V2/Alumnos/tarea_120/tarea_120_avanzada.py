import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import numpy as np
from scipy.linalg import svd

# Función para calcular la matriz Jacobiana
def jacobian(f, x):
    return jax.jacobian(f)(x)

# Función para truncar la SVD de una matriz a rango k
def truncate_svd(A, k):
    U, S, V = svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ V[:k, :]

# Función de ejemplo: una red neuronal simple
def neural_network(x, W1, b1, W2, b2):
    h1 = jnp.tanh(jnp.dot(x, W1) + b1)
    return jnp.dot(h1, W2) + b2

# Parámetros de la red neuronal
W1 = jnp.array([[0.1, 0.2], [0.3, 0.4]])
b1 = jnp.array([0.1, 0.2])
W2 = jnp.array([[0.5, 0.6], [0.7, 0.8]])
b2 = jnp.array([0.3, 0.4])

# Entrada de prueba
x = jnp.array([1.0, 2.0])

# Calcular la matriz Jacobiana
J = jacobian(lambda x: neural_network(x, W1, b1, W2, b2), x)

# Truncar la SVD de la matriz Jacobiana a rango k
k = 1
J_truncated = truncate_svd(J, k)

# Función de pérdida
def loss(W1, b1, W2, b2, x, y):
    y_pred = neural_network(x, W1, b1, W2, b2)
    return jnp.mean((y_pred - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = jit(grad(loss, argnums=(0, 1, 2, 3)))

# Datos de entrenamiento
X_train = jnp.array([[1.0, 2.0], [2.0, 3.0]])
y_train = jnp.array([[0.5, 0.6], [0.7, 0.8]])

# Entrenamiento con regularización SVD
for i in range(100):
    gradients = vmap(grad_loss, in_axes=(None, None, None, None, 0, 0))(W1, b1, W2, b2, X_train, y_train)
    W1 -= 0.01 * gradients[0].mean(axis=0)
    b1 -= 0.01 * gradients[1].mean(axis=0)
    W2 -= 0.01 * gradients[2].mean(axis=0)
    b2 -= 0.01 * gradients[3].mean(axis=0)
    
    # Truncar la SVD de la matriz Jacobiana
    J = jacobian(lambda x: neural_network(x, W1, b1, W2, b2), x)
    J_truncated = truncate_svd(J, k)
    
    # Actualizar los pesos con la matriz Jacobiana truncada
    W1 = J_truncated @ W1
    W2 = J_truncated @ W2

# Imprimir los pesos finales
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)