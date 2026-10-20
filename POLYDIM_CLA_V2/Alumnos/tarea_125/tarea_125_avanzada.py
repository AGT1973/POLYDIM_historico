import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import numpy as np
from scipy.linalg import svd

# Definición de la función para aplicar un rotor de Clifford
def apply_rotor(v, rotor):
    return rotor @ v @ rotor.T

# Definición de la función para truncar SVD
def truncate_svd(T, k):
    U, S, Vt = svd(T, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    return U_k @ np.diag(S_k) @ Vt_k

# Definición de la función de pérdida
def loss_fn(params, X, y):
    W1, W2 = params
    hidden = jnp.dot(X, W1)
    hidden = jax.nn.relu(hidden)
    hidden = truncate_svd(hidden, 10)  # Truncamiento SVD
    output = jnp.dot(hidden, W2)
    return jnp.mean((output - y) ** 2)

# Definición de la función de entrenamiento
@jit
def train_step(params, X, y, optimizer_state, optimizer):
    grads = grad(loss_fn)(params, X, y)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    params = jax.tree_multimap(lambda p, u: p - u, params, updates)
    return params, optimizer_state

# Inicialización de parámetros
np.random.seed(42)
W1 = np.random.randn(784, 128)
W2 = np.random.randn(128, 10)
params = [W1, W2]

# Inicialización de datos
X = np.random.randn(100, 784)
y = np.random.randint(0, 10, size=(100,))

# Inicialización del optimizador
from jax.experimental import optimizers
optimizer = optimizers.adam(0.001)
optimizer_state = optimizer.init(params)

# Entrenamiento
for epoch in range(100):
    params, optimizer_state = train_step(params, X, y, optimizer_state, optimizer)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss_fn(params, X, y)}")

# Validación final
print(f"Final Loss: {loss_fn(params, X, y)}")