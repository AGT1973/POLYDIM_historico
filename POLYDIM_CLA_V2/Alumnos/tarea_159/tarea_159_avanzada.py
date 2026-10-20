import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import scipy.linalg

# Definición de una capa de transformación de Lie
class LieLayer:
    def __init__(self, n, group='SO'):
        self.n = n
        self.group = group
        if group == 'SO':
            self.identity = jnp.eye(n)
        else:
            raise ValueError("Unsupported Lie group")

    def forward(self, x):
        # Aplicar una transformación de Lie (por ejemplo, una rotación)
        if self.group == 'SO':
            theta = jnp.pi / 4  # Ejemplo de ángulo de rotación
            R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                           [jnp.sin(theta), jnp.cos(theta)]])
            return jnp.dot(R, x)
        else:
            raise ValueError("Unsupported Lie group")

# Función de truncamiento SVD
def truncate_svd(W, rank):
    U, S, V = jnp.linalg.svd(W, full_matrices=False)
    S_truncated = jnp.diag(S[:rank])
    U_truncated = U[:, :rank]
    V_truncated = V[:rank, :]
    return jnp.dot(U_truncated, jnp.dot(S_truncated, V_truncated))

# Definición de la red neuronal
class LieNeuralNet:
    def __init__(self, layers, rank):
        self.layers = layers
        self.rank = rank

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            x = truncate_svd(x, self.rank)  # Aplicar truncamiento SVD
        return x

    def loss(self, x, y):
        y_pred = self.forward(x)
        return jnp.mean((y_pred - y) ** 2)

# Ejemplo de uso
n = 2
rank = 1
layers = [LieLayer(n) for _ in range(3)]
lnn = LieNeuralNet(layers, rank)

# Datos de ejemplo
x = jnp.array([[1.0, 0.0]])
y = jnp.array([[0.0, 1.0]])

# Gradiente de la función de pérdida
grad_loss = jit(grad(lnn.loss, argnums=1))

# Entrenamiento (simulación)
learning_rate = 0.01
for i in range(100):
    grad_val = grad_loss(x, y)
    x = x - learning_rate * grad_val
    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {lnn.loss(x, y)}")

# Validación final
print("Final Loss:", lnn.loss(x, y))