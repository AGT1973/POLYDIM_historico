import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.lax import svd
from scipy.linalg import orth
import torch

# Función para generar una matriz ortogonal aleatoria (SO(n))
def random_orthogonal_matrix(n, key):
    key1, key2 = jax.random.split(key)
    Q, R = jnp.linalg.qr(jax.random.normal(key1, (n, n)))
    return Q * jnp.sign(jnp.diag(R))

# Capa de red neuronal isométrica
class IsometricLayer:
    def __init__(self, n, key):
        self.W = random_orthogonal_matrix(n, key)
    
    def forward(self, x):
        return jnp.dot(self.W, x)

# Función de pérdida
def loss_fn(params, x, y):
    x = params['layer'](x)
    return jnp.mean((x - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = grad(loss_fn)

# Función de truncamiento SVD periódico
def svd_truncation(W, k):
    U, S, Vh = svd(W)
    S_truncated = jnp.concatenate([S[:k], jnp.zeros(S.shape[0] - k)])
    return jnp.dot(U, jnp.dot(jnp.diag(S_truncated), Vh))

# Ejemplo de entrenamiento
key = jax.random.PRNGKey(0)
n = 10
k = 5
layer = IsometricLayer(n, key)
params = {'layer': layer}

# Datos de ejemplo
x = jax.random.normal(key, (n,))
y = jax.random.normal(key, (n,))

# Entrenamiento
for i in range(100):
    grads = grad_loss(params, x, y)
    params['layer'].W -= 0.01 * grads['layer'].W
    params['layer'].W = svd_truncation(params['layer'].W, k)

# Validación
x_test = jax.random.normal(key, (n,))
y_test = jax.random.normal(key, (n,))
y_pred = params['layer'].forward(x_test)
loss = loss_fn(params, x_test, y_test)
print(f"Loss: {loss}")