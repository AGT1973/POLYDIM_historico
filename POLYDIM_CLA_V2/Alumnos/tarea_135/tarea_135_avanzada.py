import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
from scipy.linalg import svd

# Definición de la red neuronal simétrica
class SymmetricNeuralNet:
    def __init__(self, layers, rank_truncation):
        self.layers = layers
        self.rank_truncation = rank_truncation

    @jit
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.truncate_svd(x)
        return x

    @jit
    def truncate_svd(self, x):
        U, S, Vh = svd(x, full_matrices=False)
        S_truncated = jnp.diag(S[:self.rank_truncation])
        return U[:, :self.rank_truncation] @ S_truncated @ Vh[:self.rank_truncation, :]

# Definición de una capa lineal con matriz jacobiana ortogonal
class OrthogonalLinearLayer:
    def __init__(self, in_features, out_features):
        self.W = jax.random.orthogonal(jax.random.PRNGKey(0), (out_features, in_features))

    def __call__(self, x):
        return jnp.dot(self.W, x)

# Creación de la red neuronal
layers = [OrthogonalLinearLayer(2, 2), OrthogonalLinearLayer(2, 2)]
rank_truncation = 1  # Ejemplo de truncamiento de rango
net = SymmetricNeuralNet(layers, rank_truncation)

# Validación empírica
input_data = jnp.array([[1.0, 0.0], [0.0, 1.0]])
output_data = net.forward(input_data)

# Verificación de la invariancia isométrica
def isometry_check(input_data, output_data):
    input_metric = jnp.einsum('ij,ik->jk', input_data, input_data)
    output_metric = jnp.einsum('ij,ik->jk', output_data, output_data)
    return jnp.allclose(input_metric, output_metric, atol=1e-5)

isometry = isometry_check(input_data, output_data)
print(f"Invariancia isométrica: {isometry}")