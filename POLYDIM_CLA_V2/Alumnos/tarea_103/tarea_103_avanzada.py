import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import scipy
from scipy.spatial.distance import pdist, squareform

# Definir la red neuronal (ejemplo simple)
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(2, 10)
        self.fc2 = torch.nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Función para calcular la desviación de la invarianza isométrica
@jit
def isometric_deviation(x, y, model):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    x_transformed = model(x)
    y_transformed = model(y)
    distance_input = jnp.linalg.norm(x - y)
    distance_output = jnp.linalg.norm(x_transformed - y_transformed)
    return (distance_output - distance_input) ** 2

# Generar datos de prueba
def generate_data(num_samples, input_dim):
    return jnp.random.randn(num_samples, input_dim)

# Función principal para evaluar la invarianza isométrica
def evaluate_isometric_invariance(model, num_samples=1000, input_dim=2):
    data = generate_data(num_samples, input_dim)
    distances = squareform(pdist(data, 'euclidean'))
    pairs = [(data[i], data[j]) for i in range(num_samples) for j in range(i + 1, num_samples)]
    deviations = vmap(lambda x, y: isometric_deviation(x, y, model))(pairs[0], pairs[1])
    mean_deviation = jnp.mean(deviations)
    return mean_deviation

# Crear y entrenar un modelo simple (por simplicidad, omitimos el entrenamiento)
model = SimpleNN()

# Evaluar la invarianza isométrica
mean_deviation = evaluate_isometric_invariance(model)
print(f"Mean Isometric Deviation: {mean_deviation:.6f}")