import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ortho_group

# Definir una red neuronal simple con PyTorch
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 16 * 28 * 28)
        x = self.fc1(x)
        return x

# Función para generar transformaciones isométricas aleatorias
def generate_isometric_transformations(batch_size, input_dim):
    transformations = []
    for _ in range(batch_size):
        # Generar una matriz ortogonal aleatoria
        R = ortho_group.rvs(input_dim)
        transformations.append(R)
    return jnp.array(transformations)

# Función para aplicar transformaciones isométricas a los datos
@jit
def apply_isometric_transformations(transformations, data):
    return vmap(lambda R, x: jnp.dot(R, x))(transformations, data)

# Función para verificar la invariancia isométrica
def check_isometric_invariance(model, data, transformations, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    
    # Aplicar transformaciones isométricas
    transformed_data = apply_isometric_transformations(transformations, data)
    
    # Convertir a tensores de PyTorch
    data_tensor = torch.tensor(data, dtype=torch.float32)
    transformed_data_tensor = torch.tensor(transformed_data, dtype=torch.float32)
    
    # Obtener las salidas de la red
    output = model(data_tensor.unsqueeze(1))
    transformed_output = model(transformed_data_tensor.unsqueeze(1))
    
    # Calcular la pérdida
    loss = criterion(output, transformed_output)
    
    # Retropropagación y optimización
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Crear una instancia de la red neuronal
model = SimpleCNN()

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generar datos de entrada aleatorios
batch_size = 32
input_dim = 28 * 28
data = jnp.random.rand(batch_size, input_dim)

# Generar transformaciones isométricas aleatorias
transformations = generate_isometric_transformations(batch_size, input_dim)

# Verificar la invariancia isométrica
loss = check_isometric_invariance(model, data, transformations, criterion, optimizer)
print(f"Loss: {loss}")