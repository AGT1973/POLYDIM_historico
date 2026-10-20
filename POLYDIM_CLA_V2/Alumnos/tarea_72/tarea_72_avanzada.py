import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import norm

# Definición de la red neuronal
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Función para calcular el gradiente
def compute_gradient(model, x, y):
    model.zero_grad()
    output = model(x)
    loss = torch.mean((output - y) ** 2)
    loss.backward()
    gradients = []
    for param in model.parameters():
        gradients.append(param.grad.view(-1))
    return torch.cat(gradients)

# Función para proyectar el gradiente sobre el espacio tangente
def project_gradient(gradient, metric):
    # Aquí se define la métrica, por ejemplo, la norma euclidiana
    projected_gradient = gradient / norm(gradient, ord=metric)
    return projected_gradient

# Función de entrenamiento
@jit
def train_step(model, x, y, optimizer, metric):
    gradient = compute_gradient(model, x, y)
    projected_gradient = project_gradient(gradient, metric)
    # Actualización de los pesos
    for param, grad in zip(model.parameters(), vmap(lambda g: g.reshape(param.shape))(projected_gradient)):
        param.grad = torch.tensor(grad, requires_grad=True)
    optimizer.step()

# Datos de ejemplo
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# Inicialización del modelo y del optimizador
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Métrica (norma euclidiana)
metric = 2

# Entrenamiento
for epoch in range(100):
    train_step(model, x, y, optimizer, metric)
    if epoch % 10 == 0:
        output = model(x)
        loss = torch.mean((output - y) ** 2)
        print(f'Epoch {epoch}, Loss: {loss.item()}')