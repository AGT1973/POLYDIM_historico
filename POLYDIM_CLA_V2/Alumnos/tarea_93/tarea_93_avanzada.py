import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental import optimizers
import numpy as np
import torch
from torch import nn, optim
from scipy.linalg import expm

# Definición de la capa topológica
class TopologicalLayer(nn.Module):
    def __init__(self, d):
        super(TopologicalLayer, self).__init__()
        self.d = d
        self.W = nn.Parameter(torch.eye(d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        # Aseguramos que W está en SO(d)
        W = expm(self.W - self.W.T)
        return torch.einsum('ij,bj->bi', W, x) + self.b

# Definición de la red neuronal topológica
class TopologicalNeuralNetwork(nn.Module):
    def __init__(self, d, hidden_layers):
        super(TopologicalNeuralNetwork, self).__init__()
        layers = [TopologicalLayer(d) for _ in range(hidden_layers)]
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(d, 1)

    def forward(self, x):
        x = self.layers(x)
        return self.output(x)

# Función de pérdida
def loss_fn(model, x, y):
    y_pred = model(x)
    return torch.mean((y_pred - y) ** 2)

# Entrenamiento de la red
def train(model, x, y, num_epochs=100, learning_rate=0.01):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, x, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Datos de ejemplo
d = 2
hidden_layers = 3
x = torch.randn(100, d)
y = torch.randn(100, 1)

# Inicialización del modelo
model = TopologicalNeuralNetwork(d, hidden_layers)

# Entrenamiento
train(model, x, y)

# Validación empírica
with torch.no_grad():
    y_pred = model(x)
    print(f'Mean Squared Error: {torch.mean((y_pred - y) ** 2).item()}')