import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import quad

# Definir una red neuronal simple con JAX
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Función de pérdida original (ejemplo: MSE)
def loss_fn(model, x, y):
    y_pred = model(x)
    return torch.mean((y_pred - y) ** 2)

# Regularización del fibrado tangente
def tangent_bundle_reg(model, x):
    x = torch.tensor(x, requires_grad=True)
    y_pred = model(x)
    grad_y_pred = torch.autograd.grad(y_pred, x, create_graph=True)[0]
    reg = torch.sum(grad_y_pred ** 2)
    return reg

# Función de pérdida total
def total_loss(model, x, y, lambda_reg):
    loss = loss_fn(model, x, y)
    reg = tangent_bundle_reg(model, x)
    return loss + lambda_reg * reg

# Entrenamiento de la red neuronal
def train(model, x, y, lambda_reg, num_epochs=100, learning_rate=0.01):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = total_loss(model, x, y, lambda_reg)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Datos de ejemplo
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# Crear y entrenar el modelo
model = SimpleNet()
lambda_reg = 0.1
train(model, x, y, lambda_reg)

# Validación empírica de la regularización
def empirical_validation(model, x, y, lambda_reg):
    loss = loss_fn(model, x, y).item()
    reg = tangent_bundle_reg(model, x).item()
    total_loss_val = loss + lambda_reg * reg
    print(f'Loss: {loss}, Regularization: {reg}, Total Loss: {total_loss_val}')

empirical_validation(model, x, y, lambda_reg)