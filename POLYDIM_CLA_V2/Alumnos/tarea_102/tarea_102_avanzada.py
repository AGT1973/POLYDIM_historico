import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize

# Definición de la red neuronal usando PyTorch
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Función de pérdida
def loss_fn(model, x, y):
    y_pred = model(x)
    return torch.mean((y_pred - y) ** 2)

# Función de pérdida regularizada
def regularized_loss_fn(model, x, y, lambda_reg):
    loss = loss_fn(model, x, y)
    model_grad = grad(lambda params: loss_fn(model, x, y))(model.parameters())
    reg_term = lambda_reg * jnp.sum(jnp.array([jnp.linalg.norm(g) for g in model_grad]))
    return loss + reg_term

# Datos de ejemplo
x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=torch.float32)
y = torch.tensor([[3.0], [5.0], [7.0]], dtype=torch.float32)

# Instanciación del modelo
model = SimpleNN()

# Optimizador
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Hiperparámetro de regularización
lambda_reg = 0.01

# Entrenamiento
for epoch in range(1000):
    optimizer.zero_grad()
    loss = regularized_loss_fn(model, x, y, lambda_reg)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Evaluación final
with torch.no_grad():
    y_pred = model(x)
    print(f'Predicted: {y_pred}, True: {y}')