import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Definir una red neuronal simple
class InvariantNet(nn.Module):
    def __init__(self):
        super(InvariantNet, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Función para generar isometrías en 2D
def generate_isometry(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])

# Función de pérdida invariante
def invariant_loss(model, x, isometries, device):
    x = x.to(device)
    losses = []
    for phi in isometries:
        phi_x = jnp.dot(phi, x.detach().cpu().numpy().T).T
        phi_x = torch.tensor(phi_x, dtype=torch.float32).to(device)
        loss = torch.mean((model(x) - model(phi_x)) ** 2)
        losses.append(loss)
    return torch.mean(torch.stack(losses))

# Inicializar la red y el optimizador
model = InvariantNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generar datos de entrenamiento
x_train = torch.tensor(np.random.randn(1000, 2), dtype=torch.float32).to(device)
isometries = [generate_isometry(theta) for theta in np.linspace(0, 2 * np.pi, 10)]

# Entrenar la red
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = invariant_loss(model, x_train, isometries, device)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Validar la invariancia
x_test = torch.tensor(np.random.randn(100, 2), dtype=torch.float32).to(device)
test_isometries = [generate_isometry(theta) for theta in np.linspace(0, 2 * np.pi, 10)]
test_loss = invariant_loss(model, x_test, test_isometries, device)
print(f"Test Loss: {test_loss.item()}")