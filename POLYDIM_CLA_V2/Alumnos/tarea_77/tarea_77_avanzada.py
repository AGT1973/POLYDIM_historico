import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
from scipy.linalg import expm

# Definición de la red neuronal
class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(10, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.fc3 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Función para calcular el rotor de Clifford
def clifford_rotor(theta, n):
    # Generamos una matriz de rotación en n dimensiones
    R = jnp.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            R = R @ expm(jnp.array([[0, -theta], [theta, 0]]))
    return R

# Función para aplicar el rotor a los gradientes
def apply_rotor(grads, rotor):
    return rotor @ grads

# Función de pérdida
def loss_fn(params, x, y):
    model = NeuralNet()
    model.load_state_dict(params)
    y_pred = model(x)
    loss = torch.nn.MSELoss()(y_pred, y)
    return loss

# Función para calcular los gradientes y aplicar el rotor
def compute_gradients_and_apply_rotor(params, x, y, theta):
    loss_value, grads = jax.value_and_grad(loss_fn)(params, x, y)
    rotor = clifford_rotor(theta, len(grads))
    stabilized_grads = apply_rotor(grads, rotor)
    return loss_value, stabilized_grads

# Inicialización
x = torch.randn(100, 10)
y = torch.randn(100, 1)
model = NeuralNet()
params = model.state_dict()

# Parámetros del rotor
theta = 0.1

# Cálculo de gradientes y aplicación del rotor
loss_value, stabilized_grads = compute_gradients_and_apply_rotor(params, x, y, theta)

print(f"Loss: {loss_value}")
print(f"Stabilized Gradients: {stabilized_grads}")