import jax
import jax.numpy as jnp
import torch
import scipy
from jax import vmap, jit, grad
from torch import nn, optim

# Definir la red neuronal
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Función de pérdida
def loss_fn(model, x, y):
    y_pred = model(x)
    loss = (y_pred - y).pow(2).mean()
    return loss

# Isometría en el espacio de parámetros
def isometry(params, Q):
    return jnp.dot(Q, params)

# Función para calcular la norma del gradiente
def gradient_norm(model, x, y):
    params = jax.tree_util.tree_leaves(model.state_dict())
    params_vector = jnp.concatenate([p.flatten() for p in params])
    loss = loss_fn(model, x, y)
    grad_loss = grad(loss_fn)(model, x, y)
    grad_vector = jnp.concatenate([g.flatten() for g in jax.tree_util.tree_leaves(grad_loss)])
    return jnp.linalg.norm(grad_vector)

# Generar datos de ejemplo
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# Inicializar la red neuronal
model = SimpleNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Entrenar la red neuronal
for epoch in range(100):
    optimizer.zero_grad()
    loss = loss_fn(model, x, y)
    loss.backward()
    optimizer.step()

    # Calcular la norma del gradiente
    grad_norm = gradient_norm(model, x, y)
    print(f"Epoch {epoch}, Loss: {loss.item()}, Grad Norm: {grad_norm}")

    # Aplicar isometría a los parámetros
    params = jax.tree_util.tree_leaves(model.state_dict())
    params_vector = jnp.concatenate([p.flatten() for p in params])
    Q = scipy.linalg.orth(jnp.eye(params_vector.shape[0]))
    isometric_params = isometry(params_vector, Q)

    # Asignar los parámetros isométricos a la red
    idx = 0
    for param in model.parameters():
        param.data = torch.tensor(isometric_params[idx:idx + param.numel()].reshape(param.shape))
        idx += param.numel()

    # Calcular la norma del gradiente después de la isometría
    isometric_grad_norm = gradient_norm(model, x, y)
    print(f"Epoch {epoch}, Isometric Grad Norm: {isometric_grad_norm}")

    # Verificar la invariancia isométrica
    assert jnp.allclose(grad_norm, isometric_grad_norm, atol=1e-5), "Gradient norms are not invariant under isometry"