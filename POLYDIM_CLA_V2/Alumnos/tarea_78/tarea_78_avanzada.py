import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import torch
from torch import einsum
from scipy.optimize import minimize

# Definir una función de pérdida simple
def loss_function(params, inputs, targets):
    predictions = model(params, inputs)
    return jnp.mean((predictions - targets) ** 2)

# Definir el modelo de red neuronal
def model(params, inputs):
    W1, b1, W2, b2 = params
    hidden = jnp.tanh(jnp.dot(inputs, W1) + b1)
    return jnp.dot(hidden, W2) + b2

# Inicializar parámetros del modelo
def initialize_params(input_dim, hidden_dim, output_dim):
    W1 = jax.random.normal(jax.random.PRNGKey(0), (input_dim, hidden_dim))
    b1 = jax.random.normal(jax.random.PRNGKey(1), (hidden_dim,))
    W2 = jax.random.normal(jax.random.PRNGKey(2), (hidden_dim, output_dim))
    b2 = jax.random.normal(jax.random.PRNGKey(3), (output_dim,))
    return (W1, b1, W2, b2)

# Gradiente de la función de pérdida
grad_loss = jit(grad(loss_function))

# Función para estabilizar los gradientes
def stabilize_gradients(gradients):
    norms = jnp.linalg.norm(gradients, axis=1)
    normalized_gradients = gradients / norms[:, None]
    return normalized_gradients

# Generar datos de ejemplo
input_dim = 10
hidden_dim = 5
output_dim = 1
num_samples = 100

inputs = jax.random.normal(jax.random.PRNGKey(4), (num_samples, input_dim))
targets = jax.random.normal(jax.random.PRNGKey(5), (num_samples, output_dim))

# Inicializar parámetros
params = initialize_params(input_dim, hidden_dim, output_dim)

# Entrenar el modelo
for i in range(1000):
    gradients = grad_loss(params, inputs, targets)
    stabilized_gradients = stabilize_gradients(gradients)
    params = [p - 0.01 * g for p, g in zip(params, stabilized_gradients)]

    if i % 100 == 0:
        loss = loss_function(params, inputs, targets)
        print(f"Iteration {i}, Loss: {loss:.4f}")

# Validación final
final_loss = loss_function(params, inputs, targets)
print(f"Final Loss: {final_loss:.4f}")

# Validación adicional con PyTorch
def torch_model(params, inputs):
    W1, b1, W2, b2 = params
    hidden = torch.tanh(torch.einsum('ij,jk->ik', inputs, W1) + b1)
    return torch.einsum('ij,jk->ik', hidden, W2) + b2

torch_params = [torch.tensor(p, requires_grad=True) for p in params]
torch_inputs = torch.tensor(inputs, dtype=torch.float32)
torch_targets = torch.tensor(targets, dtype=torch.float32)

torch_predictions = torch_model(torch_params, torch_inputs)
torch_loss = torch.mean((torch_predictions - torch_targets) ** 2)
torch_loss.backward()

print(f"PyTorch Loss: {torch_loss.item():.4f}")
print(f"PyTorch Gradients: {torch_params[0].grad}")