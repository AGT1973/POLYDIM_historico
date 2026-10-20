import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
from scipy.linalg import svd
import numpy as np

# Definición de la red neuronal simple
class SimpleNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Función de truncamiento SVD
def truncate_svd(W, k):
    U, S, Vt = svd(W.detach().numpy())
    S_truncated = np.diag(S[:k])
    W_truncated = U[:, :k] @ S_truncated @ Vt[:k, :]
    return torch.tensor(W_truncated, dtype=torch.float32)

# Función de entrenamiento con truncamiento SVD periódico
@jit
def train_step(model, optimizer, x, y, k, step):
    def loss_fn(params):
        preds = model(x)
        loss = torch.nn.functional.mse_loss(preds, y)
        return loss
    
    grads = grad(loss_fn)(model.parameters())
    optimizer.step()
    
    if step % 10 == 0:  # Truncamiento periódico
        with torch.no_grad():
            for param in model.parameters():
                param.data = truncate_svd(param.data, k)
    
    return loss_fn(model.parameters())

# Parámetros del modelo y entrenamiento
input_dim = 10
hidden_dim = 50
output_dim = 1
k = 5  # Rango máximo de los tensores de peso
num_epochs = 100
batch_size = 32

# Crear datos de ejemplo
x = torch.randn(batch_size, input_dim)
y = torch.randn(batch_size, output_dim)

# Inicializar el modelo y el optimizador
model = SimpleNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Entrenamiento
for epoch in range(num_epochs):
    loss = train_step(model, optimizer, x, y, k, epoch)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Verificar el rango de los tensores de peso
for name, param in model.named_parameters():
    U, S, Vt = svd(param.detach().numpy())
    rank = np.sum(S > 1e-6)
    print(f"Tensor {name} rank: {rank}")