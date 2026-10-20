import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import scipy.linalg

# Función para truncar una matriz a rango k usando SVD
def truncate_svd(A, k):
    U, S, Vt = jnp.linalg.svd(A, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    A_k = U_k @ jnp.diag(S_k) @ Vt_k
    return A_k

# Función para aplicar truncamiento SVD a un tensor de pesos
def truncate_weights(weights, k):
    shape = weights.shape
    weights_2d = weights.reshape(shape[0], -1)
    truncated_weights_2d = truncate_svd(weights_2d, k)
    truncated_weights = truncated_weights_2d.reshape(shape)
    return truncated_weights

# Ejemplo de red neuronal con capa lineal y función de activación ReLU
class SimpleNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rank):
        super(SimpleNN, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        self.rank = rank

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def truncate_weights(self):
        self.linear1.weight.data = torch.tensor(truncate_weights(self.linear1.weight.data.numpy(), self.rank))
        self.linear2.weight.data = torch.tensor(truncate_weights(self.linear2.weight.data.numpy(), self.rank))

# Función para entrenar la red neuronal
def train_model(model, x, y, num_epochs, learning_rate, rank):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # Truncar los pesos periódicamente
        if epoch % 10 == 0:
            model.truncate_weights()
        
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

# Datos de ejemplo
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# Parámetros del modelo
input_dim = 10
hidden_dim = 50
output_dim = 1
rank = 10
learning_rate = 0.01
num_epochs = 1000

# Crear y entrenar el modelo
model = SimpleNN(input_dim, hidden_dim, output_dim, rank)
train_model(model, x, y, num_epochs, learning_rate, rank)