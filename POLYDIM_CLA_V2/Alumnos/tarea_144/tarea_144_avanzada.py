import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap, jit, grad
from scipy.linalg import svd

# Parámetros iniciales
key = random.PRNGKey(0)
input_dim = 10
hidden_dim = 5
output_dim = 1
batch_size = 32

# Inicialización de pesos
W1 = random.normal(key, (hidden_dim, input_dim))
W2 = random.normal(key, (output_dim, hidden_dim))

# Función de activación
def activation(x):
    return jnp.tanh(x)

# Función de pérdida
def loss_fn(params, x, y):
    W1, W2 = params
    z1 = activation(x @ W1.T)
    z2 = z1 @ W2.T
    return jnp.mean((z2 - y) ** 2)

# Gradiente de la función de pérdida
grad_loss_fn = grad(loss_fn)

# Función de truncamiento SVD
def truncate_svd(W, rank):
    U, S, Vh = svd(W, full_matrices=False)
    S_truncated = jnp.diag(S[:rank])
    U_truncated = U[:, :rank]
    Vh_truncated = Vh[:rank, :]
    return U_truncated @ S_truncated @ Vh_truncated

# Entrenamiento con truncamiento SVD periódico
def train(params, x, y, num_steps=1000, rank=2):
    for step in range(num_steps):
        grads = grad_loss_fn(params, x, y)
        W1_grad, W2_grad = grads
        W1 -= 0.01 * W1_grad
        W2 -= 0.01 * W2_grad
        
        # Truncamiento SVD periódico
        if step % 100 == 0:
            W1 = truncate_svd(W1, rank)
            W2 = truncate_svd(W2, rank)
        
        if step % 100 == 0:
            loss = loss_fn(params, x, y)
            print(f"Step {step}, Loss: {loss:.4f}")
    
    return params

# Datos de ejemplo
x = random.normal(key, (batch_size, input_dim))
y = random.normal(key, (batch_size, output_dim))

# Entrenamiento
params = (W1, W2)
params = train(params, x, y)

# Evaluación final
final_loss = loss_fn(params, x, y)
print(f"Final Loss: {final_loss:.4f}")