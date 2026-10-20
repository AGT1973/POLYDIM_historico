import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import scipy.linalg

# Parámetros de la red
input_dim = 100
output_dim = 100
rank_limit = 10
num_layers = 5

# Función para crear un rotor de Clifford
def create_rotor(theta, bivector):
    return jnp.exp(0.5 * theta * bivector)

# Función para aplicar un rotor a un vector
def apply_rotor(rotor, x):
    return rotor @ x @ rotor.T

# Función para truncar una matriz a un rango específico usando SVD
def truncate_svd(matrix, rank):
    U, S, Vt = jnp.linalg.svd(matrix)
    U_trunc = U[:, :rank]
    S_trunc = S[:rank]
    Vt_trunc = Vt[:rank, :]
    return U_trunc @ jnp.diag(S_trunc) @ Vt_trunc

# Inicialización de capas de rotores
rotors = [create_rotor(jnp.random.uniform(0, jnp.pi), jnp.random.randn(input_dim, input_dim)) for _ in range(num_layers)]

# Función de forward pass de la red
@jit
def forward_pass(x, rotors):
    for rotor in rotors:
        x = apply_rotor(rotor, x)
        x = truncate_svd(x, rank_limit)
    return x

# Función de pérdida (ejemplo: MSE)
def loss_fn(params, x, y):
    y_pred = forward_pass(x, params)
    return jnp.mean((y_pred - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = grad(loss_fn)

# Datos de ejemplo
x = jnp.random.randn(input_dim, input_dim)
y = jnp.random.randn(output_dim, output_dim)

# Entrenamiento de la red
learning_rate = 0.01
num_steps = 100

for step in range(num_steps):
    grads = grad_loss(rotors, x, y)
    for i in range(num_layers):
        rotors[i] -= learning_rate * grads[i]

    if step % 10 == 0:
        current_loss = loss_fn(rotors, x, y)
        print(f"Step {step}, Loss: {current_loss:.4f}")

# Validación final
y_pred = forward_pass(x, rotors)
final_loss = loss_fn(rotors, x, y)
print(f"Final Loss: {final_loss:.4f}")