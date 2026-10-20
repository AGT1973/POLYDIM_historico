import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import scipy.optimize

# Definición de la función rotor de Clifford
def rotor_clifford(theta, e1, e2):
    """
    Crea un rotor de Clifford para una rotación de ángulo theta en el plano definido por e1 y e2.
    """
    return jnp.cos(theta / 2) - jnp.sin(theta / 2) * (e1 * e2)

# Definición de la función de rotación usando el rotor
def rotate_vector(v, rotor):
    """
    Rota un vector v usando un rotor de Clifford.
    """
    return rotor * v * rotor.inverse()

# Definición de la red neuronal con rotación en capas ocultas
def neural_network(x, params, rotor):
    """
    Red neuronal simple con una capa oculta y una capa de salida.
    Se aplica una rotación en la capa oculta.
    """
    w1, b1, w2, b2 = params
    h1 = jnp.dot(w1, x) + b1
    h1_rotated = rotate_vector(h1, rotor)
    h1_activated = jax.nn.relu(h1_rotated)
    out = jnp.dot(w2, h1_activated) + b2
    return out

# Función de pérdida
def loss_fn(params, rotor, x, y):
    """
    Función de pérdida para la red neuronal.
    """
    y_pred = neural_network(x, params, rotor)
    return jnp.mean((y_pred - y) ** 2)

# Inicialización de parámetros
def init_params(key, input_dim, hidden_dim, output_dim):
    """
    Inicializa los parámetros de la red neuronal.
    """
    w1 = jax.random.normal(key, (hidden_dim, input_dim))
    b1 = jax.random.normal(key, (hidden_dim,))
    w2 = jax.random.normal(key, (output_dim, hidden_dim))
    b2 = jax.random.normal(key, (output_dim,))
    return (w1, b1, w2, b2)

# Optimización de la red neuronal
def optimize_network(x, y, params, rotor, num_steps=1000, learning_rate=0.01):
    """
    Optimiza los parámetros de la red neuronal usando descenso de gradiente.
    """
    opt_init, opt_update, get_params = jax.experimental.optimizers.adam(learning_rate)
    opt_state = opt_init(params)

    @jit
    def step(opt_state, x, y, rotor):
        params = get_params(opt_state)
        loss, grads = jax.value_and_grad(loss_fn)(params, rotor, x, y)
        opt_state = opt_update(0, grads, opt_state)
        return opt_state, loss

    for i in range(num_steps):
        opt_state, loss = step(opt_state, x, y, rotor)
        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")

    return get_params(opt_state)

# Datos de ejemplo
x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
y = jnp.array([[0.0], [1.0]])

# Parámetros iniciales
key = jax.random.PRNGKey(0)
params = init_params(key, input_dim=2, hidden_dim=10, output_dim=1)

# Rotor de Clifford para una rotación de 45 grados en el plano definido por e1 y e2
theta = jnp.pi / 4
e1 = jnp.array([1, 0])
e2 = jnp.array([0, 1])
rotor = rotor_clifford(theta, e1, e2)

# Optimización de la red neuronal
optimized_params = optimize_network(x, y, params, rotor)

# Evaluación de la red neuronal optimizada
y_pred = neural_network(x, optimized_params, rotor)
print("Predicciones finales:", y_pred)