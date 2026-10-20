import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import scipy

# Definición de la red neuronal
class TopologicalNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.params = {
            'W1': jnp.zeros((input_dim, hidden_dim)),
            'b1': jnp.zeros(hidden_dim),
            'W2': jnp.zeros((hidden_dim, output_dim)),
            'b2': jnp.zeros(output_dim)
        }

    def forward(self, x):
        hidden = jnp.tanh(jnp.dot(x, self.params['W1']) + self.params['b1'])
        output = jnp.dot(hidden, self.params['W2']) + self.params['b2']
        return output

    def loss(self, params, x, y):
        y_pred = self.forward(x)
        return jnp.mean((y - y_pred) ** 2)

    def apply_lie_symmetry(self, params, group_action):
        # Aplicar una transformación de Lie a los pesos
        W1_transformed = jnp.dot(group_action, params['W1'])
        W2_transformed = jnp.dot(params['W2'], group_action.T)
        return {
            'W1': W1_transformed,
            'b1': params['b1'],
            'W2': W2_transformed,
            'b2': params['b2']
        }

# Generar datos de ejemplo
def generate_data(n_samples, input_dim, output_dim):
    X = jnp.random.randn(n_samples, input_dim)
    y = jnp.random.randn(n_samples, output_dim)
    return X, y

# Entrenar la red neuronal
def train_network(network, X, y, num_epochs, learning_rate):
    optimizer = jax.experimental.optimizers.adam(learning_rate)
    opt_init, opt_update, get_params = optimizer
    opt_state = opt_init(network.params)

    @jit
    def step(i, opt_state, X, y):
        params = get_params(opt_state)
        loss, grads = jax.value_and_grad(network.loss)(params, X, y)
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, loss

    for epoch in range(num_epochs):
        opt_state, loss = step(epoch, opt_state, X, y)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

    return get_params(opt_state)

# Validar la simetría de Lie
def validate_lie_symmetry(network, params, X, y, group_action):
    params_transformed = network.apply_lie_symmetry(params, group_action)
    loss_original = network.loss(params, X, y)
    loss_transformed = network.loss(params_transformed, X, y)
    print(f'Loss original: {loss_original:.4f}, Loss transformado: {loss_transformed:.4f}')

# Ejecución
input_dim = 10
hidden_dim = 50
output_dim = 1
n_samples = 1000
num_epochs = 1000
learning_rate = 0.01

# Crear la red neuronal
network = TopologicalNeuralNetwork(input_dim, hidden_dim, output_dim)

# Generar datos de ejemplo
X, y = generate_data(n_samples, input_dim, output_dim)

# Entrenar la red neuronal
params = train_network(network, X, y, num_epochs, learning_rate)

# Definir una acción de grupo de Lie (rotación)
group_action = jnp.array(scipy.linalg.expm(jnp.random.randn(input_dim, input_dim)))

# Validar la simetría de Lie
validate_lie_symmetry(network, params, X, y, group_action)