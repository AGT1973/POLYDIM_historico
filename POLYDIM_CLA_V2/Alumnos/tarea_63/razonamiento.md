### Concepto Teórico: Topología y Redes Neuronales

#### 1. Concepto Teórico: Invariancia Topológica en Redes Neuronales

**Motivación:**
En el contexto de las redes neuronales, la invariancia a transformaciones topológicas es crucial para garantizar que el modelo sea robusto frente a deformaciones y perturbaciones en los datos de entrada. Un ejemplo clásico es la invariancia a rotaciones y traslaciones en imágenes, que es fundamental en tareas de visión por computador.

**Idea Principal:**
Podemos definir una red neuronal que sea invariante a ciertas transformaciones topológicas utilizando grupos de Lie. Específicamente, podemos construir una red neuronal que sea invariante a las acciones de un grupo de Lie \( G \) en el espacio de entrada. Esto se logra mediante la incorporación de capas que computen funciones invariantes bajo la acción de \( G \).

**Formalización Matemática:**
Consideremos un grupo de Lie \( G \) que actúa sobre el espacio de entrada \( X \). Una función \( f: X \to \mathbb{R} \) es invariante bajo \( G \) si para todo \( g \in G \) y \( x \in X \):

\[ f(g \cdot x) = f(x) \]

Podemos diseñar una red neuronal \( \mathcal{N} \) que sea invariante a \( G \) al asegurarnos de que cada capa \( \mathcal{N}_i \) satisfaga:

\[ \mathcal{N}_i(g \cdot x) = \mathcal{N}_i(x) \]

Para lograr esto, podemos utilizar capas de convolución invariante a \( G \), como las capas de convolución steerable.

#### 2. Razonamiento Matemático

**Grupos de Lie y Acciones:**
Un grupo de Lie \( G \) es un grupo que también es una variedad diferenciable. La acción de \( G \) sobre \( X \) es una aplicación \( \cdot: G \times X \to X \) que satisface:

1. \( e \cdot x = x \) para todo \( x \in X \), donde \( e \) es el elemento identidad de \( G \).
2. \( (gh) \cdot x = g \cdot (h \cdot x) \) para todo \( g, h \in G \) y \( x \in X \).

**Funciones Invariantes:**
Una función \( f: X \to \mathbb{R} \) es invariante bajo \( G \) si:

\[ f(g \cdot x) = f(x) \]

**Capas de Convolución Steerable:**
Las caplas de convolución steerable son capas de convolución que son invariantes a ciertas transformaciones. Por ejemplo, una capa de convolución steerable a rotaciones es invariante a rotaciones del input.

#### 3. Script Python para Validación Empírica

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
import scipy.optimize
import torch

# Definición del grupo de Lie SO(2) (rotaciones en 2D)
def so2_action(theta, x):
    """Aplica una rotación en 2D a un vector x."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    R = jnp.array([[c, -s], [s, c]])
    return R @ x

# Definición de una función invariante a SO(2)
def invariant_function(x):
    """Función invariante a rotaciones en 2D."""
    return jnp.linalg.norm(x)  # Norma Euclidiana

# Verificación de la invariancia
def check_invariance(theta, x):
    rotated_x = so2_action(theta, x)
    return invariant_function(rotated_x) == invariant_function(x)

# Función de red neuronal simple invariante a SO(2)
def invariant_neural_network(x, params):
    """Red neuronal invariante a rotaciones en 2D."""
    w1, b1, w2, b2 = params
    h1 = jnp.linalg.norm(x)  # Capa invariante a rotaciones
    h2 = jnp.tanh(jnp.dot(w1, h1) + b1)
    return jnp.dot(w2, h2) + b2

# Función de pérdida
def loss(params, x, y):
    """Función de pérdida para la red neuronal."""
    y_pred = invariant_neural_network(x, params)
    return jnp.mean((y_pred - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = jit(grad(loss))

# Optimización de la red neuronal
def optimize_neural_network(x, y, params, num_steps=1000, learning_rate=0.01):
    for i in range(num_steps):
        grads = grad_loss(params, x, y)
        params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)
        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss(params, x, y):.4f}")
    return params

# Datos de ejemplo
np.random.seed(42)
x = np.random.randn(100, 2)
y = np.linalg.norm(x, axis=1)  # Etiquetas son la norma Euclidiana

# Inicialización de parámetros
w1 = np.random.randn(1)
b1 = np.random.randn(1)
w2 = np.random.randn(1)
b2 = np.random.randn(1)
params = (w1, b1, w2, b2)

# Optimización
optimized_params = optimize_neural_network(x, y, params)

# Verificación de la invariancia
theta = np.pi / 4  # Rotación de 45 grados
x_rotated = vmap(so2_action, in_axes=(None, 0))(theta, x)
y_pred = invariant_neural_network(x, optimized_params)
y_pred_rotated = invariant_neural_network(x_rotated, optimized_params)

# Cálculo del error
error = jnp.mean(jnp.abs(y_pred - y_pred_rotated))
print(f"Error de invariancia: {error:.4f}")
```

### Explicación del Script

1. **Definición del Grupo de Lie SO(2):** Se define la acción de rotación en 2D.
2. **Función Invariante:** Se define una función que es invariante a rotaciones, en este caso, la norma Euclidiana.
3. **Verificación de la Invariancia:** Se verifica que la función invariante sea efectivamente invariante a rotaciones.
4. **Red Neuronal Invariante:** Se define una red neuronal simple que es invariante a rotaciones en 2D.
5. **Función de Pérdida:** Se define la función de pérdida para la red neuronal.
6. **Optimización:** Se implementa la optimización de la red neuronal utilizando el gradiente descendente.
7. **Verificación de la Invariancia Empírica:** Se verifica que la red neuronal sea invariante a rotaciones aplicando una rotación a los datos de entrada y comparando las predicciones antes y después de la rotación.

Este script demuestra cómo se puede construir y optimizar una red neuronal que sea invariante a ciertas transformaciones topológicas, en este caso, rotaciones en 2D.