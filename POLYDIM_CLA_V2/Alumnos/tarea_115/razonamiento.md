### Concepto Teórico: Topología de la Activación en Redes Neuronales

#### 1. Concepto Teórico

**Definición:** Consideremos una red neuronal feedforward con \( L \) capas, donde cada capa \( l \) tiene una función de activación \( \sigma_l \). La salida de la capa \( l \) se denota como \( h_l \), y la entrada a la capa \( l \) como \( z_l \). La topología de la activación \( T \) es definida como la estructura topológica del espacio de activaciones \( \{ h_l \}_{l=1}^L \) en el espacio de entrada \( \mathcal{X} \).

**Hipótesis:** La topología de la activación \( T \) es invariante a ciertas transformaciones isométricas del espacio de entrada \( \mathcal{X} \). Específicamente, si \( \phi: \mathcal{X} \to \mathcal{X} \) es una isometría (es decir, preserva las distancias), entonces la topología de la activación \( T \) de la red neuronal es la misma para \( \mathcal{X} \) y \( \phi(\mathcal{X}) \).

**Implicaciones:** Si la hipótesis es cierta, esto significa que la estructura topológica de las activaciones de la red neuronal no cambia bajo transformaciones que preservan la geometría del espacio de entrada. Esto tiene implicaciones importantes en la robustez de las redes neuronales a ciertas perturbaciones y en la interpretación de las características aprendidas por la red.

#### 2. Razonamiento Matemático

**Definición de Isometría:** Una función \( \phi: \mathcal{X} \to \mathcal{X} \) es una isometría si para todo \( x, y \in \mathcal{X} \), se cumple que \( d(\phi(x), \phi(y)) = d(x, y) \), donde \( d \) es una métrica en \( \mathcal{X} \).

**Teorema:** Si \( \phi: \mathcal{X} \to \mathcal{X} \) es una isometría, entonces la topología de la activación \( T \) de la red neuronal es invariante bajo \( \phi \).

**Demostración:**
1. **Preservación de Distancias:** Dado que \( \phi \) es una isometría, las distancias entre puntos en \( \mathcal{X} \) se preservan. Esto implica que la estructura métrica del espacio de entrada no cambia.
2. **Preservación de Activaciones:** Las funciones de activación \( \sigma_l \) son generalmente continuas y no lineales. Sin embargo, la continuidad asegura que pequeñas perturbaciones en la entrada resultan en pequeñas perturbaciones en la salida. Dado que \( \phi \) preserva las distancias, las perturbaciones en \( \mathcal{X} \) se mapean de manera consistente a perturbaciones en \( \{ h_l \}_{l=1}^L \).
3. **Invariancia Topológica:** La invariancia topológica se sigue de la preservación de la estructura métrica y la continuidad de las funciones de activación. La topología de la activación \( T \) es una propiedad global que depende de la estructura métrica y la continuidad, por lo que debe ser invariante bajo isometrías.

#### 3. Script Python para Validar Empíricamente la Teoría

```python
import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import scipy.spatial

# Definir una red neuronal simple usando JAX
def neural_network(params, x):
    for W, b in params:
        x = jnp.dot(x, W) + b
        x = jax.nn.relu(x)
    return x

# Generar datos de entrada
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (100, 10))

# Definir una isometría simple (rotación)
def isometry(x):
    # Generar una matriz de rotación aleatoria
    rotation_matrix = scipy.stats.ortho_group.rvs(10)
    return jnp.dot(x, rotation_matrix)

# Inicializar parámetros de la red neuronal
params = [(jax.random.normal(key, (10, 10)), jax.random.normal(key, (10,))),
          (jax.random.normal(key, (10, 5)), jax.random.normal(key, (5,)))]

# Calcular las activaciones para los datos originales y transformados
activations_original = vmap(neural_network, in_axes=(None, 0))(params, x)
activations_transformed = vmap(neural_network, in_axes=(None, 0))(params, isometry(x))

# Calcular la distancia euclidiana entre las activaciones
distances_original = scipy.spatial.distance.pdist(activations_original, 'euclidean')
distances_transformed = scipy.spatial.distance.pdist(activations_transformed, 'euclidean')

# Comparar las distancias
distance_difference = jnp.linalg.norm(distances_original - distances_transformed)

print(f"Distancia entre las distancias de activaciones: {distance_difference:.6f}")
```

### Explicación del Script

1. **Definición de la Red Neuronal:** Se define una red neuronal simple usando JAX, con dos capas ocultas y funciones de activación ReLU.
2. **Generación de Datos de Entrada:** Se generan datos de entrada aleatorios \( x \) de dimensión 100x10.
3. **Definición de Isometría:** Se define una isometría simple como una rotación aleatoria en el espacio de entrada.
4. **Cálculo de Activaciones:** Se calculan las activaciones de la red neuronal para los datos originales y los datos transformados por la isometría.
5. **Comparación de Distancias:** Se calculan las distancias euclidianas entre las activaciones de los datos originales y transformados, y se mide la diferencia entre estas distancias.

Si la hipótesis es cierta, la diferencia entre las distancias de activaciones debe ser pequeña, indicando que la topología de la activación es invariante a la isometría.