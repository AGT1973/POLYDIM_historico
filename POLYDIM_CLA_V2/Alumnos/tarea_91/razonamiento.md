### Concepto Teórico: Topological Invariance in Neural Network Layers

#### 1. Concepto Teórico

**Objetivo:** Diseñar una capa de red neuronal que sea invariante a transformaciones topológicas, específicamente a isometrías. Esto significa que la salida de la capa no debe cambiar si la entrada se transforma mediante una isometría (rotación, traslación, reflexión).

**Razonamiento Matemático:**

1. **Isometrías y Transformaciones Lineales:**
   - Una isometría es una transformación que preserva las distancias. En un espacio euclidiano \( \mathbb{R}^n \), una isometría \( T \) puede ser representada como:
     \[
     T(\mathbf{x}) = R\mathbf{x} + \mathbf{t}
     \]
     donde \( R \) es una matriz de rotación (ortogonal) y \( \mathbf{t} \) es un vector de traslación.

2. **Invariancia a Isometrías:**
   - Para que una capa de red neuronal sea invariante a isometrías, la salida \( f(T(\mathbf{x})) \) debe ser igual a \( f(\mathbf{x}) \) para cualquier isometría \( T \).

3. **Capa de Red Neuronal:**
   - Consideremos una capa lineal seguida de una función de activación no lineal:
     \[
     y = \sigma(W\mathbf{x} + \mathbf{b})
     \]
     donde \( W \) es la matriz de pesos, \( \mathbf{b} \) es el vector de sesgos, y \( \sigma \) es la función de activación.

4. **Invariancia a Rotaciones:**
   - Para que la capa sea invariante a rotaciones, \( W \) debe ser una matriz que commute con cualquier matriz de rotación \( R \). Esto implica que \( W \) debe ser una matriz diagonal (considerando el teorema de Schur).

5. **Invariancia a Traslaciones:**
   - Para la invariancia a traslaciones, la capa debe ser capaz de eliminar el efecto de la traslación. Esto puede lograrse mediante el uso de una capa de pooling global, que toma el máximo o la media de las características a lo largo de todas las posiciones.

#### 2. Razonamiento Matemático (Markdown)

La invariancia a isometrías en una capa de red neuronal puede ser lograda mediante el diseño de una capa que combine una matriz de pesos diagonal y una capa de pooling global. La matriz diagonal \( W \) asegura la invariancia a rotaciones, mientras que el pooling global asegura la invariancia a traslaciones.

La capa puede ser definida como:
\[
y = \sigma(\text{pool}(W\mathbf{x} + \mathbf{b}))
\]
donde:
- \( W \) es una matriz diagonal.
- \( \text{pool} \) es una operación de pooling global (por ejemplo, max-pooling).
- \( \sigma \) es una función de activación no lineal.

#### 3. Script Python Autocontenidos

```python
import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
from scipy.spatial.transform import Rotation as R

# Definición de la capa invariante a isometrías
def invariant_layer(x, W, b, pool_type='max'):
    # W debe ser una matriz diagonal
    W_diagonal = jnp.diag(W)
    # Aplicar la transformación lineal
    z = jnp.dot(W_diagonal, x) + b
    # Aplicar la función de activación (ReLU)
    z_activated = jnp.maximum(z, 0)
    # Aplicar el pooling global
    if pool_type == 'max':
        y = jnp.max(z_activated, axis=0)
    elif pool_type == 'mean':
        y = jnp.mean(z_activated, axis=0)
    else:
        raise ValueError("Invalid pooling type. Use 'max' or 'mean'.")
    return y

# Función para generar una isometría aleatoria
def random_isometry(dim):
    # Generar una matriz de rotación aleatoria
    rotation = R.random().as_matrix()[:dim, :dim]
    # Generar un vector de traslación aleatorio
    translation = jnp.array(torch.randn(dim))
    return rotation, translation

# Función para aplicar una isometría a un vector
def apply_isometry(x, rotation, translation):
    return jnp.dot(rotation, x) + translation

# Función para verificar la invariancia
def check_invariance(x, W, b, pool_type='max'):
    # Generar una isometría aleatoria
    rotation, translation = random_isometry(x.shape[0])
    # Aplicar la isometría a x
    x_transformed = apply_isometry(x, rotation, translation)
    # Calcular la salida antes y después de la isometría
    y_original = invariant_layer(x, W, b, pool_type)
    y_transformed = invariant_layer(x_transformed, W, b, pool_type)
    # Verificar si las salidas son iguales
    return jnp.allclose(y_original, y_transformed, atol=1e-6)

# Parámetros de la capa
dim = 3
W = jnp.array([1.0, 2.0, 3.0])  # Diagonal de la matriz de pesos
b = jnp.array([0.5, -0.5, 0.0])  # Vector de sesgos
x = jnp.array([1.0, 2.0, 3.0])  # Entrada de prueba

# Verificar la invariancia
invariant = check_invariance(x, W, b, pool_type='max')
print(f"La capa es invariante a isometrías: {invariant}")
```

Este script define una capa de red neuronal invariante a isometrías y verifica empíricamente su invariancia aplicando una isometría aleatoria a la entrada y comparando las salidas antes y después de la transformación.