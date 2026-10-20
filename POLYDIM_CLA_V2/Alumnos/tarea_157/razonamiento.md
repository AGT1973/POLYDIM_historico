### Concepto Teórico: Topología Diferencial y Redes Neuronales con Truncamiento SVD

#### 1. Concepto Teórico

**Título: Topología Diferencial y Redes Neuronales con Truncamiento SVD**

**Descripción:**
En este concepto teórico, conectamos la topología diferencial con las redes neuronales mediante el uso de descomposiciones de tensores y técnicas de optimización. La idea principal es utilizar la descomposición de valor singular (SVD) para regularizar y controlar el rango de los tensores de peso en las redes neuronales, lo que ayuda a mantener la invariancia isométrica y estabilizar los gradientes durante el entrenamiento.

**Matemática:**
1. **Topología Diferencial y Redes Neuronales:**
   - Las redes neuronales pueden verse como funciones diferenciables que mapean de un espacio de entrada a un espacio de salida.
   - La invariancia isométrica en estos mapeos es crucial para mantener la estructura topológica de los datos.
   - La descomposición SVD ayuda a regularizar los pesos de la red, manteniendo la invariancia isométrica y evitando la explosión del rango.

2. **Truncamiento SVD:**
   - La descomposición SVD de una matriz \( W \) es \( W = U \Sigma V^T \), donde \( U \) y \( V \) son matrices ortogonales y \( \Sigma \) es una matriz diagonal de valores singulares.
   - El truncamiento SVD consiste en mantener solo los \( k \) valores singulares más grandes, lo que reduce el rango de la matriz a \( k \).
   - Esto ayuda a evitar la explosión del rango y mantiene la estructura topológica de los datos.

#### 2. Razonamiento Matemático

**Teorema de Eckart-Young:**
- El teorema de Eckart-Young establece que la mejor aproximación de rango \( k \) a una matriz \( W \) en términos de norma Frobenius es dada por la descomposición SVD truncada.
- Si \( W = U \Sigma V^T \) es la SVD de \( W \), entonces la mejor aproximación de rango \( k \) es \( W_k = U_k \Sigma_k V_k^T \), donde \( U_k \), \( \Sigma_k \), y \( V_k \) contienen los primeros \( k \) valores singulares y sus correspondientes vectores singulares.

**Aplicación en Redes Neuronales:**
- Durante el entrenamiento, aplicamos el truncamiento SVD periódicamente a las matrices de peso de las capas de la red.
- Esto ayuda a mantener el rango de las matrices de peso bajo control, evitando la explosión del rango y estabilizando los gradientes.
- La invariancia isométrica se mantiene al regularizar los pesos de la red, lo que es crucial para la preservación de la estructura topológica de los datos.

#### 3. Script Python

```python
import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import torch
import scipy.linalg

# Definición de la función de pérdida
def loss_fn(params, x, y):
    y_pred = model(params, x)
    return jnp.mean((y_pred - y) ** 2)

# Definición del modelo simple
def model(params, x):
    return jnp.dot(params, x)

# Función de truncamiento SVD
def truncate_svd(W, k):
    U, S, V = jnp.linalg.svd(W, full_matrices=False)
    S_k = jnp.diag(S[:k])
    U_k = U[:, :k]
    V_k = V[:k, :]
    return jnp.dot(U_k, jnp.dot(S_k, V_k))

# Función de optimización con truncamiento SVD
@jit
def optimize_step(params, x, y, k):
    grads = grad(loss_fn)(params, x, y)
    params -= 0.01 * grads  # Paso de gradiente descendente
    params = truncate_svd(params, k)  # Truncamiento SVD
    return params

# Datos de ejemplo
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

# Inicialización de parámetros
params = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Número de iteraciones y rango truncado
num_iterations = 100
k = 2

# Bucle de optimización
for i in range(num_iterations):
    params = optimize_step(params, x, y, k)
    if i % 10 == 0:
        print(f"Iteración {i}, Pérdida: {loss_fn(params, x, y)}")

# Resultados finales
print("Parámetros finales:")
print(params)
print("Pérdida final:")
print(loss_fn(params, x, y))
```

### Explicación del Código

1. **Definición de la Función de Pérdida:**
   - `loss_fn` calcula la pérdida cuadrática media entre las predicciones del modelo y los valores reales.

2. **Definición del Modelo:**
   - `model` es un modelo simple que realiza una multiplicación de matriz por vector.

3. **Función de Truncamiento SVD:**
   - `truncate_svd` realiza la descomposición SVD de una matriz y retorna la aproximación de rango \( k \).

4. **Función de Optimización con Truncamiento SVD:**
   - `optimize_step` realiza un paso de gradiente descendente y aplica el truncamiento SVD para regularizar los pesos.

5. **Datos de Ejemplo:**
   - `x` y `y` son datos de entrada y salida de ejemplo.

6. **Inicialización de Parámetros:**
   - `params` son los pesos iniciales del modelo.

7. **Bucle de Optimización:**
   - Se realizan `num_iterations` iteraciones de optimización, donde cada 10 iteraciones se imprime la pérdida actual.

8. **Resultados Finales:**
   - Se imprimen los parámetros finales y la pérdida final.

Este script demuestra cómo la descomposición SVD truncada puede ser utilizada para regularizar los pesos de una red neuronal, manteniendo la invariancia isométrica y estabilizando los gradientes durante el entrenamiento.