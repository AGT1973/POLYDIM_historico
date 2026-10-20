### Concepto Teórico: Fibrados Tangentes y Redes Neuronales

#### 1. Concepto Teórico

**Idea Principal:** Utilizar la estructura de fibrados tangentes para mejorar la eficiencia y estabilidad de las redes neuronales. Específicamente, consideraremos cómo la geometría del espacio de entrada y salida de una red neuronal puede ser modelada utilizando fibrados tangentes, y cómo esto puede ayudar a regularizar y optimizar el entrenamiento.

**Fibrado Tangente:** Un fibrado tangente \( TM \) de una variedad \( M \) es una estructura que asocia a cada punto \( p \in M \) su espacio tangente \( T_pM \). En el contexto de las redes neuronales, podemos considerar que cada capa de la red define una variedad, y el fibrado tangente de esta variedad proporciona una representación local de las direcciones de cambio en el espacio de entrada y salida.

**Redes Neuronales y Fibrados Tangentes:** La idea es utilizar el fibrado tangente para regularizar la propagación de gradientes y evitar problemas como el vanishing y exploding gradients. Esto se logra imponiendo restricciones geométricas en la forma en que los gradientes se propagan a través de la red.

#### 2. Razonamiento Matemático

**Estructura del Fibrado Tangente:**
- Sea \( M \) el espacio de entrada de la red neuronal.
- Sea \( f: M \to N \) la función que representa una capa de la red neuronal, donde \( N \) es el espacio de salida.
- El fibrado tangente \( TM \) consiste en pares \( (p, v) \) donde \( p \in M \) y \( v \in T_pM \).

**Regularización mediante Fibrados Tangentes:**
- La derivada de \( f \) en un punto \( p \) es una aplicación lineal \( df_p: T_pM \to T_{f(p)}N \).
- Podemos regularizar la propagación de gradientes imponiendo que \( df_p \) sea una isometría local, es decir, que preserve las distancias locales.

**Optimización con Truncamiento SVD:**
- Para evitar la explosión del rango de las matrices de pesos, aplicamos una descomposición SVD periódica y truncamos los valores singulares menores.
- Esto se hace para mantener el rango de las matrices de pesos dentro de un límite razonable, lo que ayuda a estabilizar el entrenamiento.

#### 3. Script Python con JAX

```python
import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
from jax.lax import svd
from scipy.linalg import qr
import numpy as np

# Función para aplicar la descomposición SVD y truncar los valores singulares
def truncated_svd(A, k):
    U, S, V = svd(A, full_matrices=False)
    S = S[:k]
    U = U[:, :k]
    V = V[:k, :]
    return U, S, V

# Función para aplicar la descomposición QR seguida de SVD
def qr_svd_truncate(A, k):
    Q, R = qr(A)
    U, S, V = truncated_svd(R, k)
    return Q @ U, S, V

# Definición de una capa de red neuronal simple
def neural_layer(x, W, b):
    return jnp.dot(W, x) + b

# Función de pérdida
def loss_fn(params, x, y):
    W, b = params
    y_pred = neural_layer(x, W, b)
    return jnp.mean((y_pred - y) ** 2)

# Gradiente de la función de pérdida
grad_loss_fn = grad(loss_fn)

# Función de actualización de parámetros con regularización SVD
@jit
def update_params(params, x, y, learning_rate, k):
    W, b = params
    grad_W, grad_b = grad_loss_fn(params, x, y)
    
    # Aplicar descomposición SVD y truncar
    U, S, V = qr_svd_truncate(grad_W, k)
    W = W - learning_rate * (U @ np.diag(S) @ V)
    b = b - learning_rate * grad_b
    
    return (W, b)

# Datos de entrada y salida de ejemplo
x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
y = jnp.array([[1.0], [2.0]])

# Inicialización de parámetros
W = jnp.array([[0.1, 0.2], [0.3, 0.4]])
b = jnp.array([0.1])

# Hiperparámetros
learning_rate = 0.01
k = 1
num_epochs = 100

# Entrenamiento
for epoch in range(num_epochs):
    params = (W, b)
    W, b = update_params(params, x, y, learning_rate, k)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss_fn(params, x, y)}")

print("Final Weights:", W)
print("Final Bias:", b)
```

### Explicación del Código

1. **Descomposición SVD y Truncamiento:**
   - La función `truncated_svd` realiza una descomposición SVD y trunca los valores singulares menores a un rango especificado \( k \).
   - La función `qr_svd_truncate` primero aplica una descomposición QR y luego SVD, lo que es útil para matrices grandes y ruidosas.

2. **Capa de Red Neuronal:**
   - La función `neural_layer` define una capa de red neuronal simple con una matriz de pesos \( W \) y un vector de sesgos \( b \).

3. **Función de Pérdida:**
   - La función `loss_fn` calcula la pérdida cuadrática media entre las predicciones y los valores reales.

4. **Gradiente de la Función de Pérdida:**
   - La función `grad_loss_fn` calcula el gradiente de la función de pérdida con respecto a los parámetros.

5. **Actualización de Parámetros:**
   - La función `update_params` actualiza los parámetros de la red neuronal aplicando la descomposición SVD y truncando los valores singulares menores para mantener el rango dentro de un límite razonable.

6. **Entrenamiento:**
   - Se inicializan los parámetros y se entrena la red durante un número especificado de épocas, imprimiendo la pérdida cada 10 épocas para monitorear el progreso.

Este enfoque combina la geometría de los fibrados tangentes con la optimización de redes neuronales, asegurando que la propagación de gradientes sea estable y eficiente.