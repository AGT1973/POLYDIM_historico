### Concepto Teórico: Fibrados Tangentes y Redes Neuronales

**Título: Fibrados Tangentes y Optimización de Redes Neuronales con Teorema de Eckart-Young**

**Concepto:**
Consideremos una red neuronal de múltiples capas, donde cada capa puede ser vista como una transformación diferenciable \( f: \mathbb{R}^n \to \mathbb{R}^m \). El fibrado tangente \( T\mathbb{R}^n \) en cada punto \( x \in \mathbb{R}^n \) es un espacio vectorial que contiene todas las direcciones posibles de cambio en \( f \) en ese punto. La matriz Jacobiana \( J_f(x) \) de la transformación \( f \) en \( x \) proporciona una representación lineal de cómo las pequeñas perturbaciones en \( x \) afectan a \( f(x) \).

**Hipótesis:**
La estabilización de gradientes en redes neuronales profunda puede ser lograda mediante la regularización del fibrado tangente asociado a cada capa. Específicamente, aplicando el Teorema de Eckart-Young para truncar la descomposición en valores singulares (SVD) de la matriz Jacobiana en cada paso de entrenamiento, podemos evitar la explosión del rango y mejorar la estabilidad del modelo.

### Razonamiento Matemático

1. **Fibrado Tangente y Jacobiana:**
   - El fibrado tangente \( T\mathbb{R}^n \) en un punto \( x \) es el espacio vectorial de todas las derivadas direccionales en \( x \).
   - La matriz Jacobiana \( J_f(x) \) es la representación matricial de la transformación lineal que aproxima \( f \) en \( x \).

2. **Teorema de Eckart-Young:**
   - El Teorema de Eckart-Young establece que la mejor aproximación de rango \( k \) a una matriz \( A \) en términos de norma Frobenius es dada por la truncación de la SVD de \( A \) a los \( k \) primeros valores singulares.
   - Aplicando este teorema a la matriz Jacobiana \( J_f(x) \), podemos truncarla a un rango menor para regularizar la red neuronal.

3. **Optimización y Estabilidad:**
   - La regularización mediante truncamiento SVD ayuda a evitar la explosión del rango de la matriz Jacobiana, lo que a su vez estabiliza el entrenamiento de la red neuronal.
   - Esto es particularmente útil en redes neuronales profundas, donde el problema de la explosión del rango puede llevar a inestabilidades y dificultades en la convergencia.

### Script Python Autocontenido

```python
import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import numpy as np
from scipy.linalg import svd

# Función para calcular la matriz Jacobiana
def jacobian(f, x):
    return jax.jacobian(f)(x)

# Función para truncar la SVD de una matriz a rango k
def truncate_svd(A, k):
    U, S, V = svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ V[:k, :]

# Función de ejemplo: una red neuronal simple
def neural_network(x, W1, b1, W2, b2):
    h1 = jnp.tanh(jnp.dot(x, W1) + b1)
    return jnp.dot(h1, W2) + b2

# Parámetros de la red neuronal
W1 = jnp.array([[0.1, 0.2], [0.3, 0.4]])
b1 = jnp.array([0.1, 0.2])
W2 = jnp.array([[0.5, 0.6], [0.7, 0.8]])
b2 = jnp.array([0.3, 0.4])

# Entrada de prueba
x = jnp.array([1.0, 2.0])

# Calcular la matriz Jacobiana
J = jacobian(lambda x: neural_network(x, W1, b1, W2, b2), x)

# Truncar la SVD de la matriz Jacobiana a rango k
k = 1
J_truncated = truncate_svd(J, k)

# Función de pérdida
def loss(W1, b1, W2, b2, x, y):
    y_pred = neural_network(x, W1, b1, W2, b2)
    return jnp.mean((y_pred - y) ** 2)

# Gradiente de la función de pérdida
grad_loss = jit(grad(loss, argnums=(0, 1, 2, 3)))

# Datos de entrenamiento
X_train = jnp.array([[1.0, 2.0], [2.0, 3.0]])
y_train = jnp.array([[0.5, 0.6], [0.7, 0.8]])

# Entrenamiento con regularización SVD
for i in range(100):
    gradients = vmap(grad_loss, in_axes=(None, None, None, None, 0, 0))(W1, b1, W2, b2, X_train, y_train)
    W1 -= 0.01 * gradients[0].mean(axis=0)
    b1 -= 0.01 * gradients[1].mean(axis=0)
    W2 -= 0.01 * gradients[2].mean(axis=0)
    b2 -= 0.01 * gradients[3].mean(axis=0)
    
    # Truncar la SVD de la matriz Jacobiana
    J = jacobian(lambda x: neural_network(x, W1, b1, W2, b2), x)
    J_truncated = truncate_svd(J, k)
    
    # Actualizar los pesos con la matriz Jacobiana truncada
    W1 = J_truncated @ W1
    W2 = J_truncated @ W2

# Imprimir los pesos finales
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)
```

### Explicación del Código

1. **Cálculo de la Matriz Jacobiana:**
   - Usamos `jax.jacobian` para calcular la matriz Jacobiana de la red neuronal en un punto dado.

2. **Truncamiento SVD:**
   - La función `truncate_svd` aplica el Teorema de Eckart-Young para truncar la SVD de la matriz Jacobiana a un rango \( k \).

3. **Función de Pérdida y Gradiente:**
   - Definimos una función de pérdida y su gradiente usando `jax.grad`.

4. **Entrenamiento con Regularización SVD:**
   - Realizamos el entrenamiento de la red neuronal usando el gradiente descendente.
   - En cada paso de entrenamiento, truncamos la SVD de la matriz Jacobiana y actualizamos los pesos de la red neuronal con la matriz Jacobiana truncada.

Este enfoque combina la teoría de fibrados tangentes y la regularización SVD para mejorar la estabilidad y el rendimiento de las redes neuronales profundas.