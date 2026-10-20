### 1. Concepto Teórico: Fibrados Tangentes y Redes Neuronales

#### Concepto: Fibrados Tangentes para la Estabilización de Gradientes en Redes Neuronales

**Motivación:**
En redes neuronales, la estabilización de gradientes es crucial para evitar problemas como el desvanecimiento y el explosión de gradientes. Estos problemas pueden ser mitigados mediante la utilización de técnicas que mantienen la geometría del espacio de características más controlada. Un enfoque prometedor es el uso de fibrados tangentes para definir una estructura geométrica sobre el espacio de características de la red neuronal.

**Definición:**
Dada una red neuronal \( f: \mathbb{R}^n \to \mathbb{R}^m \), podemos considerar el fibrado tangente \( TM \) sobre el espacio de entrada \( M = \mathbb{R}^n \). El fibrado tangente \( TM \) es una variedad que, en cada punto \( x \in M \), tiene asociado el espacio tangente \( T_xM \). La idea es que, durante el entrenamiento, los gradientes se proyecten sobre el espacio tangente de manera que se mantengan controlados y estables.

**Objetivo:**
Diseñar un mecanismo que, durante el entrenamiento, proyecte los gradientes sobre el fibrado tangente, asegurando que los gradientes no se desvanezcan ni exploten. Esto se logra mediante la definición de una métrica en el fibrado tangente que mide la "distancia" entre gradientes y ajusta los pesos de la red de manera que los gradientes se mantengan dentro de un rango controlado.

### 2. Razonamiento Matemático

**Fibrado Tangente:**
El fibrado tangente \( TM \) de una variedad \( M \) es la unión disjunta de los espacios tangentes \( T_xM \) para cada punto \( x \in M \). En el contexto de redes neuronales, \( M \) es el espacio de entrada \( \mathbb{R}^n \), y \( T_xM \) es el espacio tangente en el punto \( x \).

**Métrica en el Fibrado Tangente:**
Definimos una métrica \( g \) en \( TM \) que mide la "distancia" entre gradientes. Esta métrica puede ser una simple norma euclidiana o una métrica más compleja que tenga en cuenta la estructura del espacio de características.

**Proyección de Gradientes:**
Durante el entrenamiento, los gradientes \( \nabla f(x) \) se proyectan sobre el espacio tangente \( T_xM \) utilizando la métrica \( g \). Esto se puede escribir como:
\[ \nabla f(x) \mapsto P_{T_xM}(\nabla f(x)) \]
donde \( P_{T_xM} \) es el operador de proyección sobre el espacio tangente.

**Estabilización de Gradientes:**
La proyección de los gradientes sobre el espacio tangente asegura que los gradientes se mantengan dentro de un rango controlado. Esto se logra mediante la regularización de los gradientes, evitando que se desvanezcan o exploten.

### 3. Script Python con JAX, PyTorch y SciPy

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import norm

# Definición de la red neuronal
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Función para calcular el gradiente
def compute_gradient(model, x, y):
    model.zero_grad()
    output = model(x)
    loss = torch.mean((output - y) ** 2)
    loss.backward()
    gradients = []
    for param in model.parameters():
        gradients.append(param.grad.view(-1))
    return torch.cat(gradients)

# Función para proyectar el gradiente sobre el espacio tangente
def project_gradient(gradient, metric):
    # Aquí se define la métrica, por ejemplo, la norma euclidiana
    projected_gradient = gradient / norm(gradient, ord=metric)
    return projected_gradient

# Función de entrenamiento
@jit
def train_step(model, x, y, optimizer, metric):
    gradient = compute_gradient(model, x, y)
    projected_gradient = project_gradient(gradient, metric)
    # Actualización de los pesos
    for param, grad in zip(model.parameters(), vmap(lambda g: g.reshape(param.shape))(projected_gradient)):
        param.grad = torch.tensor(grad, requires_grad=True)
    optimizer.step()

# Datos de ejemplo
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# Inicialización del modelo y del optimizador
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Métrica (norma euclidiana)
metric = 2

# Entrenamiento
for epoch in range(100):
    train_step(model, x, y, optimizer, metric)
    if epoch % 10 == 0:
        output = model(x)
        loss = torch.mean((output - y) ** 2)
        print(f'Epoch {epoch}, Loss: {loss.item()}')
```

### Explicación del Código

1. **Definición de la Red Neuronal:**
   - Se define una red neuronal simple con dos capas lineales y una activación ReLU.

2. **Cálculo del Gradiente:**
   - La función `compute_gradient` calcula el gradiente de la pérdida con respecto a los parámetros del modelo.

3. **Proyección del Gradiente:**
   - La función `project_gradient` proyecta el gradiente sobre el espacio tangente utilizando la norma euclidiana como métrica.

4. **Entrenamiento:**
   - La función `train_step` realiza un paso de entrenamiento, calcula el gradiente, lo proyecta sobre el espacio tangente y actualiza los pesos del modelo.
   - Se utiliza `vmap` para aplicar la proyección a cada parámetro del modelo.

5. **Datos de Ejemplo y Entrenamiento:**
   - Se generan datos de ejemplo y se inicializa el modelo y el optimizador.
   - Se realiza el entrenamiento durante 100 épocas, imprimiendo la pérdida cada 10 épocas.

Este código demuestra cómo se puede utilizar la proyección sobre el fibrado tangente para estabilizar los gradientes en una red neuronal, asegurando que los gradientes se mantengan dentro de un rango controlado durante el entrenamiento.