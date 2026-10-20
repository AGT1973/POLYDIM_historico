### Concepto Teórico: Topología Diferencial y Redes Neuronales

#### Benchmark Matemático: Invariancia Isométrica en Redes Neuronales

**Concepto:**
La invariancia isométrica es una propiedad fundamental en la topología diferencial que se refiere a la preservación de la estructura métrica bajo transformaciones. En el contexto de las redes neuronales, esta propiedad puede ser utilizada para construir modelos que sean invariantes a ciertas transformaciones geométricas, como rotaciones y traslaciones. Este concepto es especialmente útil en tareas de reconocimiento de patrones y clasificación, donde la invariancia a las transformaciones puede mejorar la robustez del modelo.

**Benchmark:**
Diseñaremos un benchmark para evaluar la invariancia isométrica de una red neuronal. La idea es entrenar una red neuronal para clasificar imágenes y luego evaluar su rendimiento bajo diferentes transformaciones isométricas (rotaciones y traslaciones). El objetivo es verificar si la red mantiene un rendimiento similar a pesar de estas transformaciones.

### Razonamiento Matemático

**1. Definición de Invariancia Isométrica:**
Una función \( f: \mathbb{R}^n \to \mathbb{R}^m \) es isométrica si preserva la distancia entre puntos, es decir, para cualquier \( x, y \in \mathbb{R}^n \):
\[ d(f(x), f(y)) = d(x, y) \]
donde \( d \) es una métrica en \( \mathbb{R}^n \) y \( \mathbb{R}^m \).

**2. Aplicación a Redes Neuronales:**
En el contexto de las redes neuronales, queremos que la salida de la red sea invariante a ciertas transformaciones isométricas aplicadas a la entrada. Formalmente, si \( T \) es una transformación isométrica (por ejemplo, una rotación o traslación), queremos que:
\[ f(T(x)) = f(x) \]

**3. Evaluación Empírica:**
Para evaluar la invariancia isométrica, entrenaremos una red neuronal en un conjunto de datos y luego aplicaremos la red a versiones transformadas de las imágenes de validación. Compararemos el rendimiento de la red en las imágenes originales y transformadas.

### Script Python

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.ndimage import rotate, shift
import jax
import jax.numpy as jnp
from jax import vmap, jit, grad

# Definición del modelo
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 14 * 14)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Transformaciones isométricas
def rotate_image(image, angle):
    return rotate(image, angle, reshape=False, mode='constant')

def shift_image(image, shift_x, shift_y):
    return shift(image, (shift_x, shift_y), mode='constant')

# Carga del conjunto de datos
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Entrenamiento del modelo
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # 10 epochs
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

# Evaluación en el conjunto de test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy on the 10000 test images: {100 * correct / total}%')

# Evaluación con transformaciones isométricas
angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
shifts = [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2)]

for angle in angles:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = torch.stack([torch.tensor(rotate_image(img.squeeze().numpy(), angle)) for img in images])
            images = images.unsqueeze(1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on rotated images by {angle} degrees: {100 * correct / total}%')

for shift in shifts:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = torch.stack([torch.tensor(shift_image(img.squeeze().numpy(), *shift)) for img in images])
            images = images.unsqueeze(1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on shifted images by {shift}: {100 * correct / total}%')
```

### Explicación del Código

1. **Definición del Modelo:** Se define una red neuronal simple con capas convolucionales y fully connected.
2. **Transformaciones Isométricas:** Se implementan funciones para rotar y trasladar imágenes.
3. **Carga de Datos:** Se cargan los datos de MNIST y se aplican transformaciones necesarias.
4. **Entrenamiento:** Se entrena la red neuronal durante 10 epochs.
5. **Evaluación:** Se evalúa la precisión del modelo en el conjunto de test y se aplica la evaluación a imágenes rotadas y trasladadas.
6. **Resultados:** Se imprime la precisión del modelo en las imágenes originales y transformadas.

Este script permite evaluar empíricamente la invariancia isométrica de la red neuronal, lo que es crucial para entender su robustez frente a transformaciones geométricas.