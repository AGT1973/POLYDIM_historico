### 1. Concepto Teórico: Topología y Redes Neuronales

#### Concepto: Topología de Activación en Redes Neuronales

La idea principal es utilizar la topología para analizar y optimizar las funciones de activación en redes neuronales. La topología puede proporcionar una estructura para entender cómo las funciones de activación transforman el espacio de entrada en el espacio de salida, manteniendo ciertas propiedades invariantes.

#### Benchmark: Invariancia Isométrica en Funciones de Activación

**Definición**: Dada una red neuronal con una función de activación \( \sigma \), queremos verificar si la función de activación preserva la distancia isométrica entre los puntos en el espacio de entrada y el espacio de salida. Esto significa que para cualquier par de puntos \( x_1 \) y \( x_2 \) en el espacio de entrada, la distancia entre \( \sigma(x_1) \) y \( \sigma(x_2) \) en el espacio de salida debe ser igual a la distancia entre \( x_1 \) y \( x_2 \) en el espacio de entrada.

### 2. Razonamiento Matemático

La invariancia isométrica se puede expresar matemáticamente como:
\[ d(\sigma(x_1), \sigma(x_2)) = d(x_1, x_2) \]

Donde \( d \) es una métrica de distancia, generalmente la distancia euclidiana:
\[ d(x_1, x_2) = \sqrt{\sum_{i=1}^{n} (x_{1i} - x_{2i})^2} \]

Para verificar esta propiedad, podemos generar un conjunto de puntos en el espacio de entrada, aplicar la función de activación y calcular las distancias antes y después de la transformación.

### 3. Script Rust

Aquí tienes un script Rust que utiliza Rayon para paralelizar las operaciones y ndarray para manejar los tensores. El script verifica la invariancia isométrica de una función de activación simple, como la función sigmoide.

```rust
use ndarray::prelude::*;
use rayon::prelude::*;
use std::f64;

// Función de activación sigmoide
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

// Aplicar la función de activación a un tensor
fn apply_activation(arr: &Array1<f64>) -> Array1<f64> {
    arr.mapv(sigmoid)
}

// Calcular la distancia euclidiana entre dos puntos
fn euclidean_distance(x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
    (x1 - x2).mapv(|x| x.powi(2)).sum().sqrt()
}

// Verificar la invariancia isométrica
fn check_isometric_invariance(input: &Array2<f64>, output: &Array2<f64>) -> bool {
    let n = input.shape()[0];
    let d = input.shape()[1];

    // Verificar para todos los pares de puntos
    (0..n).into_par_iter().all(|i| {
        (0..n).all(|j| {
            if i != j {
                let dist_input = euclidean_distance(&input.slice(s![i, ..]), &input.slice(s![j, ..]));
                let dist_output = euclidean_distance(&output.slice(s![i, ..]), &output.slice(s![j, ..]));
                (dist_input - dist_output).abs() < 1e-6
            } else {
                true
            }
        })
    })
}

fn main() {
    // Generar datos de entrada
    let n = 100; // Número de puntos
    let d = 10;  // Dimensión de los puntos
    let input = Array::from_shape_fn((n, d), |(i, j)| (i as f64 + j as f64) / (n + d) as f64);

    // Aplicar la función de activación a los datos de entrada
    let output = input.axis_iter(Axis(0))
        .map(|row| apply_activation(&row))
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Array2<f64>>();

    // Verificar la invariancia isométrica
    let is_isometric = check_isometric_invariance(&input, &output);

    println!("Is the activation function isometrically invariant? {}", is_isometric);
}
```

### Explicación del Código

1. **Función de Activación Sigmoide**: `sigmoid` aplica la función sigmoide a un valor scalar.
2. **Aplicar Activación a un Tensor**: `apply_activation` aplica la función sigmoide a cada elemento de un tensor unidimensional.
3. **Calcular Distancia Euclidiana**: `euclidean_distance` calcula la distancia euclidiana entre dos puntos en un espacio multidimensional.
4. **Verificar Invariancia Isométrica**: `check_isometric_invariance` verifica si la distancia entre todos los pares de puntos en el espacio de entrada es igual a la distancia entre los mismos puntos en el espacio de salida.
5. **Main Function**: Genera datos de entrada, aplica la función de activación y verifica la invariancia isométrica.

Este script utiliza Rayon para paralelizar la verificación de la invariancia isométrica, lo que mejora la eficiencia computacional.