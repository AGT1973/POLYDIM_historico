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