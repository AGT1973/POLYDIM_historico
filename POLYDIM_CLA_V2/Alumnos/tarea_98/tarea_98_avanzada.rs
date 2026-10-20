use ndarray::{Array1, Array2, arr1, arr2};
use rayon::prelude::*;
use ndarray_linalg::solve::Inverse;
use ndarray_linalg::c64;

fn isometry(x: &Array1<f64>, theta: f64) -> Array1<f64> {
    // Aplica una rotación 2D (isometría) al vector x
    let rotation = arr2(&[
        [theta.cos(), -theta.sin()],
        [theta.sin(), theta.cos()],
    ]);
    rotation.dot(x)
}

fn neural_network(x: &Array1<f64>, weights: &Array2<f64>, bias: &Array1<f64>) -> Array1<f64> {
    // Simple red neuronal con una capa oculta
    weights.dot(x) + bias
}

fn main() {
    // Definir entradas y parámetros
    let x = arr1(&[1.0, 0.0]);
    let theta = std::f64::consts::PI / 4.0; // 45 grados
    let weights = arr2(&[[1.0, 0.5], [0.5, 1.0]]);
    let bias = arr1(&[0.1, 0.1]);

    // Calcular salida original
    let y = neural_network(&x, &weights, &bias);

    // Calcular salida después de aplicar la isometría
    let x_rotated = isometry(&x, theta);
    let y_rotated = neural_network(&x_rotated, &weights, &bias);

    // Verificar la invariancia isométrica
    let tolerance = 1e-6;
    let is_invariant = (y - y_rotated).mapv(f64::abs).iter().all(|&val| val < tolerance);

    println!("Salida original: {:?}", y);
    println!("Salida rotada: {:?}", y_rotated);
    println!("Invariancia isométrica: {}", is_invariant);
}