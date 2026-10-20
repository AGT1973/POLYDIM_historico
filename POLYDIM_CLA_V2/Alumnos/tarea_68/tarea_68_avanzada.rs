use ndarray::{Array2, Array1, arr1, arr2};
use ndarray_linalg::solve::Inverse;
use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

fn isometric_transform(x: &Array1<f64>, theta: &Array1<f64>) -> Array1<f64> {
    // Ejemplo de transformación isométrica: rotación en 2D
    let theta = theta[0];
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let rotation_matrix = arr2(&[[cos_theta, -sin_theta], [sin_theta, cos_theta]]);
    rotation_matrix.dot(x)
}

fn neural_network(x: &Array1<f64>, weights: &Array2<f64>, biases: &Array1<f64>) -> f64 {
    // Simple perceptrón lineal
    weights.dot(x) + biases
}

fn loss(y_pred: f64, y_true: f64) -> f64 {
    (y_pred - y_true).powi(2)
}

fn gradient_descent(x: &Array1<f64>, y_true: f64, weights: &Array2<f64>, biases: &Array1<f64>, learning_rate: f64) -> (Array2<f64>, Array1<f64>) {
    let y_pred = neural_network(x, weights, biases);
    let grad_weights = (y_pred - y_true) * x;
    let grad_biases = y_pred - y_true;

    let new_weights = weights - learning_rate * &grad_weights;
    let new_biases = biases - learning_rate * grad_biases;

    (new_weights, new_biases)
}

fn main() {
    let mut rng = rand::thread_rng();
    let x = arr1(&[1.0, 1.0]);
    let y_true = 1.0;
    let weights = Array2::random((1, 2), StandardNormal);
    let biases = Array1::random(1, StandardNormal);
    let learning_rate = 0.01;
    let num_iterations = 1000;

    for _ in 0..num_iterations {
        let theta = arr1(&[rng.gen::<f64>() * std::f64::consts::PI * 2.0]);
        let x_transformed = isometric_transform(&x, &theta);
        let (new_weights, new_biases) = gradient_descent(&x_transformed, y_true, &weights, &biases, learning_rate);
        weights.assign(&new_weights);
        biases.assign(&new_biases);
    }

    let y_pred = neural_network(&x, &weights, &biases);
    println!("Final prediction: {}", y_pred);
    println!("Final loss: {}", loss(y_pred, y_true));
}