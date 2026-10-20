use ndarray::{Array2, Array1, arr1, arr2, s};
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::solve::solve_into;
use rayon::prelude::*;
use std::time::Instant;

// Función de activación estabilizada
fn tanh_stabilized(x: f64, alpha: f64) -> f64 {
    (alpha * x).tanh()
}

// Derivada de la función de activación estabilizada
fn tanh_stabilized_derivative(x: f64, alpha: f64) -> f64 {
    1.0 - (alpha * x).tanh().powi(2)
}

// Capa de normalización (simplificada)
fn normalize(arr: &Array1<f64>) -> Array1<f64> {
    let mean = arr.mean().unwrap();
    let std = arr.std(0.0);
    (arr - mean) / std
}

// Propagación hacia adelante
fn forward_pass(x: &Array1<f64>, weights: &[Array2<f64>], biases: &[Array1<f64>], alpha: f64) -> Result<Array1<f64>, LinalgError> {
    let mut activations = vec![x.clone()];
    for (w, b) in weights.iter().zip(biases.iter()) {
        let z = activations.last().unwrap().dot(w) + b;
        let a = z.mapv(|x| tanh_stabilized(x, alpha));
        let a_norm = normalize(&a);
        activations.push(a_norm);
    }
    Ok(activations.pop().unwrap())
}

// Propagación hacia atrás
fn backward_pass(activations: &[Array1<f64>], weights: &[Array2<f64>], biases: &[Array1<f64>], alpha: f64, target: &Array1<f64>) -> Result<(Vec<Array2<f64>>, Vec<Array1<f64>>), LinalgError> {
    let mut gradients = vec![];
    let mut delta = &activations.last().unwrap() - target;
    let mut weight_grads = vec![];
    let mut bias_grads = vec![];

    for (i, (w, b)) in weights.iter().zip(biases.iter()).enumerate().rev() {
        let a = &activations[i];
        let z = a.dot(w) + b;
        let delta = delta * z.mapv(|x| tanh_stabilized_derivative(x, alpha));
        let weight_grad = a.outer(delta);
        let bias_grad = delta.clone();

        weight_grads.push(weight_grad);
        bias_grads.push(bias_grad);

        if i > 0 {
            delta = delta.dot(&w.t());
        }
    }

    weight_grads.reverse();
    bias_grads.reverse();

    Ok((weight_grads, bias_grads))
}

fn main() -> Result<(), LinalgError> {
    let input_size = 10;
    let hidden_size = 20;
    let output_size = 5;

    let x = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let target = arr1(&[0.1, 0.2, 0.3, 0.4, 0.5]);

    let weights = vec![
        Array2::random((input_size, hidden_size), rand::distributions::Standard),
        Array2::random((hidden_size, output_size), rand::distributions::Standard),
    ];

    let biases = vec![
        Array1::random(hidden_size, rand::distributions::Standard),
        Array1::random(output_size, rand::distributions::Standard),
    ];

    let alpha = 1.0;

    let start = Instant::now();
    let output = forward_pass(&x, &weights, &biases, alpha)?;
    let (weight_grads, bias_grads) = backward_pass(&activations, &weights, &biases, alpha, &target)?;
    let elapsed = start.elapsed();

    println!("Output: {:?}", output);
    println!("Time elapsed: {:.2?}", elapsed);

    // Update weights and biases using gradients (e.g., gradient descent)
    // This is a simplified example, in practice you would use an optimizer

    Ok(())
}