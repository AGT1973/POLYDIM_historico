use ndarray::{Array2, Array1, s, concatenate, Axis};
use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::time::Instant;

fn generate_data(n: usize, d: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = rand::thread_rng();
    let x = Array2::from_shape_fn((n, d), |(_, i)| rng.sample(StandardNormal));
    let y = Array1::from_shape_fn(n, |_| rng.gen_range(0.0..1.0));
    (x, y)
}

fn layer_forward(x: &Array2<f64>, w: &Array2<f64>, b: &Array1<f64>) -> Array2<f64> {
    x.dot(w) + b
}

fn activation(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|a| a.max(0.0)) // ReLU activation
}

fn loss(y_pred: &Array2<f64>, y_true: &Array1<f64>) -> f64 {
    y_pred.column(0).mapv(|a| (a - y_true).powi(2)).mean().unwrap()
}

fn backpropagation(x: &Array2<f64>, y_true: &Array1<f64>, w: &mut Array2<f64>, b: &mut Array1<f64>, learning_rate: f64) {
    let z1 = layer_forward(x, w, b);
    let a1 = activation(&z1);
    let grad_z1 = 2.0 * (a1.column(0) - y_true) * (a1.column(0) > 0.0).mapv(|a| a as f64);
    let grad_w = x.t().dot(&Array2::from_shape_vec((grad_z1.len(), 1), grad_z1.to_vec()).unwrap());
    let grad_b = grad_z1.sum();

    *w -= learning_rate * grad_w;
    *b -= learning_rate * grad_b;
}

fn main() {
    let n = 1000;
    let d = 10;
    let (x, y_true) = generate_data(n, d);
    let mut w = Array2::from_shape_fn((d, 1), |(_, _)| 0.01 * rand::random::<f64>());
    let mut b = Array1::from_shape_fn(1, |_| 0.01 * rand::random::<f64>());
    let learning_rate = 0.01;
    let epochs = 1000;

    let start = Instant::now();
    for _ in 0..epochs {
        backpropagation(&x, &y_true, &mut w, &mut b, learning_rate);
    }
    let duration = start.elapsed();

    let y_pred = activation(&layer_forward(&x, &w, &b));
    let final_loss = loss(&y_pred, &y_true);

    println!("Final loss: {}", final_loss);
    println!("Training time: {:.2?}", duration);
}