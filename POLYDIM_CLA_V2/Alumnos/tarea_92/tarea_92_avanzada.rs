use ndarray::{Array, Array2, ArrayView};
use ndarray_linalg::Norm;
use rayon::prelude::*;
use std::f64;

fn isometry(x: &Array2<f64>, theta: f64) -> Array2<f64> {
    let (m, n) = (x.shape()[0], x.shape()[1]);
    let mut y = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            y[[i, j]] = x[[i, j]].cos() * theta - x[[i, (j + 1) % n]].sin() * theta;
            y[[i, (j + 1) % n]] = x[[i, j]].sin() * theta + x[[i, (j + 1) % n]].cos() * theta;
        }
    }
    y
}

fn neural_network(x: &Array2<f64>) -> Array2<f64> {
    // Ejemplo simple de una red neuronal: suma de los elementos
    x.sum_axis(ndarray::Axis(1)).to_owned().insert_axis(ndarray::Axis(1))
}

fn gradient(f: &dyn Fn(&Array2<f64>) -> Array2<f64>, x: &Array2<f64>, h: f64) -> Array2<f64> {
    let (m, n) = (x.shape()[0], x.shape()[1]);
    let mut grad = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut x_plus = x.clone();
            x_plus[[i, j]] += h;
            let mut x_minus = x.clone();
            x_minus[[i, j]] -= h;
            grad[[i, j]] = (f(&x_plus) - f(&x_minus))[i] / (2.0 * h);
        }
    }
    grad
}

fn main() {
    let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let theta = std::f64::consts::PI / 4.0;
    let x_transformed = isometry(&x, theta);

    let f = neural_network;
    let grad_x = gradient(&f, &x, 1e-6);
    let grad_x_transformed = gradient(&f, &x_transformed, 1e-6);

    let diff = (grad_x - grad_x_transformed).mapv(|x| x.abs()).sum_axis(ndarray::Axis(1)).sum();
    println!("Diferencia en gradientes: {}", diff);

    if diff < 1e-6 {
        println!("La red neuronal es invariante isométrica.");
    } else {
        println!("La red neuronal no es invariante isométrica.");
    }
}