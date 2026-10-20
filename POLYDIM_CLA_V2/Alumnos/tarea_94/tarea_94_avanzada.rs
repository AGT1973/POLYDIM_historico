use ndarray::prelude::*;
use ndarray::Zip;
use rayon::prelude::*;
use ndarray_linalg::solve::Inverse;
use ndarray_linalg::error::LinalgError;
use std::f64;

fn main() -> Result<(), LinalgError> {
    // Definir los parámetros
    let input_dim = 10;
    let output_dim = 5;
    let num_points = 1000;

    // Generar datos de entrada aleatorios
    let x = Array::random((num_points, input_dim), rand::distributions::Standard);

    // Definir una red neuronal simple (lineal por simplicidad)
    let weights = Array::random((output_dim, input_dim), rand::distributions::Standard);
    let bias = Array::random((output_dim,), rand::distributions::Standard);

    // Función de pérdida (MSE)
    let target = Array::random((num_points, output_dim), rand::distributions::Standard);

    // Función de la red neuronal
    let f = |x: &Array2<f64>| -> Array2<f64> {
        x.dot(&weights.t()) + &bias
    };

    // Función de pérdida
    let loss = |y_pred: &Array2<f64>, y_true: &Array2<f64>| -> f64 {
        (y_pred - y_true).mapv(|x| x.powi(2)).mean().unwrap()
    };

    // Gradiente de la función de pérdida respecto a la salida
    let grad_loss = |y_pred: &Array2<f64>, y_true: &Array2<f64>| -> Array2<f64> {
        2.0 * (y_pred - y_true)
    };

    // Gradiente de la red neuronal
    let grad_f = |x: &Array2<f64>| -> Array2<f64> {
        weights.clone()
    };

    // Calcular el gradiente del error respecto a la entrada
    let grad_x = |x: &Array2<f64>, target: &Array2<f64>| -> Array2<f64> {
        let y_pred = f(x);
        let grad_y = grad_loss(&y_pred, target);
        grad_y.dot(&grad_f(x))
    };

    // Calcular la norma del gradiente para cada punto de entrada
    let norms: Vec<f64> = x.axis_iter(Axis(0))
        .into_par_iter()
        .map(|x_i| {
            let x_i = x_i.insert_axis(Axis(0));
            let grad_i = grad_x(&x_i, &target.slice(s![x_i.shape()[0], ..]));
            grad_i.mapv(|x| x.powi(2)).sum().sqrt()
        })
        .collect();

    // Verificar que la norma del gradiente está uniformemente acotada
    let max_norm = norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_norm = norms.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("Max norm: {}", max_norm);
    println!("Min norm: {}", min_norm);

    Ok(())
}