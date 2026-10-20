use ndarray::{Array2, Array1, s};
use rayon::prelude::*;
use rand::distributions::Standard;
use rand::Rng;

// Definir una isometría (en este caso, una rotación en 2D)
fn isometry(x: &Array1<f64>, theta: f64) -> Array1<f64> {
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let rotation_matrix = Array2::from_shape_vec((2, 2), vec![
        cos_theta, -sin_theta,
        sin_theta, cos_theta,
    ]).unwrap();
    rotation_matrix.dot(x)
}

// Definir la función de activación ReLU
fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

// Definir la red neuronal
fn neural_network(x: &Array1<f64>, w1: &Array2<f64>, b1: &Array1<f64>, w2: &Array2<f64>, b2: &Array1<f64>) -> Array1<f64> {
    let hidden = w1.dot(x) + b1;
    let hidden_activated = hidden.mapv(relu);
    w2.dot(&hidden_activated) + b2
}

// Función principal para validar la invariancia isométrica
fn main() {
    let rng = &mut rand::thread_rng();
    let input_dim = 2;
    let hidden_dim = 3;
    let output_dim = 1;

    // Inicializar matrices de pesos y sesgos
    let w1: Array2<f64> = Array2::from_shape_fn((hidden_dim, input_dim), |_| rng.sample(Standard));
    let b1: Array1<f64> = Array1::from_shape_fn(hidden_dim, |_| rng.sample(Standard));
    let w2: Array2<f64> = Array2::from_shape_fn((output_dim, hidden_dim), |_| rng.sample(Standard));
    let b2: Array1<f64> = Array1::from_shape_fn(output_dim, |_| rng.sample(Standard));

    // Vector de entrada
    let x: Array1<f64> = Array1::from_shape_fn(input_dim, |_| rng.sample(Standard));

    // Ángulo de rotación
    let theta = 0.5; // Ejemplo de ángulo de rotación

    // Calcular la salida de la red neuronal para el vector de entrada original
    let output_original = neural_network(&x, &w1, &b1, &w2, &b2);

    // Aplicar la isometría (rotación) al vector de entrada
    let x_rotated = isometry(&x, theta);

    // Calcular la salida de la red neuronal para el vector de entrada rotado
    let output_rotated = neural_network(&x_rotated, &w1, &b1, &w2, &b2);

    // Comprobar la invariancia isométrica
    let tolerance = 1e-6;
    if (output_original - output_rotated).mapv(|x| x.abs()).iter().all(|&x| x < tolerance) {
        println!("La red neuronal es invariante a isometrías.");
    } else {
        println!("La red neuronal no es invariante a isometrías.");
    }
}