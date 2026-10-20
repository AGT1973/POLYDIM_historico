use ndarray::{Array2, Array1, s};
use ndarray_linalg::{SVD, QR};
use rayon::prelude::*;
use std::f64;

// Función para truncar la SVD
fn truncate_svd(matrix: &Array2<f64>, k: usize) -> Array2<f64> {
    let (u, s, vt) = matrix.svd(true, true).unwrap();
    let s_truncated = Array1::from_vec(s.slice(s![..k]).to_vec());
    let u_truncated = u.slice(s![.., ..k]);
    let vt_truncated = vt.slice(s![..k, ..]);
    u_truncated.dot(&s_truncated.diag().dot(&vt_truncated))
}

// Función para estabilizar el gradiente usando TSG
fn stabilize_gradient(matrix: &Array2<f64>, k: usize) -> Array2<f64> {
    let qr = matrix.qr().unwrap();
    let q = qr.q();
    let r = qr.r();
    let r_truncated = truncate_svd(&r, k);
    q.dot(&r_truncated)
}

fn main() {
    // Ejemplo de matriz de pesos de una capa de la red neuronal
    let weights = Array2::random((100, 100), rand::distributions::Standard);

    // Número de valores singulares a mantener
    let k = 10;

    // Estabilizar el gradiente
    let stabilized_weights = stabilize_gradient(&weights, k);

    // Imprimir la matriz estabilizada
    println!("Matriz de pesos estabilizada:\n{}", stabilized_weights);
}