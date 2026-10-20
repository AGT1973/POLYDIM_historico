use ndarray::{Array2, s, concatenate, Axis};
use ndarray_linalg::svd::{SVD, compute_svd};
use ndarray_linalg::qr::QR;
use rayon::prelude::*;
use std::time::Instant;

fn truncate_svd(matrix: &Array2<f64>, k: usize) -> Array2<f64> {
    let svd = compute_svd(matrix, true, true).unwrap();
    let (u, s, vt) = (svd.u.unwrap(), svd.s, svd.v_t.unwrap());

    let u_truncated = u.slice(s![.., ..k]).to_owned();
    let s_truncated = s.slice(s![..k]).to_owned();
    let vt_truncated = vt.slice(s![..k, ..]).to_owned();

    u_truncated.dot(&s_truncated.diag()).dot(&vt_truncated)
}

fn stabilize_gradient(matrix: &mut Array2<f64>, k: usize) {
    *matrix = truncate_svd(matrix, k);
}

fn main() {
    let size = 100;
    let rank = 10;
    let iterations = 100;

    let mut weights = Array2::random((size, size), rand::distributions::Standard);

    let start = Instant::now();

    (0..iterations).into_par_iter().for_each(|_| {
        // Simulación de una operación de red neuronal (por ejemplo, una capa lineal)
        let output = weights.dot(&weights.t());

        // Estabilización de gradientes mediante truncamiento SVD
        stabilize_gradient(&mut weights, rank);
    });

    let duration = start.elapsed();
    println!("Tiempo de ejecución: {:?}", duration);
}