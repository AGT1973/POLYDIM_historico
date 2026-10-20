use ndarray::{Array2, s};
use ndarray_linalg::SVD;
use rayon::prelude::*;
use nalgebra::linalg::QR;

fn qr_decomposition(matrix: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let qr = QR::new(matrix.clone());
    (qr.q(), qr.r())
}

fn svd_truncation(matrix: &Array2<f64>, k: usize) -> Array2<f64> {
    let (u, s, vt) = matrix.svd(true, true).unwrap();
    let s_truncated = Array2::from_diag(&s.slice(s![..k])).insert_axis(ndarray::Axis(1));
    let u_truncated = u.slice(s![.., ..k]);
    let vt_truncated = vt.slice(s![..k, ..]);
    u_truncated.dot(&s_truncated.dot(&vt_truncated))
}

fn periodic_truncation_svd(matrix: &mut Array2<f64>, k: usize, steps: usize) {
    for _ in 0..steps {
        let (q, r) = qr_decomposition(matrix);
        *matrix = svd_truncation(&r, k);
    }
}

fn main() {
    let initial_matrix = Array2::from_shape_fn((10, 10), |(i, j)| (i + j) as f64);
    let k = 5; // Rango objetivo
    let steps = 10; // Número de pasos de truncamiento

    let mut matrix = initial_matrix.clone();
    periodic_truncation_svd(&mut matrix, k, steps);

    println!("Matriz inicial:\n{}", initial_matrix);
    println!("Matriz después de truncamiento SVD:\n{}", matrix);
}