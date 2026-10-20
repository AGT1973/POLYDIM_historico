use ndarray::{Array2, Array1, s, Axis};
use ndarray_linalg::SVD;
use rayon::prelude::*;
use std::time::Instant;

fn truncated_svd(matrix: &Array2<f64>, k: usize) -> Array2<f64> {
    let (u, s, vt) = matrix.svd(true, true).unwrap();
    let s_truncated = s.slice(s![..k]);
    let u_truncated = u.slice(s![.., ..k]);
    let vt_truncated = vt.slice(s![..k, ..]);
    
    u_truncated.dot(&s_truncated.to_diag()).dot(&vt_truncated)
}

fn main() {
    let n = 1000;
    let k = 10;
    let matrix = Array2::random((n, n), rand::distributions::Standard);

    let start = Instant::now();
    let truncated_matrix = truncated_svd(&matrix, k);
    let elapsed = start.elapsed();

    println!("Truncado SVD completado en {:.2?}", elapsed);
    println!("Norma Frobenius de la matriz original: {}", matrix.frobenius_norm());
    println!("Norma Frobenius de la matriz truncada: {}", truncated_matrix.frobenius_norm());
}