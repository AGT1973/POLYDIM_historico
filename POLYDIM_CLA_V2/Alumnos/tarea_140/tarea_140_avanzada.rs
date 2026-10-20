use ndarray::{Array2, Array, s};
use ndarray_linalg::SVD;
use rayon::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::StandardNormal;

fn svd_truncation(matrix: &Array2<f64>, k: usize) -> Array2<f64> {
    let (u, s, vt) = matrix.svd(true, true).unwrap();
    let s_truncated = Array::from_vec(s.slice(s![..k]).to_vec());
    let u_truncated = u.slice(s![.., ..k]);
    let vt_truncated = vt.slice(s![..k, ..]);
    u_truncated.dot(&s_truncated.diag().dot(&vt_truncated))
}

fn main() {
    // Parámetros
    let n = 100; // Dimension de la matriz
    let k = 10;  // Rango truncado
    let epochs = 10;

    // Generar una matriz aleatoria
    let mut matrix = Array2::random((n, n), StandardNormal);

    // Iterar y aplicar truncación SVD periódicamente
    for _ in 0..epochs {
        matrix = svd_truncation(&matrix, k);
    }

    // Imprimir la matriz final truncada
    println!("Matriz final truncada:\n{}", matrix);
}