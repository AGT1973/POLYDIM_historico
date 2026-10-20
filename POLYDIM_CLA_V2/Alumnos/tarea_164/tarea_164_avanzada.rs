use ndarray::{Array2, s};
use ndarray_linalg::{SVD, QR};
use rayon::prelude::*;

fn truncate_svd(A: &Array2<f64>, k: usize) -> Array2<f64> {
    let (U, S, Vt) = A.svd(true, true).unwrap();
    let S_k = S.slice(s![0..k]);
    let U_k = U.slice(s![.., 0..k]);
    let Vt_k = Vt.slice(s![0..k, ..]);
    U_k.dot(&S_k.diag().dot(&Vt_k))
}

fn compose_tensors_with_truncation(A: &Array2<f64>, B: &Array2<f64>, k: usize) -> Array2<f64> {
    let AB = A.dot(B);
    truncate_svd(&AB, k)
}

fn main() {
    let size = 100;
    let rank = 10;

    // Generar tensores de prueba
    let A = Array2::from_shape_fn((size, size), |(i, j)| (i + j) as f64);
    let B = Array2::from_shape_fn((size, size), |(i, j)| (i * j) as f64);

    // Componer tensores con truncamiento SVD periódico
    let result = compose_tensors_with_truncation(&A, &B, rank);

    println!("Resultado de la composición con truncamiento SVD:");
    println!("{:?}", result);

    // Validación empírica: Comparar con la composición sin truncamiento
    let result_without_truncation = A.dot(&B);
    let error = (&result - &result_without_truncation).mapv(f64::abs).sum();

    println!("Error de truncamiento: {}", error);
}