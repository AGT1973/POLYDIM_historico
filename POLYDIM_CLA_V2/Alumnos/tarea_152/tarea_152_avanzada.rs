use ndarray::{Array2, s, Array};
use ndarray_linalg::{SVD, svd};
use rayon::prelude::*;
use std::time::Instant;

/// Función para aplicar el truncamiento SVD a un tensor.
fn truncated_svd(A: &Array2<f64>, k: usize) -> Array2<f64> {
    let (U, S, Vt) = A.svd(true, true).unwrap();
    let S_truncated = S.slice(s![..k]);
    let U_truncated = U.slice(s![.., ..k]);
    let Vt_truncated = Vt.slice(s![..k, ..]);
    U_truncated.dot(&S_truncated.diag().dot(&Vt_truncated))
}

/// Función para aplicar el truncamiento SVD periódico a una lista de tensores.
fn periodic_truncated_svd(tensors: &mut Vec<Array2<f64>>, k: usize, period: usize) {
    let start = Instant::now();
    tensors.par_iter_mut().enumerate().for_each(|(i, tensor)| {
        if i % period == 0 {
            *tensor = truncated_svd(tensor, k);
        }
    });
    println!("Tiempo de procesamiento: {:?}", start.elapsed());
}

fn main() {
    // Crear una lista de tensores de ejemplo
    let mut tensors: Vec<Array2<f64>> = (0..100).map(|_| Array::random((100, 100), rand::distributions::Standard)).collect();

    // Parámetros para el truncamiento SVD
    let k = 10; // Número de valores singulares a mantener
    let period = 10; // Periodo de truncamiento

    // Aplicar el truncamiento SVD periódico
    periodic_truncated_svd(&mut tensors, k, period);

    // Imprimir el primer tensor truncado para verificar
    println!("Primer tensor truncado:\n{}", tensors[0]);
}