use ndarray::{Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::svd::SVD;
use ndarray_linalg::qr::QR;
use rayon::prelude::*;
use std::sync::Arc;

/// Aplica la descomposición QR a una matriz
fn qr_decomposition<T: Data<Elem = f64>>(matrix: &ArrayBase<T, Ix2>) -> (Array2<f64>, Array2<f64>) {
    let qr = matrix.qr().unwrap();
    (qr.q().unwrap(), qr.r().unwrap())
}

/// Aplica la descomposición SVD a una matriz y trunca a un rango específico
fn svd_truncation(matrix: &Array2<f64>, rank: usize) -> Array2<f64> {
    let svd = matrix.svd(true, true).unwrap();
    let u = svd.u.unwrap();
    let v = svd.v_t.unwrap();
    let s = svd.s.unwrap();

    let k = rank.min(s.len());
    let u_truncated = u.slice(s![.., ..k]);
    let s_truncated = Array2::from_diag(&s.slice(s![..k]));
    let v_truncated = v.slice(s![..k, ..]);

    u_truncated.dot(&s_truncated.dot(&v_truncated))
}

/// Función principal que aplica TTT a una lista de tensores
fn topological_tensor_truncation(tensors: Vec<Array2<f64>>, rank: usize) -> Vec<Array2<f64>> {
    tensors.par_iter()
        .map(|tensor| {
            let (q, r) = qr_decomposition(tensor);
            svd_truncation(&r, rank)
        })
        .collect()
}

fn main() {
    // Ejemplo de tensores de bajo rango
    let tensors: Vec<Array2<f64>> = vec![
        Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap(),
        Array2::from_shape_vec((3, 3), vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap(),
    ];

    let rank: usize = 2; // Rango de truncación

    // Aplicar TTT
    let truncated_tensors = topological_tensor_truncation(tensors, rank);

    // Imprimir los tensores truncados
    for (i, tensor) in truncated_tensors.iter().enumerate() {
        println!("Tensor {} after TTT:\n{}", i + 1, tensor);
    }
}