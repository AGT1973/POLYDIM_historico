use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use ndarray_linalg::{SVD, QR};
use rayon::prelude::*;

fn svd_truncation(mut tensor: Array2<f64>, rank: usize) -> Array2<f64> {
    let (u, s, vt) = tensor.svd(true, true).unwrap();
    let s_truncated = s.slice(s![..rank]);
    let u_truncated = u.slice(s![.., ..rank]);
    let vt_truncated = vt.slice(s![..rank, ..]);

    u_truncated.dot(&s_truncated.diag()).dot(&vt_truncated)
}

fn qr_decomposition(tensor: Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (q, r) = tensor.qr(false).unwrap();
    (q, r)
}

fn update_tensor(tensor: Array2<f64>, gradient: Array2<f64>, learning_rate: f64, rank: usize) -> Array2<f64> {
    let new_tensor = tensor - learning_rate * gradient;
    svd_truncation(new_tensor, rank)
}

fn main() {
    // Ejemplo de tensor de rango bajo
    let tensor = Array2::from_shape_vec((100, 100), (0..10000).map(|x| x as f64).collect()).unwrap();
    let gradient = Array2::from_shape_vec((100, 100), (0..10000).map(|x| x as f64 / 10000.0).collect()).unwrap();
    let learning_rate = 0.01;
    let target_rank = 10;

    // Actualización del tensor con truncamiento SVD
    let updated_tensor = update_tensor(tensor, gradient, learning_rate, target_rank);

    // Imprimir el tensor actualizado
    println!("Updated Tensor:\n{}", updated_tensor);
}