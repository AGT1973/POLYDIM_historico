use ndarray::{Array2, s};
use ndarray_linalg::svd::SVD;
use rayon::prelude::*;
use ndarray::prelude::*;
use ndarray_linalg::eigh::Eigh;

fn qr_decomposition(matrix: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (q, r) = matrix.qr().unwrap();
    (q, r)
}

fn svd_truncation(matrix: &Array2<f64>, k: usize) -> Array2<f64> {
    let svd = matrix.svd(true, true).unwrap();
    let u = svd.u.unwrap();
    let s = svd.s.unwrap();
    let vt = svd.v_t.unwrap();

    let s_k = s.slice(s![..k]);
    let u_k = u.slice(s![.., ..k]);
    let vt_k = vt.slice(s![..k, ..]);

    let s_k_diag = Array2::from_diag(&s_k.to_vec());
    let truncated_matrix = u_k.dot(&s_k_diag).dot(&vt_k);

    truncated_matrix
}

fn compose_tensors(tensor1: &Array2<f64>, tensor2: &Array2<f64>, k: usize) -> Array2<f64> {
    let (q1, r1) = qr_decomposition(tensor1);
    let (q2, r2) = qr_decomposition(tensor2);

    let composed_matrix = r1.dot(&r2);
    let truncated_matrix = svd_truncation(&composed_matrix, k);

    q1.dot(&truncated_matrix).dot(&q2)
}

fn main() {
    let tensor1 = Array2::from_elem((100, 100), 1.0);
    let tensor2 = Array2::from_elem((100, 100), 1.0);
    let k = 10;

    let result = compose_tensors(&tensor1, &tensor2, k);
    println!("Composed and truncated tensor:\n{}", result);
}