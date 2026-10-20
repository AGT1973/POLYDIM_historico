use ndarray::{Array2, ArrayView2, s};
use ndarray_linalg::qr::QR;
use ndarray_linalg::svd::SVDDecomposed;
use rayon::prelude::*;

fn qr_decomposition(matrix: ArrayView2<f64>) -> (Array2<f64>, Array2<f64>) {
    let qr = QR::new(matrix.to_owned());
    (qr.q().unwrap(), qr.r().unwrap())
}

fn svd_truncation(matrix: ArrayView2<f64>, rank: usize) -> Array2<f64> {
    let svd = matrix.svd(true, true).unwrap();
    let u = svd.u.unwrap();
    let v = svd.v_t.unwrap();
    let s = svd.s.unwrap();

    let s_truncated = s.slice(s![..rank]).to_owned();
    let u_truncated = u.slice(s![.., ..rank]).to_owned();
    let v_truncated = v.slice(s![..rank, ..]).to_owned();

    u_truncated.dot(&s_truncated.diag()).dot(&v_truncated)
}

fn apply_topological_regularization(matrix: ArrayView2<f64>, rank: usize) -> Array2<f64> {
    // Step 1: QR Decomposition
    let (q, r) = qr_decomposition(matrix);

    // Step 2: SVD Truncation
    let r_truncated = svd_truncation(r.view(), rank);

    // Step 3: Reconstruct the matrix
    q.dot(&r_truncated)
}

fn main() {
    // Example matrix
    let matrix = Array2::from_shape_vec((100, 100), (0..10000).map(|x| x as f64).collect()).unwrap();

    // Desired rank for truncation
    let rank = 10;

    // Apply topological regularization
    let regularized_matrix = apply_topological_regularization(matrix.view(), rank);

    // Print the shape of the regularized matrix
    println!("Shape of the regularized matrix: {:?}", regularized_matrix.shape());

    // Example parallel processing
    let parallel_result: Vec<Array2<f64>> = (0..10).into_par_iter().map(|_| {
        apply_topological_regularization(matrix.view(), rank)
    }).collect();

    println!("Parallel processing complete. Number of matrices: {}", parallel_result.len());
}