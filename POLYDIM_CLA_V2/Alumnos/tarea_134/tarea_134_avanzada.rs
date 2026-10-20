use ndarray::{Array2, Array, s};
use ndarray_linalg::{SVD, QR};
use rayon::prelude::*;

fn svd_truncation(A: &Array2<f64>, k: usize) -> Array2<f64> {
    // Perform QR decomposition
    let (Q, R) = A.qr().unwrap();
    
    // Perform SVD on R
    let (U, S, Vt) = R.svd(true, true).unwrap();
    
    // Truncate singular values
    let S_truncated = S.slice(s![..k]).to_owned();
    let U_truncated = U.slice(s![.., ..k]).to_owned();
    let Vt_truncated = Vt.slice(s![..k, ..]).to_owned();
    
    // Reconstruct truncated matrix
    let S_diag = Array::from_diag(&S_truncated);
    Q.dot(&U_truncated).dot(&S_diag).dot(&Vt_truncated)
}

fn main() {
    // Example tensor
    let A: Array2<f64> = Array::random((100, 100), rand::distributions::Standard);
    
    // Truncation rank
    let k = 10;
    
    // Perform truncation
    let A_truncated = svd_truncation(&A, k);
    
    // Print result
    println!("Truncated Tensor:\n{}", A_truncated);
}