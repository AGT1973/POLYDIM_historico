use ndarray::{Array2, arr2};
use ndarray_linalg::{svd, qr};
use rayon::prelude::*;

fn truncated_svd(A: &Array2<f64>, k: usize) -> Array2<f64> {
    let (U, S, Vt) = A.svd(true, true).unwrap();
    let S_k = S.slice(s![0..k]);
    let U_k = U.slice(s![.., 0..k]);
    let Vt_k = Vt.slice(s![0..k, ..]);
    U_k.dot(&S_k.diag()).dot(&Vt_k)
}

fn compose_and_truncate(A: &Array2<f64>, B: &Array2<f64>, k: usize) -> Array2<f64> {
    let C = A.dot(B);
    truncated_svd(&C, k)
}

fn main() {
    let A = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let B = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
    let k = 1;

    let result = compose_and_truncate(&A, &B, k);
    println!("Truncated SVD Result:\n{}", result);
}