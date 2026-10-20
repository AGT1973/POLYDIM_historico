use ndarray::{Array2, ArrayView2, s};
use ndarray_linalg::SVD;
use rayon::prelude::*;
use ndarray_linalg::qr::QR;

fn svd_truncation(A: ArrayView2<f64>, k: usize) -> Array2<f64> {
    let svd = A.svd(true, true).unwrap();
    let U = svd.U.unwrap();
    let Vt = svd.Vt.unwrap();
    let S = svd.S.unwrap();

    let mut S_truncated = Array2::zeros((k, k));
    for i in 0..k {
        S_truncated[[i, i]] = S[i];
    }

    U.slice(s![.., ..k]) * S_truncated * Vt.slice(s![..k, ..])
}

fn qr_svd_truncation(A: ArrayView2<f64>, k: usize) -> Array2<f64> {
    let qr = A.qr().unwrap();
    let Q = qr.Q.unwrap();
    let R = qr.R.unwrap();

    svd_truncation(R.view(), k) * Q.t().view()
}

fn main() {
    let A: Array2<f64> = Array2::random((100, 100), rand::distributions::Standard);
    let k = 10;

    let result = qr_svd_truncation(A.view(), k);

    println!("Truncada SVD (rango {}):\n{}", k, result);
}