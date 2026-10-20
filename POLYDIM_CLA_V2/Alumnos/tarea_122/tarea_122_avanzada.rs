use ndarray::prelude::*;
use ndarray_linalg::LinalgScalar;
use ndarray_linalg::svd::SVD;
use rayon::prelude::*;

fn svd_truncate<T: LinalgScalar>(mut a: Array2<T>, k: usize) -> Array2<T> {
    let (u, s, vt) = a.svd(true, true).unwrap();
    let mut s_truncated = s.slice(s![..k]).to_owned();
    let u_truncated = u.slice(s![.., ..k]).to_owned();
    let vt_truncated = vt.slice(s![..k, ..]).to_owned();

    // Reconstruct the truncated matrix
    let a_truncated = u_truncated.dot(&s_truncated.diag().dot(&vt_truncated));
    a_truncated
}

fn main() {
    // Example tensor of shape (100, 100)
    let a = Array2::random((100, 100), rand::distributions::Standard);

    // Truncate the SVD to rank 10
    let k = 10;
    let a_truncated = svd_truncate(a, k);

    // Print the shape of the truncated tensor
    println!("Truncated tensor shape: {:?}", a_truncated.shape());

    // Parallel processing using Rayon
    let data: Vec<Array2<f64>> = (0..10).into_par_iter().map(|_| {
        let a = Array2::random((100, 100), rand::distributions::Standard);
        svd_truncate(a, k)
    }).collect();

    // Print the shapes of the truncated tensors
    for (i, tensor) in data.iter().enumerate() {
        println!("Tensor {} shape: {:?}", i, tensor.shape());
    }
}