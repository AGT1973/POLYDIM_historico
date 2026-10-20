use ndarray::prelude::*;
use ndarray_linalg::svd::*;
use rayon::prelude::*;
use std::time::Instant;

fn truncate_svd(a: &Array2<f64>, k: usize) -> Array2<f64> {
    let (u, s, vt) = a.svd(true, true).unwrap();
    let s_truncated = s.slice(s![..k]);
    let u_truncated = u.slice(s![.., ..k]);
    let vt_truncated = vt.slice(s![..k, ..]);
    u_truncated.dot(&s_truncated.diag().dot(&vt_truncated))
}

fn stabilize_gradients(weights: &mut Vec<Array2<f64>>, k: usize) {
    weights.par_iter_mut().for_each(|w| {
        *w = truncate_svd(w, k);
    });
}

fn main() {
    let start = Instant::now();
    let mut weights = vec![
        Array2::linspace(0.0, 1.0, 100).into_shape((10, 10)).unwrap(),
        Array2::linspace(1.0, 2.0, 100).into_shape((10, 10)).unwrap(),
    ];

    let k = 5; // Rango de truncamiento

    stabilize_gradients(&mut weights, k);

    for (i, w) in weights.iter().enumerate() {
        println!("Weights after SVD truncation for layer {}: {:?}", i, w);
    }

    let elapsed = start.elapsed();
    println!("Time elapsed: {:.2?}", elapsed);
}