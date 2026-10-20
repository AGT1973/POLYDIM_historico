use ndarray::{Array2, s, concatenate, Axis};
use rayon::prelude::*;
use ndarray_linalg::SVD;
use ndarray_linalg::types::Lapack;
use ndarray_linalg::error::LinalgError;

fn svd_truncation(matrix: &Array2<f64>, rank: usize) -> Result<Array2<f64>, LinalgError> {
    let (u, s, vt) = matrix.svd(true, true)?;
    let s_truncated = s.slice(s![..rank]);
    let u_truncated = u.slice(s![.., ..rank]);
    let vt_truncated = vt.slice(s![..rank, ..]);

    let approximated_matrix = u_truncated.dot(&s_truncated.dot(&vt_truncated));
    Ok(approximated_matrix)
}

fn apply_ttsn_layer(input: &Array2<f64>, weights: &Array2<f64>, rank: usize) -> Result<Array2<f64>, LinalgError> {
    let mut output = input.dot(weights);
    output = svd_truncation(&output, rank)?;
    Ok(output)
}

fn main() {
    let input = Array2::from_shape_vec((100, 10), (0..1000).map(|x| x as f64).collect()).unwrap();
    let weights = Array2::from_shape_vec((10, 100), (0..1000).map(|x| x as f64).collect()).unwrap();
    let rank = 5;

    let output = apply_ttsn_layer(&input, &weights, rank).expect("SVD truncation failed");

    println!("Output shape: {:?}", output.shape());
    println!("Output:\n{}", output);
}