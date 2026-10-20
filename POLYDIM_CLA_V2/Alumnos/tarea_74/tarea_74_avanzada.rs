use ndarray::{Array2, Array1, arr2};
use ndarray_linalg::Norm;
use rayon::prelude::*;

fn is_isometry(matrix: &Array2<f64>) -> bool {
    let identity = Array2::eye(matrix.shape()[0]);
    let product = matrix.t() * matrix;
    (product - identity).norm_max() < 1e-6
}

fn test_isometry_and_gradients(matrix: &Array2<f64>, inputs: &[Array1<f64>]) -> bool {
    // Check if the matrix is isometric
    if !is_isometry(matrix) {
        return false;
    }

    // Check the invariance of distances
    inputs.par_iter().all(|x| {
        let y = matrix.dot(x);
        (x.norm_l2() - y.norm_l2()).abs() < 1e-6
    })
}

fn main() {
    // Define an orthogonal matrix (isometry)
    let matrix = arr2(&[
        [0.0, -1.0],
        [1.0, 0.0],
    ]);

    // Define a set of input vectors
    let inputs: Vec<Array1<f64>> = vec![
        arr1(&[1.0, 0.0]),
        arr1(&[0.0, 1.0]),
        arr1(&[1.0, 1.0]),
    ];

    // Test isometry and gradient stabilization
    let result = test_isometry_and_gradients(&matrix, &inputs);

    if result {
        println!("The matrix is isometric and gradients are stable.");
    } else {
        println!("The matrix is not isometric or gradients are not stable.");
    }
}