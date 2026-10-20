use ndarray::{Array, Array2, ArrayView, ArrayViewMut, s};
use ndarray_linalg::Norm;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rayon::prelude::*;
use std::f64;

// Define a simple neural network transformation
fn neural_network_transform(input: ArrayView<f64, 2>) -> Array<f64, 2> {
    // Example transformation: linear transformation followed by ReLU
    let weights: Array2<f64> = Array::random((input.shape()[1], input.shape()[1]), Uniform::new(-1.0, 1.0));
    let bias: Array1<f64> = Array::random(input.shape()[1], Uniform::new(-1.0, 1.0));
    let output = (input.dot(&weights) + &bias).mapv_into(|x| x.max(0.0));
    output
}

// Define a rotation matrix for isometric transformation
fn rotation_matrix(angle: f64, n: usize) -> Array2<f64> {
    let mut matrix = Array2::eye(n);
    let (c, s) = (angle.cos(), angle.sin());
    matrix[[0, 0]] = c;
    matrix[[0, 1]] = -s;
    matrix[[1, 0]] = s;
    matrix[[1, 1]] = c;
    matrix
}

// Apply isometric transformation to points
fn apply_isometric_transformation(points: ArrayView<f64, 2>, angle: f64) -> Array<f64, 2> {
    let n = points.shape()[1];
    let rotation = rotation_matrix(angle, n);
    points.dot(&rotation)
}

// Calculate pairwise distances
fn pairwise_distances(points: ArrayView<f64, 2>) -> Array<f64, 2> {
    let n = points.shape()[0];
    let mut distances = Array::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            distances[[i, j]] = (points.row(i) - points.row(j)).norm();
        }
    }
    distances
}

// Main function to test isometric invariance
fn main() {
    let n_points = 100;
    let n_dims = 2;
    let angle = std::f64::consts::PI / 4.0; // 45 degrees

    // Generate random points
    let points: Array2<f64> = Array::random((n_points, n_dims), Uniform::new(-1.0, 1.0));
    let transformed_points = apply_isometric_transformation(points.view(), angle);

    // Apply neural network transformation
    let nn_output = neural_network_transform(points.view());
    let nn_output_transformed = neural_network_transform(transformed_points.view());

    // Calculate pairwise distances
    let original_distances = pairwise_distances(points.view());
    let transformed_distances = pairwise_distances(transformed_points.view());
    let nn_distances = pairwise_distances(nn_output.view());
    let nn_transformed_distances = pairwise_distances(nn_output_transformed.view());

    // Check invariance
    let tolerance = 1e-6;
    let invariance = (nn_distances - nn_transformed_distances).mapv_into(|x| x.abs()).max().unwrap();

    if invariance < tolerance {
        println!("The neural network transformation is isometrically invariant within tolerance.");
    } else {
        println!("The neural network transformation is not isometrically invariant. Maximum difference: {}", invariance);
    }
}