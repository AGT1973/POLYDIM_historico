use ndarray::prelude::*;
use ndarray_parallel::prelude::*;
use rayon::prelude::*;
use std::f64;

fn isometry_check(input: &Array2<f64>, output: &Array2<f64>, transformation: &Array2<f64>) -> bool {
    // Compute the pullback metric
    let pullback_metric = (transformation.t().dot(&output.t())).dot(&transformation);

    // Compute the original metric
    let original_metric = input.t().dot(&input);

    // Check if the metrics are approximately equal
    pullback_metric.mapv(|x| x.abs()) - original_metric.mapv(|x| x.abs()) < 1e-6
}

fn main() {
    // Define the input data as a 2D array
    let input_data = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];

    // Define the transformation matrix as a 2D array
    let transformation_matrix = array![
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5]
    ];

    // Apply the transformation to the input data
    let output_data = transformation_matrix.dot(&input_data);

    // Check if the transformation is isometric
    let is_isometric = isometry_check(&input_data, &output_data, &transformation_matrix);

    if is_isometric {
        println!("The transformation is isometric.");
    } else {
        println!("The transformation is not isometric.");
    }
}