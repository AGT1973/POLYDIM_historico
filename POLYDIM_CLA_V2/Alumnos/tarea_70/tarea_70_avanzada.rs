use ndarray::{Array, Array2, Array1, arr1, arr2};
use rayon::prelude::*;
use std::f64;

// Define the projection function
fn project_to_manifold(x: &Array1<f64>) -> Array1<f64> {
    // Example: Project to the unit sphere
    let norm = x.norm();
    x / norm
}

// Define the function to check isometric invariance
fn check_isometric_invariance(x: &Array1<f64>, y: &Array1<f64>) -> bool {
    let proj_x = project_to_manifold(x);
    let proj_y = project_to_manifold(y);

    let diff_x = x - proj_x;
    let diff_y = y - proj_y;

    let dot_x = x.dot(&y);
    let dot_proj = proj_x.dot(&proj_y);

    (dot_x - dot_proj).abs() < 1e-6
}

fn main() {
    // Example data
    let x = arr1(&[1.0, 2.0, 3.0]);
    let y = arr1(&[4.0, 5.0, 6.0]);

    // Parallelize the check
    let results: Vec<bool> = (0..10000).into_par_iter().map(|_| {
        let x = Array1::random(x.len(), rand::distributions::Standard);
        let y = Array1::random(y.len(), rand::distributions::Standard);
        check_isometric_invariance(&x, &y)
    }).collect();

    // Print the results
    let success_count = results.iter().filter(|&&r| r).count();
    let total_count = results.len();
    println!("Isometric invariance check: {} out of {} succeeded", success_count, total_count);
}