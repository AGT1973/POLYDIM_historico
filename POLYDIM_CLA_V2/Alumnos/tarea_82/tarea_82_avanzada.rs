use ndarray::Array2;
use rayon::prelude::*;
use std::f64;

fn is_isometric(A: &Array2<f64>, T: &Array2<f64>) -> bool {
    let n = A.shape()[0];
    (0..n).into_par_iter().all(|i| {
        (0..n).into_par_iter().all(|j| {
            shortest_path(A, i, j) == shortest_path(&apply_transformation(A, T), i, j)
        })
    })
}

fn shortest_path(A: &Array2<f64>, u: usize, v: usize) -> f64 {
    let n = A.shape()[0];
    let mut dist = A.to_owned();
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if dist[(i, j)] > dist[(i, k)] + dist[(k, j)] {
                    dist[(i, j)] = dist[(i, k)] + dist[(k, j)];
                }
            }
        }
    }
    dist[(u, v)]
}

fn apply_transformation(A: &Array2<f64>, T: &Array2<f64>) -> Array2<f64> {
    let n = A.shape()[0];
    let mut B = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            B[(i, j)] = T.dot(&A.slice(s![i, ..])).dot(&T.slice(s![.., j]));
        }
    }
    B
}

fn regularize_gradients(A: &mut Array2<f64>, lambda: f64) {
    let n = A.shape()[0];
    for i in 0..n {
        for j in 0..n {
            A[(i, j)] *= (1.0 - lambda * A[(i, j)].abs());
        }
    }
}

fn main() {
    let A = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]).unwrap();
    let T = Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    let lambda = 0.1;

    println!("Is the graph isometric under transformation? {}", is_isometric(&A, &T));

    let mut A_reg = A.clone();
    regularize_gradients(&mut A_reg, lambda);
    println!("Regularized adjacency matrix:\n{}", A_reg);
}