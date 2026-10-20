use ndarray::{Array, Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::Norm;
use rayon::prelude::*;

fn main() {
    // Definir una variedad diferenciable representada por una matriz de características
    let manifold: Array2<f64> = Array::from_shape_fn((100, 10), |(i, j)| {
        (i as f64 + j as f64) / 10.0
    });

    // Definir una transformación lineal (matriz de pesos)
    let weights: Array2<f64> = Array::from_shape_fn((10, 10), |(i, j)| {
        if i == j { 1.0 } else { 0.0 }
    });

    // Aplicar la transformación lineal
    let transformed_manifold = manifold.dot(&weights);

    // Validar la invariancia isométrica
    let invariance_isometric = manifold
        .outer_iter()
        .zip(transformed_manifold.outer_iter())
        .all(|(original, transformed)| {
            original.norm_l2() == transformed.norm_l2()
        });

    if invariance_isometric {
        println!("La transformación es isométrica.");
    } else {
        println!("La transformación no es isométrica.");
    }
}