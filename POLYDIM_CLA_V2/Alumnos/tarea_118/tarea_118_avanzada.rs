use ndarray::{Array2, Array3, s};
use ndarray_linalg::SVD;
use rayon::prelude::*;
use std::sync::Arc;

fn truncate_svd<T: ndarray_linalg::Scalar + Sync + Send>(tensor: &Array3<T>, rank: usize) -> Array3<T> {
    let (u, s, vt) = tensor.svd(false, false).unwrap();
    let u_truncated = u.slice(s![.., ..rank]);
    let s_truncated = s.slice(s![..rank]);
    let vt_truncated = vt.slice(s![..rank, ..]);

    u_truncated.dot(&s_truncated.dot(&vt_truncated))
}

fn isometric_transformation(tensor: &Array3<f64>, data: &Array2<f64>) -> Array2<f64> {
    let transformed = tensor.dot(&data);
    transformed
}

fn main() {
    // Parámetros
    let input_dim = 100;
    let output_dim = 100;
    let rank = 10;
    let num_samples = 1000;

    // Generar datos de prueba
    let data: Array2<f64> = Array2::random((input_dim, num_samples), rand::distributions::Standard);
    let mut tensor: Array3<f64> = Array3::random((output_dim, input_dim, rank), rand::distributions::Standard);

    // Truncar SVD periódicamente
    let truncated_tensor = truncate_svd(&tensor, rank);

    // Aplicar transformación isométrica
    let transformed_data = isometric_transformation(&truncated_tensor, &data);

    // Validar invariancia isométrica
    let original_distances = (0..num_samples)
        .into_par_iter()
        .flat_map(|i| (0..num_samples).map(move |j| (data.column(i) - data.column(j)).norm()))
        .collect::<Vec<f64>>();

    let transformed_distances = (0..num_samples)
        .into_par_iter()
        .flat_map(|i| (0..num_samples).map(move |j| (transformed_data.column(i) - transformed_data.column(j)).norm()))
        .collect::<Vec<f64>>();

    // Comparar distancias
    let distance_diffs = original_distances
        .iter()
        .zip(transformed_distances.iter())
        .map(|(a, b)| (a - b).abs())
        .collect::<Vec<f64>>();

    let max_diff = distance_diffs.iter().fold(0.0, |a, &b| a.max(b));
    println!("Max difference in distances: {}", max_diff);

    // Si max_diff es pequeño, la transformación es isométrica
    assert!(max_diff < 1e-6, "The transformation is not isometric.");
}