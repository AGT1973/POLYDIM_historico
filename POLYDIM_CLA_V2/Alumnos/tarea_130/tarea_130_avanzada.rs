use ndarray::{Array2, ArrayView2, Ix2};
use ndarray_linalg::SVD;
use rayon::prelude::*;
use ndarray::prelude::*;
use std::sync::Arc;

fn isometric_layer(input: ArrayView2<f64>, k: usize) -> Array2<f64> {
    // Realizar una descomposición QR para obtener una matriz ortogonal
    let (q, _) = input.qr().unwrap();
    // Realizar una descomposición SVD
    let (u, _, vt) = q.svd(true, true).unwrap();
    // Truncar la matriz SVD a los k valores singulares más grandes
    let u_truncated = u.slice(s![.., ..k]);
    let vt_truncated = vt.slice(s![..k, ..]);
    // Reconstruir la matriz truncada
    u_truncated.dot(&vt_truncated)
}

fn main() {
    // Datos de entrada (ejemplo)
    let input_data = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];

    // Número de valores singulares a mantener
    let k = 2;

    // Aplicar la capa isométrica
    let output_data = isometric_layer(input_data.view(), k);

    // Imprimir el resultado
    println!("Output Data:\n{}", output_data);
}