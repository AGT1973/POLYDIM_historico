use ndarray::{Array2, array, concatenate, s, Axis};
use ndarray_linalg::svd::SVD;
use rayon::prelude::*;

fn truncate_svd(matrix: &Array2<f64>, k: usize) -> Array2<f64> {
    let (u, s, vt) = matrix.svd(true, true).unwrap();
    let u_k = u.slice(s![.., ..k]);
    let s_k = s.slice(s![..k]);
    let vt_k = vt.slice(s![..k, ..]);
    u_k.dot(&s_k.diag()).dot(&vt_k)
}

fn stabilize_layers(layers: &mut Vec<Array2<f64>>, k: usize) {
    layers.par_iter_mut().for_each(|layer| {
        *layer = truncate_svd(layer, k);
    });
}

fn main() {
    // Ejemplo de capas de una red neuronal
    let mut layers = vec![
        array![[1.0, 2.0], [3.0, 4.0]],
        array![[5.0, 6.0], [7.0, 8.0]],
        array![[9.0, 10.0], [11.0, 12.0]],
    ];

    // Rango truncado
    let k = 1;

    // Estabilización de capas
    stabilize_layers(&mut layers, k);

    // Impresión de capas estabilizadas
    for (i, layer) in layers.iter().enumerate() {
        println!("Capa {}: {}", i, layer);
    }
}