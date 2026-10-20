use ndarray::{Array, Array2, ArrayD, Ix2};
use nalgebra::{DMatrix, DVector, Isometry3, Rotation3};

fn main() {
    // Definir un tensor de ejemplo
    let tensor: Array2<f64> = Array::from_shape_fn((3, 3), |(i, j)| (i * 3 + j) as f64);
    println!("Tensor original:\n{}", tensor);

    // Definir una transformación isométrica (rotación en 3D)
    let rotation: Rotation3<f64> = Rotation3::from_axis_angle(&nalgebra::Vector3::z_axis(), std::f64::consts::FRAC_PI_4);
    let isometry: Isometry3<f64> = Isometry3::new(nalgebra::Vector3::new(0.0, 0.0, 0.0), rotation);

    // Aplicar la transformación isométrica al tensor
    let transformed_tensor = apply_isometric_transform(&tensor, &isometry);
    println!("Tensor transformado:\n{}", transformed_tensor);

    // Verificar la invariancia isométrica
    let norm_original = tensor.mapv(|x| x.powi(2)).sum().sqrt();
    let norm_transformed = transformed_tensor.mapv(|x| x.powi(2)).sum().sqrt();
    println!("Norma del tensor original: {}", norm_original);
    println!("Norma del tensor transformado: {}", norm_transformed);

    assert!((norm_original - norm_transformed).abs() < 1e-6, "La norma no se ha preservado");
}

fn apply_isometric_transform(tensor: &Array2<f64>, isometry: &Isometry3<f64>) -> Array2<f64> {
    let mut transformed_tensor = Array2::zeros(tensor.dim());
    for i in 0..tensor.shape()[0] {
        for j in 0..tensor.shape()[1] {
            let point = nalgebra::Point3::new(i as f64, j as f64, 0.0);
            let transformed_point = isometry * point;
            let transformed_value = tensor[[i, j]];
            transformed_tensor[[transformed_point.coords[0] as usize, transformed_point.coords[1] as usize]] = transformed_value;
        }
    }
    transformed_tensor
}