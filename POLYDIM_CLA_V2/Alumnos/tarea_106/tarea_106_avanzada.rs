use ndarray::prelude::*;
use ndarray_linalg::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

// Definir una red neuronal simple con capas lineales
struct NeuralNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
}

impl NeuralNetwork {
    fn new(layer_sizes: &[usize]) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let w = Array2::random((layer_sizes[i + 1], layer_sizes[i]), rand_distr::Standard);
            let b = Array2::zeros((layer_sizes[i + 1], 1));
            weights.push(w);
            biases.push(b);
        }

        NeuralNetwork { weights, biases }
    }

    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut output = input.clone();
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            output = w.dot(&output) + b;
        }
        output
    }

    fn backward(&self, input: &Array2<f64>, output: &Array2<f64>, target: &Array2<f64>) -> Vec<Array2<f64>> {
        let mut gradients = Vec::new();
        let mut delta = output - target;
        let mut gradient = Array2::zeros((output.shape()[0], input.shape()[1]));

        for (i, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate().rev() {
            let gradient_w = delta.dot(&input.t());
            let gradient_b = delta.sum_axis(Axis(1)).insert_axis(Axis(1));
            gradients.push(gradient_w.clone());
            gradients.push(gradient_b.clone());

            if i > 0 {
                delta = w.t().dot(&delta);
            }
        }

        gradients.reverse();
        gradients
    }

    fn norm_gradients(&self, gradients: &[Array2<f64>]) -> f64 {
        gradients.iter().map(|g| g.norm()).sum()
    }
}

fn main() {
    let layer_sizes = [10, 20, 30, 40, 50];
    let network = NeuralNetwork::new(&layer_sizes);
    let input = Array2::random((10, 1), rand_distr::Standard);
    let target = Array2::random((50, 1), rand_distr::Standard);

    let start = Instant::now();
    let output = network.forward(&input);
    let gradients = network.backward(&input, &output, &target);
    let norm = network.norm_gradients(&gradients);
    let duration = start.elapsed();

    println!("Norm of gradients: {}", norm);
    println!("Time elapsed: {:.2?}", duration);
}