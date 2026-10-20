// core/polydim_fft_bind.rs
// Resolves technical debt: migrate O(N^2) circular convolution to O(N log N) FFT.

use rustfft::{FftPlanner, num_complex::Complex};

/// Performs VSA binding of two real hypervectors via Circular Convolution
/// utilizing FFT (Complexity: O(N log N) instead of O(N^2)).
/// Returns the resulting hypervector normalized in L2.
pub fn vsa_bind_fft(a: &[f32], b: &[f32]) -> Result<Vec<f32>, &'static str> {
    let n = a.len();
    if n != b.len() {
        return Err("Dimension mismatch between vectors for binding");
    }
    
    // 1. Convert real data to complex numbers (imaginary part = 0.0)
    let mut a_complex: Vec<Complex<f32>> = a.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut b_complex: Vec<Complex<f32>> = b.iter().map(|&x| Complex::new(x, 0.0)).collect();
    
    // 2. Initialize the FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    
    // 3. Transform both vectors to the frequency domain: F(a) and F(b)
    fft.process(&mut a_complex);
    fft.process(&mut b_complex);
    
    // 4. Element-wise complex multiplication: F(a) * F(b)
    let mut product: Vec<Complex<f32>> = a_complex.iter().zip(b_complex.iter())
        .map(|(x, y)| x * y)
        .collect();
    
    // 5. Inverse transform back to the time domain: IFFT(F(a) * F(b))
    ifft.process(&mut product);
    
    // 6. Extract the real part (scaled by 1/N due to the IFFT definition in rustfft)
    let n_f32 = n as f32;
    let mut result_vsa: Vec<f32> = product.iter()
        .map(|c| c.re / n_f32)
        .collect();
        
    // 7. L2 normalize the resulting vector to preserve VSA unitary property
    let norm = norm_l2(&result_vsa);
    if norm > 1e-10 {
        for val in result_vsa.iter_mut() {
            *val /= norm;
        }
    }
    
    Ok(result_vsa)
}

/// Direct circular convolution O(N^2) for verification and testing
pub fn vsa_bind_direct(a: &[f32], b: &[f32]) -> Result<Vec<f32>, &'static str> {
    let n = a.len();
    if n != b.len() {
        return Err("Dimension mismatch between vectors for binding");
    }
    
    let mut result = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += a[j] * b[(i + n - j) % n];
        }
        result[i] = sum;
    }
    
    let norm = norm_l2(&result);
    if norm > 1e-10 {
        for val in result.iter_mut() {
            *val /= norm;
        }
    }
    
    Ok(result)
}

/// Helper: Compute L2 norm of the vector
fn norm_l2(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_fft_bind_matches_direct() {
        // Use N=256 to verify exact match between FFT and Direct circular convolution
        let n = 256;
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        
        // Initialize with deterministic values
        for i in 0..n {
            a[i] = (i as f32).sin();
            b[i] = ((i * 2) as f32).cos();
        }
        
        let res_fft = vsa_bind_fft(&a, &b).unwrap();
        let res_direct = vsa_bind_direct(&a, &b).unwrap();
        
        assert_eq!(res_fft.len(), res_direct.len());
        for i in 0..n {
            assert!((res_fft[i] - res_direct[i]).abs() < 1e-5, 
                "FFT and Direct bind mismatch at index {}: fft={} direct={}", 
                i, res_fft[i], res_direct[i]);
        }
    }

    #[test]
    fn test_fft_bind_performance_gain() {
        // Use N=2048 to measure speedup
        let n = 2048;
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        for i in 0..n {
            a[i] = (i as f32).sin();
            b[i] = ((i * 2) as f32).cos();
        }

        // Measure direct circular convolution duration
        let start_direct = Instant::now();
        let _res_direct = vsa_bind_direct(&a, &b).unwrap();
        let duration_direct = start_direct.elapsed();

        // Measure FFT circular convolution duration
        let start_fft = Instant::now();
        let _res_fft = vsa_bind_fft(&a, &b).unwrap();
        let duration_fft = start_fft.elapsed();

        println!("Direct bind (O(N^2)) duration: {:?}", duration_direct);
        println!("FFT bind (O(N log N)) duration: {:?}", duration_fft);
        
        // FFT must be significantly faster for N=2048
        assert!(duration_fft < duration_direct, "FFT bind ({:?}) should be faster than Direct bind ({:?})", duration_fft, duration_direct);
    }
}
