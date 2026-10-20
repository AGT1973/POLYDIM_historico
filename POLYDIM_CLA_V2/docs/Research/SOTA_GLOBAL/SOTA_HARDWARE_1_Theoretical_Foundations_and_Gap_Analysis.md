# SOTA GLOBAL: Theoretical Foundations and Gap Analysis
## Weak Limits, Convergence of Measures, and Iterative Solvers

### 1. Introduction
The intersection of measure theory (specifically weak limits and convergence of measures) and highly optimized hardware-specific iterative solvers represents a frontier in numerical linear algebra. This document explores the theoretical foundations and the existing gap between continuous analytical concepts and discrete, low-precision hardware acceleration like Tensor Cores.

### 2. Weak Limits and Spectral Theory in Iterative Solvers
In the analysis of Krylov subspace methods like GMRES and Conjugate Gradient (CG), the convergence behavior is heavily dictated by the spectral distribution of the matrix operator. 
As the dimension $N \to \infty$, the discrete eigenvalue distribution can be represented as an empirical measure $\mu_N$. The weak convergence of these measures, $\mu_N \rightharpoonup \mu$, provides deep insights into the asymptotic convergence rates of iterative solvers.
- **Potential Theory**: The convergence of CG and GMRES can be modeled using logarithmic potential theory, heavily relying on the weak convergence of spectral measures.
- **Random Matrices**: In large-scale problems, the matrices often exhibit properties of random matrix ensembles. The convergence of empirical spectral distributions (e.g., to the Wigner semicircle or Marchenko-Pastur law) is formulated entirely in the language of weak convergence of probability measures.

### 3. The Theoretical Gap
While the asymptotic behavior of these solvers is well-understood via measure theory, a significant gap emerges when these algorithms are mapped to modern AI accelerators like Nvidia's Tensor Cores:
1. **Infinite vs. Finite Precision**: Weak limits assume exact arithmetic (or well-modeled uniform noise). Tensor Cores operate in highly quantized mixed-precision formats (FP16, BF16, TF32, INT8), introducing non-linear, state-dependent rounding errors that violate traditional measure-theoretic assumptions.
2. **Loss of Orthogonality**: In GMRES and CG, rounding errors lead to a rapid loss of orthogonality in the Krylov basis. The theoretical limit measures do not account for the "ghost" eigenvalues introduced by this loss of orthogonality in low-precision Tensor Cores.
3. **Discrete Hardware Mappings**: Tensor Cores require matrices to be heavily blocked (e.g., 16x16 or 32x8 tiles). The theoretical continuous measures fail to capture the highly structured, block-wise artifacts introduced by memory access patterns and MMA (Matrix-Multiply-Accumulate) instructions.

### 4. Conclusion
To bridge this gap, modern theoretical frameworks must adapt weak convergence to account for stochastic, structured noise models introduced by low-precision Tensor Core hardware.
