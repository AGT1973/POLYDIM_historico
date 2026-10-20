# Bridging the Gap: Using RMT to Optimize Tensor Core Eigensolvers

## 1. Introduction
While Block-Lanczos and Block-Arnoldi methods have been successfully ported to GPU Tensor Cores via mixed-precision GEMMs, controlling their numerical stability remains empirical. Free Probability (FP) and Random Matrix Theory (RMT) offer the theoretical machinery to model, predict, and mitigate the errors introduced by Tensor Cores in these eigensolvers.

## 2. RMT Modeling of Tensor Core Noise in Krylov Subspaces

### 2.1 The Spectrum of the Perturbed Matrix
When Block-Lanczos executes $AV$ on Tensor Cores, the operation is effectively $(A + E)V$, where $E$ represents the quantization and round-off error matrix.
- **RMT Application:** By modeling $E$ as a random matrix from a specific ensemble (e.g., a banded random matrix, given block-wise computation), RMT can predict the perturbation of the Ritz values.
- **Free Probability:** If $A$ and the noise matrix $E$ are asymptotically free, the spectral density of the computed matrix $A+E$ can be derived using free additive convolution. This allows algorithms to computationally filter out "noise-induced" eigenvalues (ghosts) without expensive exact reorthogonalization.

### 2.2 Convergence Prediction via Spectral Density
The convergence rate of Arnoldi and Lanczos heavily depends on the spectral distribution (Density of States) of $A$.
- **SOTA Synergy:** RMT provides closed-form approximations for the Density of States for many classes of structured random matrices. 
- **Application:** A Tensor Core Arnoldi solver can dynamically adjust its restart length and polynomial filters (e.g., Chebyshev acceleration) based on the RMT-predicted spectral bounds, avoiding the need for costly runtime spectral estimation.

## 3. Algorithmic Innovations Driven by RMT

### 3.1 RMT-Informed Preconditioning
For solving $Ax = \lambda x$, if the empirical spectral distribution of $A$ matches an RMT distribution (like Marchenko-Pastur for covariance matrices), we can analytically design preconditioners that optimally cluster the eigenvalues. Tensor Cores can apply these block-preconditioners in FP16, while RMT guarantees that the spectral shift remains stable.

### 3.2 Stochastic Rounding and Spectral Whitening
To prevent the low-precision errors $E$ from exhibiting coherent structure that breaks the "freeness" assumption, SOTA GPU implementations can employ hardware-level Stochastic Rounding. 
- **Result:** Stochastic rounding turns deterministic stagnation into unbiased noise, aligning perfectly with RMT models. This ensures the Krylov subspace explores the true invariant subspace rather than getting trapped by floating-point artifacts.

## 4. Conclusion
The true synthesis of Free Probability and GPU Eigensolvers lies in treating numerical precision errors not as bounds to be feared, but as stochastic perturbations to be modeled. By integrating RMT directly into the Lanczos/Arnoldi restart conditions and orthogonalization triggers, next-generation solvers can fully exploit Tensor Cores safely.
