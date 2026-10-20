# SOTA Optimization Algorithms for Lanczos and Arnoldi Solvers on GPU Tensor Cores

## 1. Introduction
Krylov subspace methods, specifically the Lanczos (for symmetric/Hermitian) and Arnoldi (for non-Hermitian) algorithms, are fundamental for computing eigenvalues of large sparse or structured matrices. Adapting these to GPU Tensor Cores requires overcoming the mismatch between Krylov's vector-centric nature and Tensor Cores' matrix-matrix (GEMM) specialization.

## 2. Algorithmic Bottlenecks on Modern GPUs
- **Memory Bound vs. Compute Bound:** Standard Lanczos/Arnoldi relies heavily on Matrix-Vector Multiplication (SpMV or GEMV) and Level-1 BLAS (dot products, axpy). These are severely memory-bandwidth bound and cannot utilize the massive compute capabilities of Tensor Cores designed for Level-3 BLAS (GEMM).
- **Loss of Orthogonality:** Low-precision operations accelerate the loss of orthogonality among Lanczos vectors, leading to spurious eigenvalues (ghosts) and stalling convergence.

## 3. SOTA Optimization Strategies

### 3.1 Block Krylov Methods
To exploit Tensor Cores, algorithms must be reformulated into Block-Lanczos or Block-Arnoldi methods.
- **Mechanism:** Instead of multiplying the matrix $A$ by a single vector $v$, $A$ is multiplied by a block of vectors $V$ (e.g., 16 or 32 vectors simultaneously). 
- **Tensor Core Utilization:** This converts the SpMV/GEMV bottleneck into an SpMM/GEMM operation, which maps directly onto Tensor Core WMMA (Warp Matrix Multiply Accumulate) instructions.

### 3.2 Mixed-Precision and Iterative Refinement
SOTA implementations utilize mixed-precision to balance speed and accuracy:
- **Low-Precision SpMM:** The core block matrix multiplication ($A \times V$) is cast to FP16 or TF32 and executed on Tensor Cores.
- **High-Precision Orthogonalization:** The Gram-Schmidt orthogonalization (or QR factorization) of the Krylov basis is performed in FP64 to maintain numerical stability and delay the loss of orthogonality.
- **Hardware-Aware Reorthogonalization:** Selective reorthogonalization (e.g., using the Paige bound) is implemented on the GPU, often fusing dot products to reduce kernel launch overhead.

### 3.3 Communication-Avoiding (s-step) Lanczos/Arnoldi
- **Matrix Powers Kernel:** Computing $[V, AV, A^2V, \dots, A^sV]$ in a single step to minimize global memory round-trips.
- **SOTA Implementations:** Recent algorithms restructure the $s$-step polynomial basis generation to use block Tensor Core operations, dramatically improving arithmetic intensity, though conditioning of the Krylov basis remains a challenge.

## 4. Conclusion
The SOTA for eigensolvers on GPUs is dominated by Block, Mixed-Precision, and Communication-Avoiding variants. The primary challenge remains the severe condition number degradation induced by low-precision Tensor Core GEMMs during basis generation.
