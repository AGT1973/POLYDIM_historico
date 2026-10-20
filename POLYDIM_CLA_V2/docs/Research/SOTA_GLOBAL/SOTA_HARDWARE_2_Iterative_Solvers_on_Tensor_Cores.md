# SOTA GLOBAL: Iterative Solvers Mapped to Tensor Cores
## GMRES, Conjugate Gradient, and Mixed-Precision Acceleration

### 1. Introduction
Tensor Cores, originally designed for deep learning, offer massive throughput for dense matrix-multiply-accumulate (MMA) operations. Exploiting this computational power for iterative solvers like Conjugate Gradient (CG) and Generalized Minimal Residual (GMRES) requires sophisticated mixed-precision strategies.

### 2. Mapping to Tensor Cores
The primary challenge in mapping iterative solvers to Tensor Cores is that these units only support low-precision arithmetic (FP16, BF16) for inputs, accumulating in FP32.
- **Preconditioning**: The most successful mapping strategy involves decoupling the solver into a low-precision preconditioner and a high-precision outer solver. 
- **Block-Jacobi and ILU**: Blocked preconditioners are naturally suited for Tensor Cores. The blocks can be densely packed and inverted using batched Tensor Core operations.
- **Mixed-Precision Iterative Refinement (IR)**: SOTA algorithms use Tensor Cores to solve the error equation $A e = r$ in low precision (FP16), while the residual $r = b - Ax$ and the solution $x = x + e$ are updated in high precision (FP64 or FP32).

### 3. State-of-the-Art (SOTA) Algorithms
- **Tensor-Core GMRES**: The SOTA approach uses GMRES based on mixed-precision IR (GMRES-IR). The inner GMRES loop runs entirely on Tensor Cores (using FP16 or TF32), while the outer loop maintains accuracy. Orthogonalization (e.g., Modified Gram-Schmidt or Householder) must be carefully restructured into dense matrix multiplications to utilize the MMA pipeline.
- **Tensor-Core Conjugate Gradient**: CG is notoriously sensitive to round-off errors, leading to the loss of A-orthogonality. SOTA implementations use a pipeline-CG approach, where multiple inner products are fused and mapped to Tensor Cores to reduce memory bandwidth bottlenecks, often combined with mixed-precision preconditioning.

### 4. Hardware Bottlenecks and Optimizations
- **Memory Bandwidth vs. Compute**: Iterative solvers (especially sparse ones) are typically memory-bound (SpMV). Tensor Cores are compute-bound. SOTA research focuses on **s-step** or **communication-avoiding** Krylov methods that increase the compute-to-memory ratio, allowing the dense operations to be offloaded to Tensor Cores.
- **Data Layouts**: Matrices must be transformed into highly specific layouts (e.g., NHWC or block-cyclic) to align with Tensor Core memory access patterns (ldmatrix instructions).

### 5. Summary
Tensor Cores provide massive acceleration for iterative solvers, provided the algorithms are heavily modified to rely on block-dense mixed-precision operations, using frameworks like MAGMA and cuSOLVER.
