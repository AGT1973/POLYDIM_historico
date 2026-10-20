# SOTA Hardware: GPU Optimizations for Cholesky and Batch-GEMM

## 1. Context
In the context of Gaussian Markov Categories, inference heavily relies on linear algebra operations over covariance and precision matrices. The two most critical operations are:
1.  **Cholesky Factorization ($A = LL^T$):** Essential for computing marginals, sampling, and solving linear systems involving symmetric positive-definite matrices (precision matrices).
2.  **General Matrix Multiply (GEMM):** Core to almost all transformations (e.g., propagating means through linear maps, updating covariances $ABA^T$).

When mapping complex compositional models to GPUs, these operations often manifest as many small, independent operations rather than a single massive operation.

## 2. State-of-the-Art in Batch-GEMM
Standard GEMM is heavily optimized on GPUs (e.g., via cuBLAS and Tensor Cores), but executing thousands of small GEMMs sequentially leads to severe underutilization due to kernel launch overhead and insufficient parallelism within each small matrix.

**SOTA Solutions:**
- **Batched Execution:** APIs like `cublasGemmBatched` allow executing thousands of small GEMMs in a single kernel launch. The GPU hardware scheduler distributes these small multiplications across Streaming Multiprocessors (SMs).
- **Variable Sized Batch-GEMM:** Real-world Gaussian models often produce operations of varying sizes. SOTA libraries (e.g., MAGMA, CUTLASS) provide grouped or variable-sized batched GEMM, where matrices in the batch don't have to be identically sized, minimizing padding overhead.
- **Tensor Core Utilization:** Modern NVIDIA GPUs (Volta architecture and newer) feature Tensor Cores that perform $4 \times 4$ or $16 \times 16$ mixed-precision matrix multiply-accumulate operations in a single clock cycle. SOTA approaches cast covariance updates into half-precision (FP16) or Brain Floating Point (BF16) where numerical stability permits, leveraging Tensor Cores for massive speedups, and refining results in FP32/FP64 if needed.

## 3. State-of-the-Art in GPU Cholesky Factorization
Cholesky factorization is inherently more difficult to parallelize than GEMM due to sequential dependencies (calculating a column depends on the previous columns).

**SOTA Solutions:**
- **Block Cholesky Factorization:** To utilize GPUs effectively, Cholesky is formulated as a block algorithm. Small diagonal blocks are factored (often on the CPU or by a single CUDA block), and the results are used to update the trailing submatrix using highly efficient Level-3 BLAS operations (TRSM, GEMM) on the GPU.
- **Batched Cholesky:** Similar to Batch-GEMM, `cusolverDn<t>potrfBatched` allows factoring many small symmetric positive-definite matrices concurrently. This is crucial for models with many conditionally independent sub-components (e.g., particle filters, parallel Markov chains).
- **Sparse Cholesky on GPUs:** For the large, sparse precision matrices typical in graphical models, libraries like `cuSOLVERSP` and `GLU` offer sparse Cholesky. SOTA techniques involve finding optimal fill-reducing permutations (like Nested Dissection) and mapping the resulting elimination tree to the GPU architecture, processing independent branches of the tree in parallel.

## 4. Hardware Implications for Categorical Models
To leverage these SOTA GPU capabilities, a Gaussian Markov Category compiler must structure its computational graph to:
1.  **Maximize Batching:** Group independent morphisms of the same type and size to utilize Batch-GEMM and Batched-Cholesky.
2.  **Expose Block Structure:** Preserve the inherent block structure of the categorical diagrams so they map cleanly to Block-Cholesky algorithms, avoiding the overhead of treating them as unstructured sparse matrices.
