# Optimization Algorithms for Path Integrals using Tensor Trains on Hardware

## 1. Introduction
Solving high-dimensional Fokker-Planck equations or evaluating Feynman-Kac path integrals can be formulated as finding the solution to a high-dimensional PDE. Tensor Trains (TT) provide a robust framework to compress these solutions. Optimizing TT representations of path integrals on modern hardware (especially GPUs with Tensor Cores) requires specialized algorithms that balance mathematical accuracy with hardware efficiency.

## 2. SOTA Tensor Train Optimization Algorithms
### 2.1 TT-Cross Interpolation
The TT-Cross algorithm (based on skeleton decomposition) evaluates the path integral only at a few chosen spatial/temporal grid points.
- **Hardware Mapping**: SOTA implementations batch the evaluation of the function at multi-dimensional grid points to leverage the parallel throughput of Tensor Cores. The core computational kernel involves dense LU/QR decompositions and GEMMs on small matrices (the cores of the TT).
### 2.2 Alternating Linear Scheme (ALS) and AMEn
For solving the linear systems arising from discretized path integrals, ALS and the Alternating Minimal Energy (AMEn) method are SOTA. 
- **Algorithm**: AMEn enriches the basis with residual information to prevent stagnation.
- **Hardware Mapping**: The local optimization step involves solving micro-systems. On Tensor Cores, these operations are formulated as batched tensor contractions. The bottleneck is the sequential nature of sweeping back and forth along the train, which underutilizes large GPUs.

## 3. Adapting Algorithms for Tensor Cores
Tensor Cores perform mixed-precision matrix multiply-accumulate operations (e.g., FP16 input, FP32 accumulate).
### 3.1 Mixed-Precision TT-SVD
Truncating tensor cores after operations (like addition or multiplication of TTs) requires SVD. Standard SVD does not map well to Tensor Cores.
- **SOTA Solution**: Randomized SVD (rSVD) or randomized trace estimation is used. rSVD replaces the deterministic SVD with matrix multiplications against random Gaussian matrices, which maps perfectly to Tensor Cores.
### 3.2 Batched Contractions and Memory Bandwidth
Path integrals often require time evolution (e.g., applying a transfer matrix). In TT format, this is a sequence of tensor contractions. Since TT cores for path integrals typically have ranks $r \in [10, 100]$, the matrices are small.
- **Optimization**: SOTA libraries (like T3F or TensorLy) batch these contractions over the spatial dimensions or multiple paths, transforming memory-bandwidth-bound operations into compute-bound batched GEMMs suitable for Tensor Cores.

## 4. Future Directions
The current frontier is fully asynchronous TT-sweeps and exploiting structured sparsity in the TT cores to push Tensor Core utilization above 70% for stochastic path integral evaluations.
