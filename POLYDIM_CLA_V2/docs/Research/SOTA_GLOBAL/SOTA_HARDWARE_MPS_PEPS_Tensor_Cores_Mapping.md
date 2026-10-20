# Mapping MPS and PEPS for Probabilistic Models onto Tensor Cores

## 1. Introduction
Matrix Product States (MPS) and Projected Entangled Pair States (PEPS) originated in quantum physics but are now SOTA tools for modeling complex classical probability distributions and partition functions. When simulating these networks, the contraction complexity grows exponentially with the network dimension. Mapping these contractions onto AI accelerators like Tensor Cores is crucial for scaling.

## 2. MPS in 1D Probabilistic Chains
MPS provides an exact or highly accurate representation of 1D stochastic processes (e.g., hidden Markov models with long-range dependencies).
### 2.1 SOTA Optimization: DMRG and TEBD
Time-Evolving Block Decimation (TEBD) and Density Matrix Renormalization Group (DMRG) algorithms are adapted for probability distribution optimization.
- **Hardware Mapping**: TEBD applies local gates (transfer matrices of the path integral) to adjacent sites. On Tensor Cores, applying a gate involves contracting a 2-site tensor with the gate, then doing an SVD. 
- **SVD Bottleneck**: SVD is memory-bound. SOTA approaches use QR decomposition combined with polar decomposition (which relies on GEMM) to bypass SVD, maximizing Tensor Core usage.

## 3. PEPS for 2D Spatial Path Integrals
For 2D interacting stochastic systems (e.g., Ising models, spatial epidemiology), PEPS is required. Exact contraction of PEPS is #P-hard.
### 3.1 Boundary MPS (bMPS) and CTMRG
Corner Transfer Matrix Renormalization Group (CTMRG) and boundary MPS are SOTA heuristic contraction methods for PEPS.
- **Contraction Trees**: The optimal contraction sequence (minimizing FLOPs) often involves intermediate tensors with shapes that are highly asymmetrical (e.g., $1000 \times 10$). 
- **Tensor Core Misalignment**: Tensor Cores prefer squarish matrices whose dimensions are multiples of 16 or 32. 

## 4. SOTA Hardware Optimizations
### 4.1 Padding and Memory Layout (cuTENSOR)
Libraries like NVIDIA's cuTENSOR analyze the contraction graph at runtime. To utilize Tensor Cores effectively, PEPS contraction algorithms dynamically pad bond dimensions to align with memory boundaries (e.g., padding $r=24$ to $r=32$) and use optimized layout transpositions to ensure contiguous memory access during the large GEMMs.
### 4.2 Tensor-Train Splitting for PEPS
A recent SOTA approach to contracting PEPS involves splitting the 2D grid into a sequence of 1D Tensor Trains, optimizing the contraction along one axis using batched GEMMs, and then applying low-rank truncations. This perfectly leverages the hierarchical memory architecture (L1/L2 caches) of Hopper GPUs while keeping Tensor Cores fed.

## 5. Summary
The bottleneck for simulating high-dimensional probability spaces using PEPS/MPS on modern hardware is no longer purely FLOPs, but data movement, memory bandwidth, and the non-GEMM components (like SVD). SOTA algorithms are increasingly replacing exact matrix factorizations with iterative, GEMM-heavy randomized algorithms to fully exploit Tensor Core architectures.
