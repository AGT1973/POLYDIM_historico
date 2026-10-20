# SOTA Hardware: Theoretical Mapping from Gaussian Markov Categories to GPUs

## 1. Introduction
Gaussian Markov Categories (GMCs) provide a rigorous, compositional, and abstract framework for reasoning about Gaussian probability distributions, linear systems, and signal processing. They allow complex probabilistic models to be constructed by wiring together simpler morphisms (string diagrams). However, a significant gap exists between this high-level categorical abstraction and the low-level execution models of modern hardware accelerators, particularly GPUs.

## 2. The Abstraction Gap
### 2.1 The Categorical Perspective
In GMCs, computation is represented as composition (sequential) and monoidal products (parallel) of linear stochastic maps. Inference in these models (such as exact marginalization or conditioning) corresponds to algebraic operations on the parameters of these maps (mean vectors and covariance/precision matrices). The structures are inherently block-sparse and often exhibit complex dependencies described by directed or undirected graphs (e.g., Markov Random Fields, Bayesian Networks).

### 2.2 The Hardware Perspective (GPUs)
GPUs are throughput-oriented architectures designed for massive data parallelism. They excel at dense, regular memory access patterns and dense linear algebra operations (e.g., General Matrix Multiply - GEMM). They feature specialized hardware like Tensor Cores that achieve peak performance on large, dense matrix multiplications at mixed precisions.

### 2.3 The Gap
The gap lies in the structural mismatch:
- **Sparsity vs. Density:** GMCs naturally yield block-sparse, highly structured precision matrices. GPUs struggle with irregular sparsity, requiring padding or complex sparse formats (CSR/BSR) that often underutilize compute units compared to dense GEMM.
- **Compositionality vs. Monolithic Kernels:** The compositional nature of GMCs implies many small, sequentially dependent operations (e.g., message passing steps). GPUs suffer from high kernel launch overheads and poor utilization when executing many small operations. They require operations to be fused into monolithic, large-scale kernels.
- **Dynamic vs. Static Shapes:** While theoretical GMCs don't dictate shape, practical implementations often involve dynamically sized state spaces depending on the observation stream. GPUs require static, predictable shapes for optimal memory allocation and loop unrolling.

## 3. Current SOTA in Bridging the Gap
To execute GMC-based inference efficiently on GPUs, researchers are exploring **Categorical Compilers**. These compilers take a string diagram representing a Gaussian model and perform:
1.  **Flattening:** Converting deep compositions into shallower, wider structures to expose parallelism.
2.  **Graph Transformation:** Applying string diagram rewrites (using the axioms of Markov categories) to find algebraically equivalent structures that are more hardware-friendly (e.g., minimizing maximum clique size in the corresponding graphical model).
3.  **Operation Fusing:** Grouping small matrix operations into larger block-diagonal operations that can be mapped to Batch-GEMM.

## 4. Conclusion
The SOTA mapping from GMCs to GPUs requires intermediate representation (IR) frameworks that can reason about both categorical equivalence (to optimize the algorithm) and hardware constraints (to optimize execution). The ultimate goal is an automated pipeline: `String Diagram -> Categorical Rewrite -> Block-Sparse IR -> Batch-GEMM/Cholesky GPU Kernels`.
