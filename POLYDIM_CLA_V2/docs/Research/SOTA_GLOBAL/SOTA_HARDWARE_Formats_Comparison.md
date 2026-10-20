# Formats Comparison: CSR vs Block-Sparse vs Dense Tensor Cores for PGM

## 1. Overview
When evaluating algorithms for Probabilistic Graphical Models (PGMs) and D-Separation, the choice of data representation dictates the hardware efficiency. We compare CSR, Block-Sparse, and Dense formats in the context of Tensor Core utilization.

## 2. Compressed Sparse Row (CSR)
- **Structure**: Stores only non-zero elements using value, column index, and row pointer arrays.
- **PGM Fit**: Excellent for representing highly sparse Bayesian Networks (which typically have low average degree).
- **Hardware Utilization**: Extremely poor on Tensor Cores. Requires indirect memory addressing (`val[col_idx[i]]`), leading to uncoalesced memory reads and stalling the high-throughput ALUs.

## 3. Dense Matrix / Tensor Cores
- **Structure**: Full O(N^2) representation of the graph adjacency or joint probability distribution.
- **PGM Fit**: Poor memory efficiency. Most PGMs are designed specifically because the joint distribution is too large to represent densely.
- **Hardware Utilization**: Maximum. Tensor Cores achieve peak TFLOPS processing dense blocks.
- **Verdict**: Only viable for small sub-graphs (e.g., cliques in a Junction Tree algorithm) where dense marginalization can be accelerated via GEMM.

## 4. Block-Sparse Formats (BSR)
- **Structure**: Stores data as a sparse matrix of dense blocks (e.g., 8x8 or 16x16 blocks).
- **PGM Fit**: Moderate. PGMs with strong community structures or modularity can be reordered into block-sparse formats.
- **Hardware Utilization**: High. Tensor Cores can process the individual dense blocks at peak efficiency, skipping the zero blocks entirely.
- **Verdict**: The current SOTA compromise. Block-sparse SpMM (Sparse Matrix-Matrix Multiplication) allows exact inference while harnessing ~60-80% of Tensor Core peak performance.

## 5. Conclusion and Recommendations
For generic D-Separation queries on large, unstructured DAGs, **CSR** remains the standard on CPUs, but limits GPU acceleration. To unlock **Tensor Cores**, the graph MUST be transformed into a **Block-Sparse** format via node clustering. **Dense** formats should be strictly reserved for dense cliques within moralized graphs or when performing continuous relaxations.
