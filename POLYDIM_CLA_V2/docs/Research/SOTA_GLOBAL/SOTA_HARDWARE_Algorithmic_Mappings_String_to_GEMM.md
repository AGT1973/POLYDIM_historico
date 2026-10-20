# SOTA HARDWARE: Algorithmic Mappings from String Diagrams to Dense GEMMs

## 1. Introduction

Bridging the gap between the abstract compositional structure of Monoidal Categories (represented visually as string diagrams) and the rigid execution semantics of Tensor Cores requires robust algorithmic transformations. The fundamental primitive of Tensor Cores is the block matrix multiplication ($D = A \times B + C$). Therefore, any complex tensor network expressed as a string diagram must ultimately be compiled into a sequence of dense General Matrix Multiplications (GEMMs) and memory permutations.

This document details the State-of-the-Art (SOTA) algorithms used to map arbitrary tensor contractions derived from string diagrams onto highly optimized GEMM calls.

## 2. From String Diagrams to Contraction Trees

A string diagram represents a tensor network $\mathcal{T} = (V, E)$, where vertices $V$ are tensors and edges $E$ are indices (wires). Connected edges represent contracted indices, and open edges represent the uncontracted indices of the final result.

The first step in execution is determining the **contraction path** (or contraction tree). Since tensor contraction is associative (analogous to the associativity of sequential composition in categories), the order of contractions drastically affects computational complexity.

### SOTA Contraction Path Algorithms
Finding the optimal contraction path is NP-hard. SOTA algorithms utilize a mix of heuristics and exact solvers:
1. **Dynamic Programming:** For small networks, exact optimal paths can be found by evaluating all sub-graphs.
2. **Graph Partitioning / Metis:** For large networks (e.g., PEPS in quantum physics), hypergraph partitioning tools are used to minimize the size of intermediate tensors (the "cut size").
3. **Greedy Algorithms:** Iteratively contracting the pair of tensors that minimizes the intermediate size or FLOP count. Libraries like `opt_einsum` or `cuTensorNet` combine greedy search with simulated annealing.

The output is a binary tree where each node represents a pairwise tensor contraction.

## 3. The Pairwise Contraction: Mapping to GEMM

Once a pairwise contraction $C_{i,j,k\dots} = \sum_{p,q\dots} A_{i,p,k,q\dots} B_{j,p,q\dots}$ is isolated from the contraction tree, it must be mapped to a GEMM.

### 3.1. Index Classification
For any pairwise contraction between $A$ and $B$ resulting in $C$, the indices are partitioned into three disjoint sets:
- **Contracted Indices ($K$):** Appear in both $A$ and $B$, but not in $C$.
- **Left Free Indices ($M$):** Appear in $A$ and $C$, but not in $B$.
- **Right Free Indices ($N$):** Appear in $B$ and $C$, but not in $A$.

### 3.2. Tensor Matricization (Reshaping)
To utilize a GEMM ($[M \times K] \times [K \times N] \to [M \times N]$), the tensors must be "matricized" (flattened). 
This requires permuting the dimensions of $A$ and $B$ such that:
- For $A$, the $M$ indices are contiguous and the $K$ indices are contiguous.
- For $B$, the $K$ indices are contiguous and the $N$ indices are contiguous.

In a string diagram, this permutation corresponds to the braiding of wires to align the appropriate ports before applying the composition box.

**The Permutation Penalty:**
Standard GEMM libraries (like cuBLAS) require contiguous memory layouts. If the tensors are not in the correct layout, an out-of-place transposition (e.g., using `cuTensor` or explicit memory copies) is required. This data movement is often bottlenecking. 

### 3.3. Transpose-Free GEMM (TTC / GETT)
SOTA approaches attempt to avoid explicit transpositions by fusing the permutation with the memory load phase of the GEMM. This is known as Transpose-Free Tensor Contraction (TTC) or General Tensor-to-Tensor (GETT) operations.
Instead of loading contiguous memory into the Tensor Core registers, the memory loads are mapped to gather/scatter operations that dynamically fetch the elements according to the permuted strides. 

## 4. Hardware-Aware Mapping of Categorical Composition

String diagrams explicitly encode both sequential ($\circ$) and parallel ($\otimes$) composition. 

### Sequential Composition
Mapped naturally to chained tensor contractions (matrix multiplications). In hardware, SOTA optimization involves **kernel fusion**. If $f$ and $g$ are sequential operations, fusing them avoids writing the intermediate output of $f$ to HBM and reading it back for $g$. Instead, it is kept in Shared Memory or Registers.

### Parallel Composition
$f \otimes g$ represents disjoint operations that can run concurrently. In hardware, this is mapped to:
1. **Batched GEMM (B-GEMM):** If $f$ and $g$ are identical operations on different data.
2. **CUDA Streams:** Launching independent operations asynchronously to saturate the SMs on the GPU.

## 5. Handling Sparsity and Symmetries
Monoidal categories with specialized structures (e.g., symmetric monoidal categories with zero morphisms, or group-equivariant categories) often yield sparse tensors or block-diagonal structures.
SOTA hardware optimizations map these properties to:
- **Block-Sparse GEMMs:** Utilizing Tensor Core features like the 2:4 sparsity pattern in NVIDIA Ampere/Hopper architectures.
- **Block-Diagonal Contractions:** Decomposing the string diagram into independent sub-diagrams based on symmetry sectors (e.g., quantum numbers in physics), mapped to independent, smaller batched GEMMs.
