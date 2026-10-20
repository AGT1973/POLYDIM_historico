# SOTA HARDWARE: Categorical Foundations and the Gap to Tensor Core Execution

## 1. Introduction

The modern paradigm of high-performance tensor computations is characterized by two disparate extremes. On one hand, we have the elegant, abstract language of **Monoidal Categories** and **String Diagrams**, which provides a mathematically rigorous framework for expressing tensor networks, quantum circuits, and compositional processes. On the other hand, we have the brutal reality of **Tensor Cores**, specialized hardware accelerators bound by strict memory hierarchies (Registers, Shared Memory, High Bandwidth Memory [HBM]) and rigid execution constraints.

This document explores the foundational gap between these two regimes and formalizes the mapping problem from abstract categorical representations to physical hardware execution.

## 2. Monoidal Categories and String Diagrams

A strict monoidal category $(\mathcal{C}, \otimes, I)$ consists of objects $A, B, C \dots$ and morphisms $f: A \to B$, equipped with a tensor product $\otimes: \mathcal{C} \times \mathcal{C} \to \mathcal{C}$ and a unit object $I$. 
String diagrams provide a 2D topological representation of morphisms in monoidal categories:
- **Wires** represent objects (tensor spaces).
- **Nodes/Boxes** represent morphisms (tensors or operators).
- **Sequential composition** $g \circ f$ is represented by connecting the output wire of $f$ to the input wire of $g$.
- **Parallel composition** $f \otimes g$ is represented by placing wires/boxes side-by-side.

In the context of tensor networks, the category of finite-dimensional vector spaces over a field $\mathbb{F}$ (usually $\mathbb{R}$ or $\mathbb{C}$), denoted $\mathbf{FVect}_{\mathbb{F}}$, serves as the standard semantic universe. Morphisms correspond to linear maps (tensors), and composition corresponds to tensor contraction.

The power of string diagrams lies in their topological invariance: deformations of the diagram (e.g., using snake equations in compact closed categories) correspond to valid algebraic identities (e.g., tensor transpositions or trace cyclic properties). However, this topological freedom assumes infinite, flat memory and cost-free data permutations—an assumption violently broken by actual hardware.

## 3. Tensor Core Memory Hierarchy

Modern GPUs (e.g., NVIDIA Hopper H100, AMD MI300) derive their theoretical peak TFLOPS from specialized Matrix-Multiply-Accumulate (MMA) units, colloquially known as Tensor Cores. These units perform block GEMM operations (e.g., $D = A \times B + C$) on fixed-size tiles (e.g., $16 \times 16$ or $32 \times 8$).

The execution model is tightly coupled to a deep memory hierarchy:
1. **High Bandwidth Memory (HBM):** High capacity (e.g., 80 GB), high latency. Bandwidth is on the order of 3-5 TB/s.
2. **L2 Cache:** Shared across Streaming Multiprocessors (SMs), ~50 MB, ~12 TB/s.
3. **Shared Memory (SRAM):** Software-managed scratchpad memory local to an SM, ~227 KB per SM, extremely high bandwidth but subject to bank conflicts.
4. **Registers:** The fastest memory, bound to individual threads in a warp. Tensor Cores operate directly on data held in registers.

## 4. The Gap: Abstract Topologies vs. Rigid Hierarchies

The translation from a string diagram (representing a complex tensor contraction network) to a sequence of Tensor Core operations reveals a profound "impedance mismatch":

### A. Dimensionality and Tiling
A string diagram may specify the contraction of a tensor of rank 5 with a tensor of rank 4 along 3 shared indices. The abstract operation is dimension-agnostic. Tensor Cores, however, only understand 2D block matrices. The high-dimensional contraction must be "flattened" into a dense General Matrix Multiply (GEMM) by grouping contracted indices into a "k" dimension and uncontracted indices into "m" and "n" dimensions. This requires extensive data transposition.

### B. The Cost of Wiring (Permutations)
In a string diagram, crossing two wires is a trivial swap (the braiding or symmetry morphism). In hardware, permuting dimensions in memory requires reading and writing data across the memory hierarchy, often incurring severe latency and bandwidth penalties if not carefully orchestrated in Shared Memory using memory swizzling techniques.

### C. Resource Bounds and Locality
String diagrams do not encode execution schedules or memory allocations. Hardware requires the workload to be decomposed into tiles that fit within Registers and Shared Memory. The optimization problem becomes one of maximizing data reuse (arithmetic intensity) by keeping intermediate tiles in SRAM or Registers, minimizing trips to HBM.

## 5. Mathematical Formulation of the Mapping Problem

Let $\mathcal{D}$ be a string diagram evaluating to a tensor $T$. The compilation of $\mathcal{D}$ to hardware can be modeled as a functor $F: \mathbf{StringDiag} \to \mathbf{HardwareSched}$. 

The hardware schedule category involves specific morphisms corresponding to memory transfers (HBM $\to$ SRAM $\to$ Reg) and compute operations (MMA instructions).
The cost function to minimize is the execution time, which is predominantly a function of memory traffic. If $Q_m$ is the total bytes moved at memory hierarchy level $m$, and $B_m$ is the bandwidth, the theoretical lower bound is governed by the arithmetic intensity $I = \frac{\text{FLOPs}}{\text{Bytes}}$:
$$ T_{exec} = \max \left( \frac{\text{FLOPs}}{P_{peak}}, \max_m \left( \frac{Q_m}{B_m} \right) \right) $$

To bridge the gap, SOTA compilers must convert the abstract topology of $\mathcal{D}$ into a sequence of flat GEMMs and carefully mapped memory access patterns, a topic explored in the subsequent documents.
