# POLYDIM: Binary Format Specification (V6.0)
**POPL 2027 Supplementary Material**

The `.polydim` format is the executable binary standard for the POLYDIM language. It is strictly designed to enforce the topological invariants of the Constitution V10.0, utilizing low-rank (LoRA) parametrization and strict metric validation (`algebra_tolerance_epsilon`).

## 1. Memory Efficiency: The LoRA Requirement

By Rule R12, dense $N \times N$ matrices are constitutionally proscribed due to $O(N^2)$ memory explosion for $N \ge 10,000$.
A transformation $T \in \text{Lin}(\mathbb{R}^N)$ is serialized strictly as:
$$ T = W_0 + U \cdot V^T $$
Where $U, V \in \mathbb{R}^{N \times r}$ with $r \ll N$. This reduces the serialized size of a standard transformation from ~400MB to ~2.5MB (a 99.4% reduction).

## 2. Binary Layout

A `.polydim` file consists of three immutable blocks.

### 2.1 Metadata Block (Header)
```c
struct PolydimHeader {
    uint8_t magic[7];          // "POLYDIM"
    uint8_t version;           // 10
    float32_t geo_id[1024];    // 0-skeleton semantic anchor
    float32_t algebra_tolerance_epsilon; // CRITICAL: Metric drift limit
    uint32_t low_rank_r;       // Factor r (e.g., 64)
    uint32_t latent_dim_n;     // Factor N (e.g., 10000)
}
```

### 2.2 Activation Block
```c
struct ActivationState {
    float32_t activations[9]; // Native subspace continuous weights [0,1]
}
```

### 2.3 Transformation Block
```c
struct TransformationBlock {
    float16_t matrix_u[N * r];
    float16_t matrix_v[r * N];
    float16_t weight_w0; // Base baseline
}
```

## 3. The Topological Monitor (Gap 23 Resolution)

The VM loader must intercept any operation where the metric drift during algebraic transport exceeds the safety bound.
If the distance between the transformed semantic position and its origin exceeds `algebra_tolerance_epsilon`, the execution yields a **Constitutional Abort**. This guarantees that the geometric identity (`GEO_ID`) of the execution scope is preserved throughout composition.
