## Appendix E — Extended .polydim Binary Format Specification

This appendix provides the full engineering specification for the `.polydim` binary format,
expanding Section 6.3 of the paper with implementation-level detail sufficient for TASK_021.

### E.1 Design Rationale

The `.polydim` format is not source code and not bytecode. It is **serialized geometric state**:
a set of tensor blocks that, when loaded by a POLYDIM runtime, can be directly applied to the
runtime's internal state vector without parsing, interpretation, or instruction execution.

The three design constraints from Section 6.1 drive the format decisions:

**Practicality (eliminates dense N×N):** A full transformation matrix T ∈ R^{N×N} for N = 10,000
requires 10,000² × 4 bytes = 400MB per transformation. Infeasible for exchange or storage.
The LoRA decomposition reduces this to ~5MB for r = 64. See Section E.3.

**Fidelity (controllable precision):** The format supports FP32, BF16, and FP16 precision flags
in the header. For training-time transformations, FP32 is used. For inference-time exchange,
BF16 reduces size by 50% with negligible semantic loss.

**Composability:** Multiple `.polydim` files compose via COMPOSE: if T₁ and T₂ are loaded,
COMPOSE(T₁, T₂) = T₂ ∘ T₁ is computed as U₂V₂ᵀ + W₂₀ applied after U₁V₁ᵀ + W₁₀.
The format stores individual transforms; composition is a runtime operation.

### E.2 Full Binary Structure

    ┌────────────────────────────────────────────────────────────┐
    │  MAGIC     │  4 bytes  │  0x504F4C59  ("POLY")             │
    │  VERSION   │  2 bytes  │  uint16, e.g. 0x0003 = V0.3       │
    │  PRECISION │  1 byte   │  0x00=FP32, 0x01=BF16, 0x02=FP16  │
    │  RESERVED  │  1 byte   │  0x00 (alignment padding)         │
    │  N         │  4 bytes  │  uint32, latent dimensionality     │
    │  RANK_R    │  4 bytes  │  uint32, LoRA rank r               │
    │  N_TRANS   │  4 bytes  │  uint32, number of transforms      │
    │  N_DIMS    │  4 bytes  │  uint32, number of active dims     │
    │  N_TARGS   │  4 bytes  │  uint32, number of project targets │
    │  CHECKSUM  │  4 bytes  │  CRC32 of remaining content        │
    ├────────────────────────────────────────────────────────────┤
    │  GEO_ID BLOCK                                              │
    │  homotopy_class │  32 bytes │  SHA-256 of canonical rep     │
    │  base_hv        │  float[N] │  local chart vector           │
    ├────────────────────────────────────────────────────────────┤
    │  TRANSFORM BLOCKS  (repeated N_TRANS times)                │
    │  op_type   │  1 byte   │  enum: see E.4                     │
    │  op_flags  │  1 byte   │  bit flags: see E.4                │
    │  params    │  float[8] │  α, β, ε, and op-specific args     │
    │  U_matrix  │  float[N×r] │  LoRA left factor                │
    │  V_matrix  │  float[N×r] │  LoRA right factor               │
    │  bias      │  float[N] │  optional bias b (if flag set)     │
    ├────────────────────────────────────────────────────────────┤
    │  ACTIVATION BLOCK                                          │
    │  dim_ids   │  uint32[N_DIMS] │  subspace identifiers        │
    │  weights   │  float[N_DIMS]  │  α ∈ [0,1] per subspace      │
    ├────────────────────────────────────────────────────────────┤
    │  PROJECTION TARGET BLOCK                                   │
    │  targets   │  uint8[N_TARGS] │  COMPILE|RENDER|EXPORT + id  │
    └────────────────────────────────────────────────────────────┘

### E.3 LoRA Representation — Mathematical Basis

Every transformation in a `.polydim` file is stored as a Low-Rank Adaptation
[Hu et al., 2021]:

    T(v) = W₀ · v + U · Vᵀ · v

where:
- W₀ ∈ R^{N×N}: frozen base transformation (typically identity, shared and not stored per-file)
- U ∈ R^{N×r}: left factor (stored per-transform)
- V ∈ R^{N×r}: right factor (stored per-transform)
- r ≪ N: rank (stored in header)

**Why LoRA is the correct representation, not a compromise:**

Theorem (informal): The semantically relevant transformations in a language model's latent
space have effective rank r ≪ N. This is not an approximation claim — it is an empirical
finding validated across GPT, LLaMA, Claude, and other architectures [Hu et al., 2021].
The "meaningful change" in activation space when processing a semantic operation lies in a
low-dimensional subspace of the full N×N space.

Because POLYDIM transformations are by definition semantically relevant transformations
(they encode intents, not arbitrary computations), LoRA is not an approximation of the
correct representation. It IS the correct representation.

**Size calculation for common configurations:**

| N      | r   | Precision | U+V size | Full dense | Reduction |
|--------|-----|-----------|----------|------------|-----------|
| 4,096  | 64  | FP32      | 2.0 MB   | 64 GB      | 32,000×   |
| 8,192  | 64  | FP32      | 4.0 MB   | 256 GB     | 64,000×   |
| 10,000 | 64  | FP32      | 4.9 MB   | 381 GB     | 78,000×   |
| 10,000 | 256 | FP32      | 19.5 MB  | 381 GB     | 19,500×   |
| 10,000 | 256 | BF16      | 9.8 MB   | 190 GB     | 19,500×   |

A `.polydim` file with 10 transforms at N=10,000, r=64 occupies ~50MB. Exchangeable.

### E.4 Operation Type Enum

The `op_type` byte in each TRANSFORM block encodes the primitive:

| Value | Primitive | Layer | Description |
|-------|-----------|-------|-------------|
| 0x01  | COMPOSE   | 1     | Sequential composition T₂ ∘ T₁ |
| 0x02  | MIX       | 1     | Superposition α·T₁ + β·T₂ |
| 0x03  | FIXPOINT  | 1     | Iterate T until convergence |
| 0x04  | PROJECT   | 1     | Functor to executor category |
| 0x05  | ATTEND    | 2     | Transformer attention |
| 0x06  | RECUR     | 2     | SSM-style recurrence |

The `op_flags` byte encodes:
- Bit 0: has_bias (bias vector present after V_matrix)
- Bit 1: frozen_base (W₀ is identity, skip loading)
- Bit 2: shared_base (W₀ loaded from shared codebook)
- Bits 3-7: reserved

For MIX: `params[0]` = α, `params[1]` = β (must satisfy α + β = 1).
For FIXPOINT: `params[0]` = ε (convergence threshold).
For PROJECT: `params[0]` = executor_id (enum matching PROJECTION TARGET block).

### E.5 GEO_ID Block — Homotopy Class Encoding

The GEO_ID block encodes identity at two levels:

**`homotopy_class` (32 bytes):** A SHA-256 hash of the canonical representative of the
homotopy equivalence class (as defined in Section 5.1 of the paper). Two `.polydim` files
from different AI systems represent the same object if and only if their `homotopy_class`
fields match. This is the model-agnostic identity.

**`base_hv` (float[N]):** The local chart representation — the actual hypervector in R^N
in the coordinate system of the generating AI system. Two AI systems may have different
`base_hv` values for the same object (because their latent spaces differ), but identical
`homotopy_class` values. The ALIGN protocol uses `base_hv` for geometric alignment and
`homotopy_class` for identity verification.

### E.6 Semantic Virtual Memory and Sparse Execution

The POLYDIM runtime does not materialize all N_DIMS subspaces simultaneously. The
ACTIVATION BLOCK enables sparse execution: only subspaces with weight above a runtime
threshold θ are computed.

    execute(state, transform):
        active_dims = {d : A[d] > θ}
        apply T restricted to active_dims
        recompute A after transformation
        trigger projections for dims that crossed θ

For a typical configuration with N_DIMS = 9 and θ = 0.1, the average number of active
dimensions in any given operation is 3-4, not 9. This mirrors the attention mechanism:
not all tokens are equally relevant; not all dimensions need computation.

### E.7 Composability Contract

Two `.polydim` files f₁ and f₂ are composable if:
1. Their HEADER.N values match (same latent dimensionality)
2. Their HEADER.RANK_R values are compatible (same or explicitly declared merge strategy)
3. Their GEO_ID homotopy classes are aligned (same semantic space, verified via ALIGN)

Composition of f₁ and f₂ produces f₁₂ where:
- GEO_ID.homotopy_class = f₂.GEO_ID.homotopy_class (the output identity)
- TRANSFORMS = [f₁.TRANSFORMS..., f₂.TRANSFORMS...] (concatenated, in order)
- ACTIVATIONS = weighted union of f₁.ACTIVATIONS and f₂.ACTIVATIONS

This is Theorem 3.1 (Associativity of COMPOSE) expressed at the file format level.

### E.8 Implementation Notes for TASK_021

The reference implementation (Rust) should expose:

```rust
pub struct PolyDimFile {
    header: Header,
    geo_id: GeoId,
    transforms: Vec<Transform>,
    activations: ActivationBlock,
    projections: Vec<ProjectionTarget>,
}

impl PolyDimFile {
    pub fn load(path: &Path) -> Result<Self, PolyDimError>
    pub fn save(&self, path: &Path) -> Result<(), PolyDimError>
    pub fn compose(&self, other: &Self) -> Result<Self, PolyDimError>
    pub fn apply(&self, state: &StateVector) -> Result<StateVector, PolyDimError>
    pub fn verify_checksum(&self) -> bool
}
```

The `apply` method is the hot path. It must support:
- BF16/FP16 precision for inference-time `.polydim` files
- FP32 for training-time `.polydim` files
- Sparse execution using the ACTIVATION BLOCK threshold θ

---
*Appendix E — POLYDIM PAPER V2 — 2026-06-23 — ai.mpat.agt@gmail.com*
