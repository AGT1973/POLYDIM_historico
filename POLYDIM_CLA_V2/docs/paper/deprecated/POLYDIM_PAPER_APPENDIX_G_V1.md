## Appendix G — MIX Correctness via High-Dimensional Quasi-Orthogonality

Section 3.2 of the paper states that MIX replaces binary branching with continuous
superposition, leveraging "high-dimensional quasi-orthogonality [Kanerva, 2009]."
This appendix makes that claim mathematically precise and provides the formal justification
for why MIX is not an approximation of if/else but its correct generalization.

### G.1 The Classical Claim and Why It Seems Wrong

A reader encountering MIX for the first time may object:

> "α·T₁ + β·T₂ does not 'choose' between T₁ and T₂. It executes a weighted average of both.
> If I want 'if condition then T₁ else T₂', I need a true binary decision, not a blend."

This objection is correct for low-dimensional systems and incorrect for high-dimensional ones.
The difference between the two regimes is the content of this appendix.

The short answer: **in R^N for N ≥ 1000, two randomly chosen transformations T₁ and T₂ act
on nearly orthogonal subspaces of the state vector, so their superposition α·T₁ + β·T₂ does
not destroy either effect. The "choice" occurs at PROJECT, not at MIX.**

The long answer follows.

### G.2 The Johnson-Lindenstrauss Lemma — Core Tool

**Lemma G.1 (Johnson-Lindenstrauss, 1984).** For any ε ∈ (0,1) and any set of n points
in R^d, there exists a projection f: R^d → R^k with k = O(log(n)/ε²) such that for all
pairs u, v:

    (1-ε)‖u-v‖² ≤ ‖f(u)-f(v)‖² ≤ (1+ε)‖u-v‖²

The key consequence for high-dimensional spaces is the **concentration of measure** phenomenon:
random vectors in R^N become nearly orthogonal as N grows. Specifically:

**Corollary G.1 (Quasi-Orthogonality).** Let u, v ∈ R^N be independently and uniformly
drawn from the unit sphere S^{N-1}. Then:

    E[⟨u, v⟩] = 0
    Var[⟨u, v⟩] = 1/N

For N = 10,000: the standard deviation of the inner product is 0.01. With overwhelming
probability, |⟨u, v⟩| < 3/√N = 0.03.

**Two randomly chosen unit vectors in R^{10,000} are almost perfectly orthogonal with
probability > 1 - e^{-N/8} ≈ 1 - 10^{-543}.**

This is not a theoretical curiosity. It is the fundamental reason why high-dimensional
vector arithmetic preserves structure: the "cross-talk" between independently generated
vectors is negligible.

### G.3 Applying Quasi-Orthogonality to MIX

Let T₁, T₂: R^N → R^N be two POLYDIM transformations. In the LoRA representation:

    T₁(v) = U₁V₁ᵀ v     (rank-r transformation in R^N)
    T₂(v) = U₂V₂ᵀ v

The image of T₁ lies in span(U₁) — a subspace of dimension r in R^N.
The image of T₂ lies in span(U₂) — a subspace of dimension r in R^N.

**Proposition G.1 (Subspace Quasi-Orthogonality).** If U₁ and U₂ are independently generated
(e.g., from different training contexts), then the canonical angles θ_i between span(U₁) and
span(U₂) satisfy:

    E[cos²(θ_i)] = r/N   for each i ∈ {1,...,r}

For r = 64, N = 10,000: E[cos²(θ_i)] = 0.0064. The subspaces are nearly orthogonal.

**Consequence:** For any state vector v ∈ R^N:

    MIX(α, T₁, β, T₂)(v) = α·T₁(v) + β·T₂(v)
                           = α·U₁V₁ᵀv  +  β·U₂V₂ᵀv

The two summands live in nearly orthogonal subspaces of R^N. Their interference (measured
by ‖α·T₁(v) + β·T₂(v)‖² - α²‖T₁(v)‖² - β²‖T₂(v)‖²) is bounded by:

    |interference| ≤ 2αβ |⟨T₁(v), T₂(v)⟩| ≤ 2αβ ‖T₁(v)‖ · ‖T₂(v)‖ · (r/N)^{1/2}

For r/N = 0.0064: |interference| ≤ 2αβ · 0.08 · ‖T₁(v)‖ · ‖T₂(v)‖.

This interference term vanishes as N → ∞ and is negligible for practical N/r ratios.
Both transformations coexist in the superposition without destructive interference.

### G.4 The Role of PROJECT as Collapse

If both T₁ and T₂ are present in the superposition simultaneously, how does the system
"choose" between them? It does not choose at MIX time. It chooses at PROJECT time.

When PROJECT(MIX(α,T₁,β,T₂), E) is computed:
1. The functor F: G → E_executor maps the geometric state to the executor's type category
2. F collapses the superposition along the axis dominant in E's subspace
3. The result is the type that E "sees" in the superposed state

**Example:** Let T₁ be a transformation that generates SQL query structure, and T₂ be a
transformation that generates Flutter widget layout. Both are present in MIX(0.7, T₁, 0.3, T₂).

    PROJECT(MIX(0.7,T₁,0.3,T₂), DIM_SQL)     →  SQL query structure (α=0.7 dominates)
    PROJECT(MIX(0.7,T₁,0.3,T₂), DIM_FLUTTER)  →  Flutter widget layout (α=0.3 present)

The system produces BOTH results simultaneously. In a Von Neumann system with if/else,
only one branch executes. In POLYDIM, both transformations are active and each executor
sees its own dominant projection.

This is a qualitatively different computational model, not a quantitative approximation.

### G.5 When MIX Recovers Classical if/else

When α ∈ {0, 1} (binary weights), MIX reduces exactly to if/else:

    MIX(1.0, T₁, 0.0, T₂)(v) = T₁(v)   (classical: condition true → T₁)
    MIX(0.0, T₁, 1.0, T₂)(v) = T₂(v)   (classical: condition false → T₂)

This confirms Remark 3.2 in the paper: POLYDIM subsumes binary branching; it does not
eliminate it. Binary if/else is the degenerate case of MIX where the weights are in {0,1}.

The non-degenerate case (α ∈ (0,1)) is the novel contribution: both transformations
coexist in the state, and the "decision" is deferred to the observer (PROJECT).
This is semantically correct for AI systems that must simultaneously consider multiple
courses of action before committing to one via observation.

### G.6 The Non-Interference Theorem

**Theorem G.2 (Non-Interference of MIX in High Dimensions).**
Let T₁, T₂: R^N → R^N be rank-r transformations with independently generated
column spaces, and let v ∈ R^N be a unit state vector. Then:

    ‖MIX(α,T₁,β,T₂)(v)‖² = α²‖T₁(v)‖² + β²‖T₂(v)‖² + O(r/N)

*Proof sketch.* Expand: ‖α·T₁(v) + β·T₂(v)‖² = α²‖T₁(v)‖² + β²‖T₂(v)‖² + 2αβ⟨T₁(v), T₂(v)⟩.
By Proposition G.1, ⟨T₁(v), T₂(v)⟩ = O(r/N). □

The O(r/N) term is the interference. For r = 64, N = 10,000, it is 0.64%. For r = 256,
N = 10,000, it is 2.56%. For higher N (typical of production language models), it decreases
proportionally.

**Corollary G.2.** In the limit N → ∞ (or equivalently, r/N → 0), MIX becomes an exact
parallel execution with no interference. The superposition is perfectly separable.

### G.7 Connection to Quantum Mechanics — A Pedagogical Caution

The MIX operation is often compared to quantum superposition. The analogy is instructive
but requires care.

**Similarity:** Both involve continuous superpositions of multiple states that coexist
simultaneously, with "collapse" occurring at measurement (PROJECT in POLYDIM; measurement
in QM).

**Critical difference:** Quantum superposition uses complex probability amplitudes with
interference that can be negative (destructive). POLYDIM's MIX uses real-valued weights
α, β ≥ 0 in classical vector spaces. The quasi-orthogonality that makes MIX work is a
property of high-dimensional classical geometry, not quantum mechanics.

The correct analogy is not QM but **multi-objective optimization in high-dimensional spaces**:
when optimization variables are nearly orthogonal, multiple objectives can be optimized
simultaneously without one destroying the other's progress. MIX exploits this same geometry.

POLYDIM is not quantum computing. It is high-dimensional classical computing where
the geometry of the space provides superposition-like properties at no quantum cost.

---
*Appendix G — POLYDIM PAPER V2 — 2026-06-23 — ai.mpat.agt@gmail.com*
