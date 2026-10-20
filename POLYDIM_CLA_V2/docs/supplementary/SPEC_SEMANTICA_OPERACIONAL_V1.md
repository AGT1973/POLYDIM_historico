# POLYDIM: Big-Step Operational Semantics (V1.0)
**POPL 2027 Supplementary Material**

This document specifies the operational semantics of the POLYDIM algebraic core. It integrates the resolution of the *Impedance Mismatch* by defining transformations natively over $R^N$, and resolves the control-flow gap (Gap 15/24) via the OSCAR model and Positional Condition Encoding (PCE).

## 1. State Definition (GeoID and Activations)

A computation state in POLYDIM is not a named variable, but a position $P = (g, S)$:
$$ g \in R^N $$ (Topological invariant, GEO_ID)
$$ S = (V, D, A) $$ (Mutable state)
Where $V \in R^N$ is the latent vector, $D$ is the set of native subspaces, and $A: D \to [0,1]$ are the continuous activation weights.

## 2. Core Primitives (Algebraic Layer)

We write $\langle s, T \rangle \Rightarrow s'$ to denote that state $s$ evaluated under transformation $T$ yields state $s'$.

### 2.1 Sequential Composition (`COMPOSE`)
$$ \frac{\langle s, T_1 \rangle \Rightarrow s_1 \quad \langle s_1, T_2 \rangle \Rightarrow s_2}{\langle s, \text{COMPOSE}(T_1, T_2) \rangle \Rightarrow s_2} $$
**Interpretation:** Morphism composition in the category of geometric transformations. Sequential imperative instructions do not exist.

### 2.2 Superposition and Branching (`MIX`)
$$ \frac{s_1 = T_1(s) \quad s_2 = T_2(s)}{\langle s, \text{MIX}(\alpha, T_1, \beta, T_2) \rangle \Rightarrow \alpha \cdot s_1 + \beta \cdot s_2} $$
**Interpretation:** Replaces boolean `if/else` control flow. Following the OSCAR model and PCE, branching logic is executed in superposition based on quasi-orthogonal activation weights ($\alpha, \beta$).

### 2.3 Recurrence and Loops (`FIXPOINT`)
$$ \frac{\| T(s_n) - s_n \| < \epsilon}{\langle s_0, \text{FIXPOINT}(T, \epsilon) \rangle \Rightarrow s_n} $$
**Interpretation:** Replaces `for/while` loops. Based on Banach's Fixed Point Theorem, guaranteed to converge uniquely if $T$ is a contraction map ($Lip(T) < 1$).

### 2.4 Type Compilation (`PROJECT`)
$$ \frac{v \in R^N \quad \pi_E: R^N \to DIM_E}{\langle (v, D, A), \text{PROJECT}(E) \rangle \Rightarrow \pi_E(v)} $$
**Interpretation:** Data types emerge via orthogonal projection to an executor subspace ($DIM_E$). Proven to act as a proper categorical Functor preserving identity and composition.

## 3. Implementation Layer (Architecture-Specific)

While the algebraic core remains invariant, the implementation layer resolves actual hardware-level executions (e.g., Transformers vs SSMs).

### 3.1 Transformer Attention (`ATTEND`)
$$ \langle (Q, K, V), \text{ATTEND} \rangle \Rightarrow \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V $$
*Note: Under Sargsyan's Theorem, softmax attention cannot satisfy strict monoidal factorization. Hence, ATTEND remains strictly in the implementation layer.*

### 3.2 State Space Recurrence (`RECUR`)
$$ \langle h_t, \text{RECUR}(A, B) \rangle \Rightarrow A \cdot h_t + B \cdot x_t $$
Used for Mamba-3 integration, ensuring bounded $O(1)$ inference.
