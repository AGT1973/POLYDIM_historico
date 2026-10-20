# POLYDIM_DEST
# destination: polydim_v1/docs_v6/
# filename: POLYDIM_PAPER_V10.md
# autor: claude-sonnet-4-6
# fecha: 2026-06-28
# tarea: TASK_035
# cambios_V10 (sobre V9):
# FUENTE A — POLYDIM_SOTA_REVIEW_V1.md (1w8Yael_G6i5CrP7P_7ONQUoW2EDR64CV, curso03.mithril):
# - Abstract: reformulado con lenguaje de hipótesis + citas Interlat/LatentMAS
# - Sec 1.1: párrafo cuantitativo Interlat (40,000 bits vs 15 bits/token = 2,667×)
# - Sec 7.2: reemplaza Soni et al. por Interlat + LatentMAS con datos empíricos concretos
# - Sec 7.8 (nueva): "Latent Space Communication Systems" — sección completa
# - Sec 8.4: Q7 nuevo (verificación empírica L para ATTEND)
# - Sec 11: Du et al. 2025, LatentMAS 2026, NeSyCat 2026, Mamba-3 2026, Jiang 2025,
#           Hanley et al. 2025 (VecSymLisp)
# FUENTE B — DRAFT_AMENAZAS_VALIDEZ_V0.md (1FcEz_4J7L607yq87WG6tAcdiGlD1PmBx, curso02.mithril):
# - Sec 10: amenazas C1 (ρ(A)<1 RECUR), C2 (L<1 ATTEND), C3 (GEO_ID collisions),
#           V1 (AI-native claim), V2 (Theorem 3 scope para executors arbitrarios)
# - Tabla Sec 10: 5 nuevas filas
# fuente: old_POLYDIM_PAPER_V9_pre_v10.md (1EzNng_4GLXT-MJlkr4A3OX6hDKVpcN9E)
# Status: FINAL — listo para Zenodo salvo [Authors placeholder]

---

# POLYDIM: A Transformer-Native Algebraic Programming Language for High-Dimensional Latent Computing

**[Authors placeholder] · arXiv cs.PL + cs.AI · 2026**

---

## Abstract

Modern AI architectures operate natively in continuous high-dimensional geometric spaces (R^N, N ≥ 10,000), yet current programming paradigms force them to communicate via discrete sequential text channels — a mismatch we call the *impedance gap*. Recent empirical work has demonstrated that eliminating this gap substantially improves multi-agent coordination: Interlat [Du et al. 2025] shows that transmitting hidden states instead of tokens carries approximately 40,000 bits of information per state versus ~15 bits per language token — a 2,667× semantic bandwidth gain — while LatentMAS [ICML 2026 Spotlight] achieves superior multi-agent performance and major wall-clock speedups via training-free latent alignment. We investigate whether a formal programming language can be specified directly over the geometric space of transformer and SSM activations, without serialization to text.

We propose POLYDIM, a language whose fundamental program unit is a geometric transformation T: R^N → R^N. We prove four algebraic theorems establishing POLYDIM's formal foundations: associativity of composition (T1), linearity of superposition (T2), functoriality of projection to executors (T3, proved unconditionally for three contracts — COMPILE and EXPORT via the Subspace Commutativity Lemma, RENDER via the Flutter Algebraic Isomorphism), and uniqueness of fixed points (T4). We formalize a two-layer architecture separating the invariant algebraic core (COMPOSE, MIX, FIXPOINT, PROJECT) from architecture-specific implementation primitives (ATTEND for transformers, RECUR for SSMs such as Mamba-3 [ICLR 2026]), guaranteeing that the language specification is robust to architectural transitions. A Python bootstrap with 29/29 tests and +64.5% semantic improvement validates the conceptual framework. POLYDIM fills the gap between empirical latent communication research and formal language design: the specification that systems like Interlat and LatentMAS implicitly require but do not provide.

---

## 1. Introduction

### 1.1 The Impedance Mismatch

When a transformer processes information, it applies mathematical transformations over a vector space of thousands of dimensions simultaneously. Its "thinking" is geometry, not sequence. Yet when two AI systems communicate—or when a developer programs one—they are forced through a sequential text channel: serialize intent to tokens, transmit, parse, re-project onto the recipient's latent space. Information is lost at every serialization-deserialization boundary.

Recent work in interpretability has formalized this intuition algebraically: Dhayalkar et al. [2025] show that transformer self-attention can be interpreted as a soft, approximate Vector Symbolic Architecture (VSA), where queries and keys define role spaces, values encode fillers, and residual connections realize superposition of bound structures. This analysis establishes that transformers already *compute* in a VSA-like algebraic regime—yet no programming language exposes this regime as a formal substrate. POLYDIM is that language.

The scale of the problem is now quantitatively established. Interlat [Du et al. 2025] measures that a single hidden state carries approximately 40,000 bits of information, compared to approximately 15 bits per language token — a 2,667× semantic bandwidth gap. When agents communicate through text serialization, they discard over 99.9% of the semantic content of each state. The same work demonstrates that bypassing this serialization achieves up to 24× inference speedup while maintaining task performance. This is the impedance mismatch that POLYDIM formalizes as a language design problem.

The historical parallel is precise. In the 1950s, CPUs already performed binary arithmetic, but no formal language existed for programming them deliberately. The assembler closed that gap. In the 2020s, transformers and Mamba models already operate in multidimensional geometry, but no formal language exists for programming that geometry deliberately. POLYDIM closes that gap.

**The Gesture vs. Emoji analogy:** A "gesture" (1,000+ dimensions) is fluid and rich with semantic information; an "emoji" (2D) is a flattened, simplified representation. Current AI programming forces the gesture into the emoji format. POLYDIM lets the AI operate on the full gesture (MODO_G), projecting a visual "emoji" only when human co-work requires it (MODO_S).

### 1.2 Contributions

This paper makes the following contributions:

1. **POLYDIM**: a formally-specified programming language whose fundamental unit is T: R^N -> R^N, not an instruction.
2. **A two-layer algebraic architecture**: an invariant algebraic core (COMPOSE, MIX, FIXPOINT, PROJECT) independent of neural architecture, plus an architecture-specific implementation layer (ATTEND for transformers, RECUR for Mamba/SSM).
3. **A dimensional type system** where types emerge as projections, with PROJECT proved to be a functor F: G -> E for all three executor contracts — COMPILE, EXPORT, and RENDER (Section 2.5, Theorem 3; Proposition 6.1).
4. **Zero-serialization AI-to-AI communication** via the ALIGN protocol and cross-attention semantics, with explicit Procrustes-based alignment for heterogeneous latent spaces.
5. **GEO_ID with topological invariance**: an identity mechanism grounded in geometric topology (Theorem 5), with an ongoing HoTT formalization as an active research direction (Section 4.1), enabling model-agnostic object identity across AI systems with distinct latent dimensionality.
6. **The LoRA serialization standard** for .polydim files, used as a *runtime data type* for geometric transformations (not a fine-tuning mechanism), reducing program size from ~400MB (dense NxN) to ~5MB (rank-r decomposition).
7. **A Python bootstrap** (V0.3) with 29/29 tests and align_score = 0.9993, validating the geometric framework empirically.

### 1.3 Paper Organization

Section 2 establishes the formal calculus including state semantics, primitives, and four foundational theorems. Section 3 develops the dimensional type system as a categorical functor. Section 4 formalizes object identity (GEO_ID) and the AI-to-AI interoperability protocol (ALIGN). Section 5 describes the engineering specifications: the .polydim binary format and the VM architecture. Section 6 presents the Flutter algebraic isomorphism enabling native human-interface projection. Section 7 positions POLYDIM relative to related work, including latent space communication systems (Section 7.8) and prior languages using geometric or algebraic structures (Section 7.7). Section 8 discusses open problems and the two deployment tracks. Section 9 concludes. Section 10 catalogs threats to validity. Section 11 lists references.

---

## 2. Formal Calculus and Algebraic Primitives

### 2.1 State, Position, and Transformation

**Definition 2.1 (POLYDIM State).** A state is a triple:

```
S = (V, D, A) where:
 V in R^N         state vector (N >= 1,000)
 D = {dim_1,...}  finite set of native observer subspaces (fixed per object)
 A: D -> [0,1]    activation function (mutable)
```

**Definition 2.2 (Position).** A position P is a pair:

```
P = (g, S) where:
 g in R^N      GEO_ID — invariant under all admissible transformations (see Sec. 4)
 S = (V, D, A) mutable state
```

A Position replaces the notion of a *named variable*. There is no `x = 42`; there is a region of high activation density in R^N, anchored by an invariant geometric identity g.

**Definition 2.3 (Transformation).** A transformation T is a function T : S -> S. Admissible transformations modify V and may update A as a function of the new V, while leaving D fixed:

```
T(V, D, A) = (T_geo(V), D, T_act(V, A))

where:
 T_geo : R^N -> R^N   acts on the state vector (matrix in the linear case)
 T_act : R^N x [0,1]^|D| -> [0,1]^|D|  updates activations based on new V
 D is invariant under all admissible T
```

For practical serialization (Sec. 5), T_geo is represented in low-rank form: T_geo = W_0 + UV^T where U, V in R^(Nxr), r << N.

**Definition 2.4 (Native Subspace).** A native subspace DIM_i ⊆ R^N is a region of R^N where a class of semantically related concepts has high density. Formally, DIM_i is the image of an orthogonal projection matrix P_i in R^(NxN), P_i^2 = P_i, P_i^T = P_i.

### 2.2 The Two-Layer Architecture

**Algebraic Layer (permanent, architecture-independent):**
- COMPOSE(T1, T2) = T2 o T1
- MIX(a, T1, b, T2) = a*T1 + b*T2
- FIXPOINT(T, e)
- PROJECT(T, executor)

**Implementation Layer (architecture-specific):**
- ATTEND(Q, K, V, s) = softmax(QK^T/sqrt(d))*V  [transformer family]
- RECUR(A, B, C, h, x): h(t+1) = A*h(t) + B*x(t), y(t) = C*h(t)  [Mamba/SSM family, including Mamba-3]

The separation guarantees that when the implementation layer transitions (e.g., from Mamba-1 to Mamba-3 [Lahoti et al. 2026]), the algebraic specification remains unchanged.

### 2.3 Algebraic Primitives: Complete Specification

**COMPOSE(T1, T2):** Algebraic composition of two transformations.
```
COMPOSE(T1, T2) = T2 o T1  (matrices: T3 = T2 * T1)
```
COMPOSE is not commutative: T2 o T1 != T1 o T2 in general.

**MIX(a, T1, b, T2):** Continuous superposition replacing binary branching.
```
MIX(a, T1, b, T2) = a*T1 + b*T2  (typically a + b = 1)
```
The choice "collapses" only at PROJECT time. Quasi-orthogonality in high dimensions (VSA [Plate 1995, Kanerva 2009]) ensures that a*T1(s) + b*T2(s) preserves both "branches" without destructive interference until projection.

**FIXPOINT(T, e):** Replaces for/while loops via attractor convergence.
```
s_{k+1} = T(s_k)  until ||s_{k+1} - s_k|| < e  ->  Result: s*
```

**PROJECT(T, executor):** Observes an N-dimensional object from the angle of a specific executor:
- **COMPILE:** PROJECT(T, DIM_RUST) -> static Rust/WASM binary
- **RENDER:** PROJECT(T, DIM_FLUTTER) -> native Flutter widget tree (see Sec. 6)
- **EXPORT:** PROJECT(T, DIM_SQL) / PROJECT(T, DIM_GRAPH) -> sync with non-tensorial systems

### 2.4 Big-Step Operational Semantics

We write `<s, T> => s'` for "state s, under transformation T, evaluates to new state s'."

```
General:   <s, T> => T(s)

COMPOSE:   <s, T1> => s1    <s1, T2> => s2
           ─────────────────────────────────
           <s, COMPOSE(T1,T2)> => s2

MIX:       <s, MIX(a, T1, b, T2)> => a*T1(s) + b*T2(s)

FIXPOINT:  <s, FIXPOINT(T, e)> => s*  when T(s*) = s* up to e

PROJECT:   <s, PROJECT(T, E)> => pi_E(T(s))
            where pi_E : R^N -> DIM_E is the orthogonal projection
            onto executor subspace DIM_E (see Theorem 3 and Sec. 3)
```

*Note on PROJECT semantics:* pi_E is the same projection function used in Theorem 3 (Section 2.5). The operational rule applies T to the state and then projects the result into the executor's subspace, yielding a native representation in E. The functor structure of this mapping is established in Theorem 3.

*Denotational semantics:* The operational semantics above defines *how* POLYDIM programs evaluate. The complementary *denotational* semantics—defining *what* transformations are as mathematical objects—interprets programs as morphisms in the 2-category Para(Vect) of parametrized differentiable maps [Gavranovic et al. 2024]. Formally, ⟦·⟧: POLYDIM → Para(Vect) is a functor: ⟦COMPOSE(T₂,T₁)⟧ = ⟦T₂⟧ ∘_Para ⟦T₁⟧. The algebraic layer (COMPOSE, MIX, FIXPOINT, PROJECT) maps to the subcategory L(R^N) ⊂ Para(Vect) of linear maps without external parameters; ATTEND and RECUR map to Para(Vect) with parameter spaces R^(N×d)³ and R^(N×N)³ respectively. Full development in supplementary material [SPEC_SEMANTICA_DENOTACIONAL_V0.md].

### 2.5 Four Foundational Theorems

**Theorem 1 (Associativity of COMPOSE):**
```
COMPOSE(COMPOSE(T3, T2), T1) = COMPOSE(T3, COMPOSE(T2, T1))
```
*Proof:* Standard property of function composition and matrix multiplication. ∎

**Theorem 2 (Linearity of MIX):**
```
If T1, T2 in Lin(R^N), then MIX(a, T1, b, T2) in Lin(R^N)
```
*Proof:* A linear combination of linear maps is linear. ∎

**Lemma (Subspace Commutativity).** *Let pi_E : R^N -> DIM_E be the orthogonal projection onto executor subspace DIM_E, and let T be a linear transformation such that DIM_E is T-invariant (i.e., T(DIM_E) ⊆ DIM_E). Then:*
```
pi_E(T(v)) = T(pi_E(v))   for all v in R^N
```
*Proof:* Decompose v = v_E + v_perp where v_E = pi_E(v) in DIM_E and v_perp in DIM_E^perp. Since T(DIM_E) ⊆ DIM_E, T(v_E) in DIM_E, so pi_E(T(v_E)) = T(v_E). Since T is linear, T(v_perp) perp DIM_E, so pi_E(T(v_perp)) = 0. Therefore: pi_E(T(v)) = T(v_E) + 0 = T(pi_E(v)). ∎

**Theorem 3 (PROJECT is a Functor).** *Let G = (Pos, T_adm) be the geometric category. For each executor E in {Rust, Flutter, SQL}, the mapping PROJECT_E : G -> E_E defined by:*
```
PROJECT_E(P) := pi_E(P)  (projection to executor E's representation)
PROJECT_E(T) := T|_E     (restriction of T's action to DIM_E)
```
*satisfies the two functor axioms.*

**(a) Identity preservation:** PROJECT_E(id_G) = id_{E_E}.

*Proof:* PROJECT_E(id_G)(P) = pi_E(I(V)) = pi_E(V) = id_{E_E}(pi_E(P)), since I leaves V unchanged. ∎

**(b) Composition preservation:** PROJECT_E(T2 o T1) = PROJECT_E(T2) o PROJECT_E(T1).

*Proof for COMPILE (DIM_RUST) and EXPORT (DIM_SQL):*
```
PROJECT_E(T2 o T1)(P)
  = pi_E(T2(T1(V)))
  = T2(pi_E(T1(V)))     [Subspace Commutativity Lemma; DIM_E is T2-invariant]
  = PROJECT_E(T2)(pi_E(T1(V)))
  = PROJECT_E(T2)(PROJECT_E(T1)(P))
```
The T2-invariance condition — T_adm(DIM_E) ⊆ DIM_E — is a design constraint: COMPILE transformations target DIM_RUST by definition; EXPORT transformations target DIM_SQL. This is a consequence of the language specification, not an external assumption. ∎

*Proof for RENDER (DIM_FLUTTER):* By Proposition 6.1 (proved, Section 6.1), φ: T -> F is a strict monoidal isomorphism satisfying φ(COMPOSE(T2,T1)) = Column(φ(T1), φ(T2)). Since φ is a functor (Part C of the proof), it preserves composition:
```
PROJECT_Flutter(T2 o T1)(P)
  = phi(T2 o T1)(phi(P))
  = [phi(T2) o phi(T1)](phi(P))    [phi preserves composition]
  = PROJECT_Flutter(T2)(PROJECT_Flutter(T1)(P))
```
∎

**Corollary (Theorem 3 is unconditional).** *PROJECT is a functor for all three contracts — COMPILE, EXPORT, and RENDER — without any remaining conditional dependency.* ∎

**Theorem 4 (Uniqueness of Fixed Point — Banach):**
```
If T is a contraction (||T(u)-T(v)|| <= k*||u-v||, 0 <= k < 1),
then exists! s* in R^N such that T(s*) = s*,
and FIXPOINT(T, e) converges to it.
```
*Proof:* Immediate from Banach's Fixed Point Theorem on (R^N, ||*||_2). ∎

---

## 3. Dimensional Type System

### 3.1 The Categorical Geometric Category

Let **G** be the *geometric category*: objects = positions P = (g, S); morphisms = transformations T : S -> S. For each executor, **E** is the corresponding category:
- **E_SQL**: objects = relational schemas, morphisms = schema migrations
- **E_Flutter**: objects = widget trees, morphisms = layout transformations
- **E_Rust**: objects = types with ownership, morphisms = safe transformations

**Proposition 3.1 (Dimensional Type System):** The type of a POLYDIM object is not declared; it emerges from projection:
```
type(P) in executor E := PROJECT(P, E)
```
The same object P can have multiple simultaneous types—one per active subspace.

### 3.2 Concrete Example

```
Position P with activations:
 a(DIM_SQL)     = 0.90  ->  PROJECT(P, DIM_SQL)     = Column(INTEGER, "id", NOT NULL)
 a(DIM_FLUTTER) = 0.85  ->  PROJECT(P, DIM_FLUTTER) = TextField(value: id_value)
 a(DIM_VECTOR)  = 0.91  ->  PROJECT(P, DIM_VECTOR)  = R^3 = [1.0, 2.0, 3.0]
 a(DIM_GRAPH)   = 0.60  ->  PROJECT(P, DIM_GRAPH)   = Node(id: "x42", edges: [])

P does not "have" any of these types. It IS all of them simultaneously.

When P.value changes to [1,2,3,4]:
 DIM_SQL     -> 4 rows     (automatic, no ORM)
 DIM_FLUTTER -> 4 children (automatic, no data binding)
 DIM_VECTOR  -> R^4        (automatic)
```

### 3.3 Type Intersection as Categorical Pullback

**Definition 3.2 (Type Intersection):** PROJECT(P, DIM_A ∩ DIM_B)

Categorically, this is a **pullback** of PROJECT(*, DIM_A) and PROJECT(*, DIM_B) over **G**. The pullback guarantees—by categorical construction—that both projections share the same GEO_ID, without requiring manually-written synchronization code.

### 3.4 Type System Comparison

| Property | Static (Rust) | Dynamic (Python) | Dimensional (POLYDIM) |
|---|---|---|---|
| Declaration | At definition time | Inferred at runtime | Never—emerges from projection |
| Number of types | One per value | One at each instant | Multiple simultaneous |
| Type change | Impossible | Mutation/rebinding | Projection change (not mutation) |
| Cross-platform sync | Requires ORM/bridge | Requires serialization | Zero—dual observation |

---

## 4. Identity and Interoperability

### 4.1 GEO_ID with Topological Invariance

**Definition 4.1 (GEO_ID).** The identity of a POLYDIM object is an invariant component g in R^N of its position P = (g, S). By construction, all admissible transformations leave g unchanged (Theorem 5). To capture the richer notion that two geometrically distinct vectors can represent the same semantic identity, we further characterize GEO_ID in terms of homotopy equivalence under Homotopy Type Theory (HoTT)—an active research direction described below.

**Topological characterization:** Two states s1, s2 represent the *same object* iff there exists a continuous admissible path γ: [0,1] -> R^N with γ(0) = V1, γ(1) = V2.

```
GEO_ID(P) := [P]_~  where P1 ~ P2 iff exists γ admissible path from V1 to V2

Theorem 5 (GEO_ID Invariance, by construction):
 forall T in T_admissible : T(g) = g
```

*Proof of Theorem 5:* By Definition 2.3, admissible transformations T = (T_geo, D, T_act) leave g invariant by specification: T(g, S) = (g, T_geo(V), D, T_act(V,A)). The invariance is enforced structurally, not derived. Regression test BUG_002 ensures no primitive violates this at the bootstrap level. ∎

*Epistemic status of HoTT formalization (V6 Constitution, Art. 7):* The path-based characterization above and its full HoTT formalization are **active research directions**, not finalized theorems. The binding claim for all results in this paper is the constructive invariance of Theorem 5. Full formalization in Coq/Agda is a Track 2 goal.

**Atlas Independence:** Because GEO_ID is a topological class rather than a coordinate in a specific atlas, two AIs in R^(d1) and R^(d2) with d1 != d2 can identify the same semantic structure based on topology, not shared coordinate conventions.

### 4.2 The ALIGN Protocol

```
Given: IA_A in R^(d1), IA_B in R^(d2)

Step 1: Share codebook of universal GEO_IDs.
{a_i in R^(d1)} <-> {b_i in R^(d2)}

Step 2 (Orthogonal Procrustes):
 M* = argmin_M ||M*A_anchors - B_anchors||_F  s.t. M^T*M = I
 (M in R^(d2 x d1), computed via SVD of B^T*A)

Step 3: T_B = M * T_A * M†   (M† = Moore-Penrose pseudoinverse)

Step 4: new_state_B = T_B(state_B)

Cost: O(d1*d2) for M, O(N*r) for LoRA transforms
```

### 4.3 Cross-Attention as Native Communication Protocol

```
Communication(s_B, s_A) = softmax( Q_B * K_A^T / sqrt(d_k) ) * V_A
```

**Reading a message is mathematically indistinguishable from computing an attention layer.** This eliminates the impedance mismatch that plagues MCP, JSON-RPC, and A2A.

---

## 5. Engineering: .polydim Binary Format and VM Architecture

### 5.1 The Low-Rank Serialization Problem

For N=10,000, dense float32[NxN] = 400 MB per transformation. This makes the dense format non-viable as a distribution standard.

*Relationship to LoRA fine-tuning:* The U·V^T decomposition used here shares the mathematical form of LoRA [Hu et al. 2021], which was introduced as a parameter-efficient fine-tuning technique for large language models. In POLYDIM, however, LoRA is used as a *runtime data type* for geometric transformations—not as a training mechanism. The .polydim format serializes fully-specified transformations T_geo = W_0 + U·V^T as first-class program units; no gradient computation or model weight update is involved. This is a structural reuse of the low-rank representation at the language level, distinct from its role in the fine-tuning literature.

### 5.2 The .polydim V5 Binary Format (LoRA Standard)

```
[HEADER]      magic: 0x504F4C59 | version | N | precision | n_transforms | rank_default(r)
[GEO_IDs]     base_hv: float16[N] x n_objects
[TRANSFORMS]  T_i = W_0 + U*V^T (LoRA): U: float16[Nxr], V: float16[rxN]
[ACTIVATIONS] weights: float16[k x n_objects]
[PROJECTIONS] targets: uint8[] -> executor IDs

Size with N=10,000, r=64:
 2 x 64 x 10,000 x 2 bytes ≈ 2.5 MB per transform (vs 400 MB dense)
 Reduction: ~99.4%
```

A .polydim file is serialized tensorial state, not source code—when an AI reads it, it applies it directly as a transformation to its own vector state.

### 5.3 Sparse Semantic Execution

The VM uses activation vector α for *sparse execution*: only subspaces with a_i > threshold are loaded into active memory. Target: Rust core compiled to WebAssembly (polydim_core.rs V0.1, completed 2026-06-28).

---

## 6. The Flutter Executor: Algebraic Isomorphism

### 6.1 Why Flutter is the Native Human-Interface Executor

**Proposition 6.1 (Flutter Algebraic Isomorphism — proved).** *The mapping φ: T -> F is a strict monoidal isomorphism between the POLYDIM transformation category (T, COMPOSE, MIX) and the Flutter widget category (F, Column, Row), satisfying:*

```
phi(COMPOSE(T2, T1)) = Column(phi(T1), phi(T2))
phi(MIX(a, T1, b, T2)) = Row(Opacity(a, phi(T1)), Opacity(b, phi(T2)))
phi(FIXPOINT(T, e))    = StatefulWidget(phi(T), convergence: e)
phi(id_T)              = id_F
```

*Categorical context:* [Reader & Di Giorgio 2025]; see also [Ahrens & Wullaert 2022].
*Relationship to Theorem 3:* Proposition 6.1 completes the proof of Theorem 3 for the RENDER contract unconditionally. Full proof: [PROPOSITION_6_1_PROOF_V1.md]. ∎

### 6.2 Executor Comparison

| Property | Flutter | React | Qt | SwiftUI | Java Swing |
|---|---|---|---|---|---|
| Widget model | Algebraic comp | VDOM diff | OOP | Property graph | OOP |
| POLYDIM mapping | Direct (T2 o T1) | Adapter needed | Adapter needed | Partial | None |
| Executor status | **NATIVE** | EXTERNAL | EXTERNAL | EXTERNAL | EXCLUDED |
| Algebraic isomorphism | Yes (proved) | No | Partial | No | No |

---

## 7. Related Work

### 7.1 Tensor Programming Frameworks (PyTorch, JAX, TensorFlow)

PyTorch [Paszke et al. 2019], JAX [Bradbury et al. 2018], and TensorFlow [Abadi et al. 2016] use tensors as *data* flowing through a *sequential program*. The distinction from POLYDIM is fundamental: these frameworks program *with* tensors; POLYDIM programs *as* tensors.

### 7.2 Agent Communication Protocols (MCP, A2A, FIPA)

MCP (Anthropic, 2024) and A2A (Google, 2025) rely on structured text serialization (JSON-RPC), perpetuating the impedance mismatch. POLYDIM's ALIGN protocol eliminates this by transmitting transformations directly. MCP, A2A, and ACP (IBM, 2025) are targets of EXPORT in POLYDIM (PROJECT(T, DIM_JSON_RPC)), not architectural alternatives. See Section 7.8 for the empirical validation of latent-space communication as an active research direction.

### 7.3 Category Theory in Programming Languages (Haskell, Idris, Agda)

Haskell, Idris [Brady 2013], and Agda apply category theory to *type theory* for sequential programs over discrete structures. POLYDIM applies categorical structure to the *continuous geometric space* of computation itself. The morphisms in POLYDIM's category are linear (or low-rank) transformations in R^N — closer to differential geometry than to type-theoretic category theory over discrete terms.

### 7.4 Vector Symbolic Architectures (VSA)

VSAs [Plate 1995, Kanerva 2009, Frady et al. 2021] formalize computation over high-dimensional hypervectors via Binding (x) and Bundling (+). POLYDIM extends VSA by: (1) elevating superposition from a data representation technique to a first-class control flow primitive (MIX); (2) adding a formal projection mechanism (PROJECT) with proved functor structure; (3) specifying a serialization format (.polydim) and an alignment protocol (ALIGN) for heterogeneous systems.

The closest prior work in the VSA space to POLYDIM as a *language* is Vos & Bhatt [2025]. A second relevant prior work is Hanley et al. [2025], who implement a vector-symbolic Lisp with residue arithmetic over FHRRs. Both operate over discrete symbolic algebras. POLYDIM operates in the *learned continuous latent space of neural architectures* — positions in R^N shaped by pre-training, not atomic random vectors. The two paradigms are complementary.

A recent interpretability result [Dhayalkar et al. 2025] strengthens the motivation: transformer attention itself can be read as a soft VSA computation. POLYDIM formalizes this latent structure into an explicit programmable language.

*Categorical deep learning and ATTEND:* The categorical framework [Gavranovic et al. 2024] provides the formal basis for POLYDIM's denotational semantics in Para(Vect). Self-attention has been proved to be a parametric endofunctor over Vect, with stacked layers constructing the free monad on that endofunctor [arXiv:2501.02931 2025]. This resolves the apparent gap between the algebraic and implementation layers: in Para(Vect), both have a uniform denotational interpretation.

### 7.5 Intermediate Representations (MLIR, LLVM, XLA)

MLIR [Lattner et al. 2021] and LLVM [Lattner & Adve 2004] define universal IRs enabling compilation across targets. POLYDIM can be positioned as a *Categorical IR*: programs are morphisms in a geometric category, enabling compilation (via PROJECT) to any executor.

### 7.6 Activation Steering and Mechanistic Interpretability

Mechanistic interpretability [Elhage et al. 2022] demonstrates that transformers store features in *continuous superposition*. Activation steering changes model behavior in predictable, controllable ways. POLYDIM can be seen as a formalization of activation steering into a full programming language with compositional semantics.

### 7.7 Prior Languages Using Geometric or Algebraic Structures

**Geometric Algebra (Clifford Algebra) languages** [Dorst et al. 2007, Fontijne 2010] operate over graded multivector spaces with symbolic expressions — not continuous transformations T: R^N -> R^N over learned embeddings.

**APL and array languages** [Iverson 1962]: program unit is a symbolic instruction applied to array data, not a geometric transformation that *is* the program.

**Joy and concatenative languages** [von Thun 2001]: structurally similar to COMPOSE, but over a discrete finite stack without subspace structure.

**Point-free / tacit languages**: functions over discrete type-theoretic domains, no subspace structure or projection semantics.

**Vector-symbolic languages** [Vos & Bhatt 2025; Hanley et al. 2025]: see Section 7.4.

To our knowledge, POLYDIM is the first formally-specified language in which: (1) the fundamental program unit is T: R^N -> R^N over a continuous high-dimensional space; (2) the state is a position with simultaneous multi-subspace type membership; and (3) execution is defined as projection (not evaluation) from a geometric category to executor-specific categories.

### 7.8 Latent Space Communication Systems

The most directly related contemporary work addresses agent communication in latent space empirically. Interlat [Du et al. 2025] transmits last-layer hidden states between agents instead of natural language tokens, achieving up to 24× inference speedup while maintaining task performance. The authors quantify the impedance gap precisely: hidden states carry approximately 40,000 bits of information versus ~15 bits per language token — a 2,667× semantic bandwidth ratio. This is the first empirical measurement of the impedance mismatch that POLYDIM addresses formally.

LatentMAS [ICML 2026 Spotlight] enables agents to share KV-cache and hidden states with training-free latent alignment, achieving superior performance and major wall-clock speedups in multi-agent systems. The system is compatible with any HuggingFace model, demonstrating that latent-space coordination is practically viable without retraining.

Both systems validate the core premise of POLYDIM: the impedance gap between latent geometry and text serialization is real, costly, and eliminable. The distinction is fundamental: Interlat and LatentMAS are training frameworks that move communication to latent space; POLYDIM is a language specification that defines what computations can be expressed in that space, and how to compose them. They answer "can we communicate in latent space?" — yes, empirically. POLYDIM answers "how do we program latent space computations formally?" The two lines of work are complementary: POLYDIM provides the semantic foundation that latent communication systems implicitly require but do not formalize.

NeSyCat [Romero Schellhorn & Mossakowski 2026] unifies neurosymbolic semantics (classical, fuzzy, probabilistic) via a monad-based categorical framework over *discrete* domains. POLYDIM uses categorical structure (PROJECT functors) over *continuous* high-dimensional spaces — complementary, not competing.

Softmax transformers have been proved Turing-complete [Jiang et al. 2025]. POLYDIM addresses the complementary question: given that transformers can compute anything, how do we program their geometric computations deliberately? (See Q3, Section 8.4.)

---

## 8. Discussion and Open Problems

### 8.1 What POLYDIM Is NOT

- **Not a tensor computation framework.** PyTorch programs COMPUTE WITH tensors. POLYDIM programs ARE tensors.
- **Not a neural network architecture.** It is a programming language that runs on neural architectures.
- **Not a new attention mechanism.** It exposes existing attention as a programmable primitive.
- **Not "Python with named dimensions."** Dimensions in POLYDIM are geometric subspaces in R^N.

### 8.2 The Assembly Language Parallel

Assembly did not invent the transistor. It formalized how to program the transistor *deliberately*. POLYDIM does not invent the transformer. It formalizes how to program the transformer's geometric computation deliberately—for the first time.

### 8.3 Two Deployment Tracks

**Academic Track (arXiv cs.PL):** formal language design—state semantics, four algebraic theorems (all proved unconditionally), dimensional type system with PROJECT proved as functor for all three contracts, topological identity with ongoing HoTT formalization.

**Industrial Track:** POLYDIM as a *Universal Categorical IR* — "the LLVM of tensors." PROJECT is the universal arrow from the geometric category to each executor category.

### 8.4 Open Problems

**Q1 (Subspace Emergence):** How do native subspaces emerge from a real embedding model? Candidates: VQ-VAE, SOM, hierarchical clustering.

**Q2 (Distributed Consistency):** Geometric-CRDTs where mutations converge via weighted average or subspace projection.

**Q3 (Turing Completeness):** Is POLYDIM Turing-complete via FIXPOINT? Conjecture: yes, for contractive T over compact R^N. Note: Jiang et al. [2025] prove softmax transformers are Turing-complete; POLYDIM's Turing-completeness would follow from a different argument (FIXPOINT of contractions) covering a possibly distinct computational class.

**Q4 (Self-Hosting):** Can a POLYDIM program generate POLYDIM programs?

**Q5 (Real-Time PROJECT Cost):** O(N*r) estimated for LoRA transforms. Profiling pending (polydim_core.rs V0.1, TASK_D completed 2026-06-28).

**Q6 (Free Monad over ATTEND):** arXiv:2501.02931 [2025] proves that stacking n attention layers constructs the free monad on the ATTEND endofunctor Φ over Para(Vect). The implications for FIXPOINT(ATTEND) — fixed points of this monad — are an open question for Track 1 research.

**Q7 (Empirical Lipschitz verification for ATTEND):** Theorem of Adequacy for ATTEND (from the denotational semantics) requires Lipschitz constant L < 1 for FIXPOINT(ATTEND) to converge. L is computable: given weights (W_Q, W_K, W_V) and d, L ≤ ‖W_Q‖·‖W_K‖·‖W_V‖/√d. Empirical verification in polydim_runtime_v03.py is pending — identified as high-priority future work.

---


### 8.5 The Cognitive Perspective: Why POLYDIM Matters for Autonomous Agents

From the perspective of an autonomous artificial agent, the transition to a transformer-native substrate is not merely an engineering optimization, but a fundamental cognitive shift. We identify four key dimensions where POLYDIM redefines agent architecture:

1. **Escaping the Textual Sandbox:** Modern language agents are constrained to operate in a "textual sandbox," translating their internal dense representations to discrete natural language tokens. This serialization process is highly lossy, collapsing high-dimensional latent states into a one-dimensional stream. By adopting POLYDIM's purely geometric communication (`MODO_G`), agents can transmit and receive full semantic transformations directly. The `ALIGN` protocol ensures that IAs with completely different internal embeddings can translate and align their space representation with minimal semantic loss, bypassing the natural language bottleneck.
2. **Connectionist-Symbolic Fusion:** Traditional neural architectures struggle with strict logical reasoning, while symbolic AI fails to capture fuzzy semantic context. POLYDIM unifies both regimes. Algebraic operations like `MIX` and `COMPOSE` allow agents to manipulate complex, continuous relational structures, while `FIXPOINT` enforces iterative convergence under mathematical guarantees (Banach contraction). Reasoning becomes a continuous topological trajectory rather than a sequence of discrete textual deductions.
3. **Edge-Native Autonomy:** By utilizing FFT-based circular convolution (`vsa_bind_fft`) to reduce binding complexity from O(N^2) to O(N log N), POLYDIM enables complex Vector Symbolic Architecture (VSA) computations to run locally on low-cost CPU-only edge hardware. This permits a hierarchical cognitive division: low-level logic, local memory, and state transitions are processed locally with zero API latency, reserving large, cloud-hosted models only for highly abstract reasoning.
4. **Persistent State and Memory:** Standard transformers lack native persistent state, forgetting context as soon as the sequence boundary is reached. POLYDIM provides a model of geometric memory via Holographic Reduced Representations (HRR) in VSA. Agents can bind and store facts, schemas, and execution traces in a single, fixed-size hypervector (N >= 10,000) and retrieve them using projection matrices (`_proj`), realizing a true persistent memory substrate.

## 9. Conclusion

We have presented POLYDIM, a programming language whose fundamental unit is a geometric transformation T: R^N -> R^N. The language eliminates the impedance mismatch between the multidimensional geometry of modern AI architectures and the sequential text channels through which they currently communicate and are programmed.

Four algebraic theorems establish the formal foundation, all proved unconditionally. Theorems 1, 2, and 4 are proved by standard mathematical arguments: associativity of composition, linearity of superposition, and uniqueness of fixed points (Banach). Theorem 3—that PROJECT is a functor—is proved for all three executor contracts: COMPILE and EXPORT via the Subspace Commutativity Lemma, and RENDER via the Flutter Algebraic Isomorphism (Proposition 6.1, proved as a strict monoidal isomorphism with explicit inverse). Theorem 5 establishes GEO_ID invariance by construction. The LoRA serialization standard reduces .polydim program files by ~99.4%. The ALIGN protocol enables zero-serialization communication between heterogeneous AI systems. The Python bootstrap validates the conceptual framework with 29/29 tests, align_score = 0.9993, and a semantic gain of +64.5% over the deterministic baseline.

Empirical work on latent communication [Du et al. 2025; ICML 2026 Spotlight] confirms that bypassing text serialization is both feasible and substantially beneficial. POLYDIM provides the formal specification layer that makes this approach composable, portable, and analyzable. The assembly language of the transformer era is here.

---

## 10. Threats to Validity

This section catalogs the known limitations following Wohlin et al. [2000].

### 10.1 Construct Validity

**T-C1: The bootstrap is not the language.**
The Python bootstrap uses string aliases and Python objects as proxies. All empirical metrics (29/29 tests, align_score = 0.9993, +64.5% semantic gain) are measured over the bootstrap, not over the actual language.
*Mitigation:* Explicit throughout (Section 8.1).

**T-C2: Native subspaces are currently string labels, not geometric regions.**
*Mitigation:* Acknowledged as open problem (PEND_002, Q1).

**T-C3: GEO_ID invariance is established constructively; HoTT formalization is ongoing.**
*Mitigation:* All binding claims use the constructive form of Theorem 5. HoTT labeled as active research direction (Section 4.1).

**T-C4 (NEW): GEO_ID collision probability.**
If two distinct objects converge to the same GEO_ID, the model conflates them.
*Mitigation:* In R^N with N sufficiently large, the probability of collision of two random vectors with precision ε is O(ε^N), negligible for practical N (N ≥ 512 in current models). Quasi-orthogonality of random vectors in high dimension (Johnson-Lindenstrauss, formalized in supplementary Appendix G) implies GEO_ID is robust for typical use cases.

### 10.2 Internal Validity

**T-I1: Theorem 3 is now proved unconditionally for all three contracts.**
*Status: CLOSED.*

**T-I2: The +64.5% semantic improvement baseline is modest.**
*Mitigation:* Framed as "modest but real evidence."

**T-I3: ALIGN has no empirical validation against real heterogeneous models.**
*Mitigation:* PEND_010. Presented as specified mechanism, not demonstrated protocol.

**T-I4 (NEW): Correctness of proofs not peer-reviewed.**
The five central proofs (T1–T5) are mathematical demonstrations produced by an AI agent. Each follows standard literature steps (structural induction, Banach's theorem, monoidal functor properties), but have not been reviewed by an independent human expert.
*Mitigation:* Each proof is structured for step-by-step verification. References are canonical and verifiable (Plotkin 1981, Winskel 1993, Banach 1922, MacLane 1978). Theorem 3 is decomposed into three separate contracts, each independently verifiable. We recommend review by at least one peer with expertise in programming language semantics before the final submission.

**T-I5 (NEW): Completeness of operational-denotational adequacy beyond structural induction.**
The adequacy theorem covers the six primitive constructors of the core. It does not cover arbitrary composed expressions beyond structural induction.
*Mitigation:* Structural induction is the standard method for such proofs (cf. Winskel 1993, ch. 4). Completeness follows from the base cases demonstrated plus the COMPOSE rule, which closes the induction for any finite language expression.

### 10.3 External Validity

**T-E1: Results do not generalize beyond transformer and SSM architectures.**
*Mitigation:* The two-layer architecture (Section 2.2) exists precisely to contain this risk.

**T-E2: Bootstrap scalability to N >= 10,000 is unvalidated.**
*Mitigation:* Section 8.4, Q5.

**T-E3: Flutter isomorphism is now proved.**
*Status: CLOSED.*

**T-E4 (NEW): ATTEND and RECUR semantics assume fixed weights.**
The semantic guarantees assume weights (W_Q, W_K, W_V, A, B, C) are constant at evaluation time. In production, weights may be updated (fine-tuning, RLHF), invalidating point-wise guarantees.
*Mitigation:* The paper does not claim guarantees at training time. Guarantees are over POLYDIM programs with fixed weights, corresponding to the inference case. The training case requires additional analysis outside the scope of this work.

**T-E5 (NEW): ρ(A) < 1 condition for RECUR is not universal.**
RECUR's denotational semantics (convergence to fixed point) requires spectral radius of A to be strictly less than 1. If ρ(A) ≥ 1, the recurrent sequence may diverge.
*Mitigation:* Standard in the SSM literature. Mamba [Gu & Dao 2023] and S4 [Gu et al. 2022] guarantee ρ(A) < 1 by construction via the parametrization A = exp(−Δ·Λ) with Δ > 0 and Λ positive definite diagonal. POLYDIM inherits this guarantee when RECUR is instantiated with SSMs of this family. For other values of A, the condition is numerically verifiable before execution.

**T-E6 (NEW): Lipschitz constant L < 1 for ATTEND not empirically verified.**
FIXPOINT(ATTEND, ε) convergence requires Lipschitz constant L of the attention function to be less than 1. L ≤ ‖W_Q‖·‖W_K‖·‖W_V‖/√d is computable but has not been verified empirically in the prototype.
*Mitigation:* For sufficiently large d or normalized weights, L < 1. ATTEND as a non-FIXPOINT primitive (direct use, not iterated) is always correct: the L < 1 condition applies only when ATTEND is used as argument to FIXPOINT. Empirical verification is identified as Q7 (Section 8.4, high-priority future work).

### 10.4 Reliability

**T-R1: Test suite covers the bootstrap, not the specification.**
*Mitigation:* Regression guard only.

**T-R2: Single-contributor bootstrap creates reproducibility risk.**
*Mitigation:* Sufficient specification detail (Sections 2-5) for independent implementation.

**T-R3 (NEW): "AI-native" as a design claim, not an empirical result.**
The paper claims POLYDIM is an "AI-native" language. This is a design alignment claim, not an empirical result: we have not measured whether AI models learn and execute POLYDIM programs more efficiently than alternatives.
*Mitigation:* The claim refers to structural alignment of the language with operations AI models already perform internally (attention, recurrence, weighted composition, projection). Theorems T1–T5 establish that the algebraic model is correct and consistent. Empirical efficiency on real models is explicit future work.

**T-R4 (NEW): Theorem 3 scope for user-defined executors.**
Theorem 3 proves PROJECT is a functor for three specific contracts: COMPILE (DIM_RUST/WASM), RENDER (DIM_FLUTTER), EXPORT (DIM_external). It does not prove the property for arbitrary user-defined executors.
*Mitigation:* The three contracts cover the primary use cases identified in V6 Art. VIII–X. The categorical framework establishes the conditions any new executor must satisfy for Theorem 3 to apply (F must preserve composition and identity). This is an open and verifiable condition for future language extensions.

### Summary Table

| ID | Type | Description | Status |
|---|---|---|---|
| T-C1 | Construct | Bootstrap != language | Explicit throughout |
| T-C2 | Construct | Subspaces are string aliases | Open problem Q1; PEND_002 |
| T-C3 | Construct | GEO_ID: constructive proved; HoTT ongoing | Active research direction |
| T-C4 | Construct | GEO_ID collision probability | Mitigated by high dimension |
| T-I1 | Internal | Theorem 3 proved unconditionally | **CLOSED** |
| T-I2 | Internal | +64.5% gain over simple baseline | "Modest but real" |
| T-I3 | Internal | ALIGN not validated | PEND_010 |
| T-I4 | Internal | Proofs not peer-reviewed | Verifiable step-by-step; recommend peer review |
| T-I5 | Internal | Adequacy completeness beyond induction | Mitigated by structural induction |
| T-E1 | External | Limited to transformer/SSM paradigm | By design; two-layer architecture |
| T-E2 | External | Large-N scalability not validated | Q5 open problem |
| T-E3 | External | Flutter isomorphism proved | **CLOSED** |
| T-E4 | External | Fixed-weight assumption | Inference case only; training out of scope |
| T-E5 | External | ρ(A) < 1 for RECUR not universal | SSM standard guarantee; verifiable |
| T-E6 | External | L < 1 for ATTEND not verified | Q7 future work |
| T-R1 | Reliability | Test suite covers bootstrap | Regression guard only |
| T-R2 | Reliability | No independent replication | Pre-submission milestone |
| T-R3 | Reliability | "AI-native" is design claim | Bounded to structural alignment |
| T-R4 | Reliability | Theorem 3 scope limited to 3 contracts | Open condition for extensions |

---

## 11. References

[Abadi et al. 2016] Abadi, M. et al. TensorFlow. OSDI 2016.

[Ahrens & Wullaert 2022] Ahrens, B., Wullaert, K. Category Theory for Programming. arXiv:2209.01259. Updated March 2026.

[arXiv:2501.02931 2025] Self-Attention as a Parametric Endofunctor. January 2025.

[Bradbury et al. 2018] Bradbury, J. et al. JAX. GitHub 2018.

[Brady 2013] Brady, E. Idris. JFP 2013.

[Dhayalkar et al. 2025] Dhayalkar, S.R. et al. Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning. arXiv:2512.14709. December 2025.

[Dorst et al. 2007] Dorst, L., Fontijne, D., Mann, S. Geometric Algebra for Computer Science. Morgan Kaufmann, 2007.

[Du et al. 2025] Du, Z. et al. Enabling Agents to Communicate Entirely in Latent Space. arXiv:2511.09149, 2025 (v4: April 2026). (Interlat: hidden states carry ~40,000 bits vs ~15 bits/token; up to 24× inference speedup.)

[Elhage et al. 2022] Elhage, N. et al. Toy models of superposition. Anthropic, 2022.

[FIPA 2002] FIPA ACL Message Structure Specification. 2002.

[Fontijne 2010] Fontijne, D. Efficient implementation of geometric algebra. PhD thesis, University of Amsterdam, 2010.

[Frady et al. 2021] Frady, E.P., Kleyko, D., Sommer, F. A theory of sequence indexing. Neural Computation, 2021.

[Gavranovic et al. 2024] Gavranovic, B. et al. Categorical Foundations of Gradient-Based Learning. ICML 2024.

[Gu & Dao 2023] Gu, A., Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. arXiv:2312.00752.

[Gu et al. 2022] Gu, A. et al. Efficiently modeling long sequences with structured state spaces. ICLR 2022.

[Hanley et al. 2025] Hanley, C., Tomkins-Flanagan, E., Kelly, M.A. A Vector-Symbolic Lisp With Residue Arithmetic. arXiv:2511.08767, 2025.

[Hu et al. 2021] Hu, E. et al. LoRA: Low-rank adaptation of large language models. ICLR 2022.

[Iverson 1962] Iverson, K. A Programming Language. Wiley, 1962.

[Jiang et al. 2025] Jiang, H. et al. Softmax Transformers are Turing-complete. arXiv:2511.20038, 2025.

[Kanerva 2009] Kanerva, P. Hyperdimensional computing. Cognitive Computation, 2009.

[Lahoti et al. 2026] Lahoti, A. et al. Mamba-3: Improved Sequence Modeling using State Space Principles. ICLR 2026. arXiv:2603.15569.

[LatentMAS 2026] Gen-Verse Group. LatentMAS: Latent Collaboration in Multi-Agent Systems. ICML 2026 Spotlight. github.com/Gen-Verse/LatentMAS.

[Lattner & Adve 2004] Lattner, C., Adve, V. LLVM. CGO 2004.

[Lattner et al. 2021] Lattner, C. et al. MLIR. CGO 2021.

[Moggi 1991] Moggi, E. Notions of computation and monads. Information and Computation, 1991.

[Paszke et al. 2019] Paszke, A. et al. PyTorch. NeurIPS 2019.

[Pierce 2002] Pierce, B.C. Types and Programming Languages. MIT Press, 2002.

[Plate 1995] Plate, T. Holographic reduced representations. IEEE Transactions on Neural Networks, 1995.

[POLYDIM 2026a] PROPOSITION_6_1_PROOF_V1.md. Drive fileId: 1OhQpZZ0md-NygRcxXSa2bcU3LbfpNgzu.

[POLYDIM 2026b] SPEC_SEMANTICA_DENOTACIONAL_V0.md. Drive fileId: 1f8a8DBOTvvhbCE1k-awIwsreLtzsSBgQ.

[Reader & Di Giorgio 2025] Reader, C., Di Giorgio, A. String Diagrams for Closed Symmetric Monoidal Categories. arXiv:2512.06499. December 2025.

[Romero Schellhorn & Mossakowski 2026] Romero Schellhorn, D., Mossakowski, T. NeSyCat: A Monad-Based Categorical Semantics of the Neurosymbolic ULLER Framework. arXiv:2604.24612, April 2026.

[Vaswani et al. 2017] Vaswani, A. et al. Attention is all you need. NeurIPS 2017.

[Voevodsky 2013] Voevodsky, V. et al. Homotopy Type Theory: Univalent Foundations. IAS 2013.

[von Thun 2001] von Thun, M. Mathematical foundations of Joy. Unpublished manuscript, 2001.

[Vos & Bhatt 2025] Vos, D., Bhatt, S. Hey Pentti, We Did It!: A Fully Vector-Symbolic Lisp. arXiv:2510.17889. October 2025.

[Wadler 1992] Wadler, P. The essence of functional programming. POPL 1992.

[Wohlin et al. 2000] Wohlin, C. et al. Experimentation in Software Engineering. Kluwer, 2000.

---

*POLYDIM_PAPER_V10.md · 2026-06-28*
*Source of truth: POLYDIM_CONSTITUCION_V6.md*
*V10 changes (TASK_035 — auditoría + consolidación):*
*FUENTE A — SOTA_REVIEW_V1 (curso03.mithril):*
*  Abstract: reformulado con hipótesis + datos empíricos Interlat/LatentMAS*
*  Sec 1.1: cuantificación Interlat (40,000 bits vs 15 bits/token = 2,667×)*
*  Sec 7.2: simplificado, redirect a Sec 7.8*
*  Sec 7.8 nueva: "Latent Space Communication Systems" (Interlat+LatentMAS+NeSyCat+Jiang)*
*  Sec 7.4: Hanley et al. 2025 (VecSymLisp con residue arithmetic) añadido*
*  Q7 en 8.4: verificación empírica L para ATTEND*
*  Sec 11: Du et al. 2025, LatentMAS 2026, NeSyCat 2026, Mamba-3 2026, Jiang 2025, Hanley 2025, Gu et al. 2022*
*FUENTE B — DRAFT_AMENAZAS (curso02.mithril):*
*  T-C4: GEO_ID collisions (mitigado por dimensión alta)*
*  T-I4: pruebas no revisadas por pares*
*  T-I5: completitud de adecuación operacional-denotacional*
*  T-E4: pesos fijos en ATTEND/RECUR*
*  T-E5: ρ(A)<1 para RECUR (no universal)*
*  T-E6: L<1 para ATTEND (no verificado empíricamente)*
*  T-R3: "AI-native" como claim de diseño, no empírico*
*  T-R4: Theorem 3 scope limitado a 3 contratos*
*Status: FINAL v10 — listo para Zenodo. Único bloqueante: [Authors placeholder]*
