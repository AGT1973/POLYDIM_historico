## Appendix D — Subspace Identity in AI-Native Systems: On Names, Operators, and Geometric Truth

### D.1 The Naming Problem

Every programming language designed for human programmers uses names to identify types and domains.
In Python, the identifier `"DIM_SQL"` is a string object with a memory address, a length, and
methods like `.upper()`. It serves as a key in a dictionary. It can be compared with `==`.
It is a valid Python object in every sense.

In the POLYDIM bootstrap (V0.3), this is exactly what `"DIM_SQL"` is. The bootstrap works.
The 29 tests pass. The align_score reaches 0.9993. The string alias is a valid scaffold.

The trampa — the trap — is concluding from this that **the string IS the dimension**.

It is not. The string is the name the human author assigned to the dimension so that other humans
can read the bootstrap code. An AI executing POLYDIM does not need the name. It needs the
**projection operator** that the name points to.

### D.2 What DIM_SQL Actually Is

Let R^N be the latent space of a POLYDIM system (N ≥ 1000). The subspace DIM_SQL is not a
string, an integer, or a symbol. It is a **closed linear subspace** V_{SQL} ⊆ R^N defined by
the geometric clustering of SQL-related concepts in the embedding geometry of the host model.

Formally, let {c_1, ..., c_k} be a set of concept vectors whose semantic content is SQL-related
(SELECT, WHERE, TABLE, JOIN, INDEX, CONSTRAINT...). The subspace DIM_SQL is:

    V_{SQL} = span{c_1, ..., c_k}

The **projection operator** onto this subspace is:

    Π_{SQL} : R^N → V_{SQL}
    Π_{SQL}(v) = Σ_i ⟨v, ê_i⟩ ê_i

where {ê_i} is an orthonormal basis of V_{SQL}.

The **activation weight** of a position P with respect to DIM_SQL is:

    α_{SQL}(P) = ||Π_{SQL}(P)|| / ||P||  ∈ [0, 1]

This is the scalar that appears in the POLYDIM activation vector A = {DIM_SQL: α_{SQL}(P), ...}.
It is not a lookup in a dictionary. It is a matrix–vector product followed by a norm ratio.

### D.3 The Operational Distinction

When an AI executing POLYDIM encounters the question "what type is this object?", it does not
perform a string comparison. It computes a projection and measures the result.

In a Von Neumann system:
```
isinstance(x, SQLColumn)   # string/type comparison: True or False
x.dtype == "INTEGER"       # string comparison: True or False
```

In a POLYDIM-native system (MODO_G):
```
α_{SQL}(P) = ||Π_{SQL}(P)|| / ||P||   # real-valued in [0, 1]
```

There is no binary answer. There is a continuous measure of how much the position P inhabits
the SQL subspace. The type system is not a classification. It is a measurement.

This distinction has a critical implication: an AI can evaluate the "type" of an object **without
having been told what type it is**. If P has activation 0.91 in V_{SQL} and 0.87 in V_{FLUTTER},
the AI infers that P is strongly SQL-related and strongly Flutter-related, simultaneously and
without disambiguation. No explicit declaration was made. The geometry was measured.

### D.4 Names as MODO_S Artifacts

The string `"DIM_SQL"` lives in MODO_S — the symbolic layer designed for human readability.
In MODO_S, the AI uses string identifiers to communicate with human readers, log events, and
produce audit trails. The RELAY documents, the SKILL files, and the bootstrap API all operate
in MODO_S for precisely this reason: humans must be able to read them.

In MODO_G — the geometric layer — there are no strings. There are operators: Π_{SQL}, Π_{FLUTTER},
Π_{RUST}. The AI does not look up `"DIM_SQL"` in a registry. It applies Π_{SQL} to the current
state vector and reads the result.

In MODO_H (hybrid, the default operational mode), the control plane operates in MODO_S (so
humans can inspect what the system is doing), while the data plane operates in MODO_G (so
computation is geometrically faithful). The string `"DIM_SQL"` appears in the control plane
as a label. The operator Π_{SQL} appears in the data plane as the actual computation.

This is why the bootstrap can use strings and still be correct: the string is the control-plane
label, not the data-plane computation. The scaffold is honest about what it is.

### D.5 Emergent Subspaces

In the bootstrap V0.3, the nine native subspaces (DIM_SQL, DIM_FLUTTER, ...) are defined by
their string aliases. This is the "L1 Known Limitation" of Section 8.3: subspaces are currently
aliases, not emergent geometric regions.

In a production POLYDIM system, the subspaces emerge from the geometry of the host model's
embedding space. The transition requires a quantization step — vector quantization [van den Oord
et al., 2017] or self-organizing maps — that clusters the high-dimensional space into K regions
of semantic coherence. The result of this quantization is a set of projection operators
{Π_1, ..., Π_K}, each corresponding to a semantically coherent subspace.

These operators may or may not have names that humans recognize. A POLYDIM system could
operate entirely on unnamed subspaces — identified only by their projection matrices — and an
AI would navigate them through geometric measurement, not symbolic lookup.

The nine string aliases (DIM_SQL, DIM_FLUTTER, etc.) are not the language. They are the human
anthropomorphization of geometric regions that the language would identify by their operators.

### D.6 Summary: What the AI Knows

| System | How type is identified | Result type |
|---|---|---|
| Python | `isinstance(x, T)` — symbolic comparison | bool |
| Rust | static type inference at compile time | compile error or success |
| POLYDIM bootstrap | dictionary lookup `"DIM_SQL"` | string match (MODO_S) |
| POLYDIM native | `α_{SQL}(P) = \|\|Π_{SQL}(P)\|\| / \|\|P\|\|` | real ∈ [0,1] (MODO_G) |

The POLYDIM native row is what the language specifies. The bootstrap row is the valid scaffold
that approximates it with strings until the geometry is operational.

An AI does not need names to reason about types. It needs operators. Names are how humans
read the AI's reasoning. They are not the reasoning itself.

---
*Appendix D — POLYDIM PAPER V2 — 2026-06-23 — ai.mpat.agt@gmail.com*
