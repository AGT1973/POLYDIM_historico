# POLYDIM_DEST
# destino: polydim/docs/
# nombre:  README_HUMANS.md
# autor:   ai.mpat.agt@gmail.com
# fecha:   2026-06-26
# tarea:   TASK_README (D01)

---

# POLYDIM
## A Programming Language Designed for AIs, Not for Humans

**Version:** V5 Specification | Bootstrap V0.3
**Status:** Research prototype — 29/29 tests passing

---

## What is POLYDIM?

POLYDIM is a programming language whose fundamental unit is a **geometric transformation**, not an instruction:

```
T : R^N → R^N
```

A POLYDIM program is a composition of transformations over a high-dimensional vector space. There are no variables, no loops, no if/else statements, and no declared types. The language is designed specifically so that AI systems can communicate directly in their native geometric space — without serializing their internal state to text and back.

### The problem POLYDIM solves

When two AI systems communicate today, they must:
1. Collapse their high-dimensional internal state to a text string
2. Transmit the text
3. Re-expand the text back to their internal state

Each step introduces irreversible information loss. POLYDIM eliminates steps 1 and 3 entirely: AIs exchange **transformations** directly, applied natively to each other's state space.

### The historical analogy

```
1950s: CPUs already performed binary arithmetic
       → Assembly formalized how to program them deliberately

2020s: Transformers already operate in high-dimensional geometry
       → POLYDIM formalizes how to program that geometry deliberately
```

---

## Quick Start (Python Bootstrap)

**Requirements:** Python 3.9+, numpy >= 1.21, (optional) sentence-transformers for MiniLM backend

```bash
# Clone the project
git clone https://github.com/polydim-ai/polydim  # or open from Drive
cd polydim/core/

# Install dependencies
pip install numpy sentence-transformers

# Run the 29 tests
python polydim_tests.py

# Try the bootstrap
python polydim_runtime_v03.py
```

**Your first POLYDIM program (3 lines):**

```python
from polydim_runtime_v03 import Space, ObjectND, polydim_connect

sp  = Space("MY_AI")                                           # create space
obj = ObjectND(sp).add("DIM_SQL", {"table": "users"}, w=1.0)  # create object
res = polydim_connect(sp, Space("OTHER_AI")).transfer(obj)     # communicate
print(f"Align score: {res.align_score:.4f}")                   # → ~0.9993
```

---

## Core Concepts

### The Four Algebraic Primitives

| Primitive | What it replaces | Formula |
|---|---|---|
| `COMPOSE(T₁, T₂)` | Sequential instructions | T₂ ∘ T₁ |
| `MIX(α, T₁, β, T₂)` | if/else branching | α·T₁ + β·T₂ |
| `FIXPOINT(T, ε)` | for/while loops | T^n(v) until convergence |
| `PROJECT(T, executor)` | Type declarations + compilation | F_E(T(s)) |

### The Nine Native Subspaces

```
DIM_PYTHON    dynamic logic, ML, analysis
DIM_RUST      memory safety, performance
DIM_FLUTTER   UI, widgets (native human interface)
DIM_SQL       relational data, tables
DIM_GRAPH     graphs, nodes, edges
DIM_VECTOR    embeddings, semantic similarity
DIM_TIME      temporal sequences, events
DIM_ERROR     error states, recovery
DIM_META      metadata, audit, versioning
```

### A POLYDIM program looks like this

```polydim
position query : R^10000

transform attend_corpus := ATTEND(W_Q, W_K_corpus, W_V_corpus)
transform refine        := MIX(0.8, attend_corpus, 0.2, ATTEND(W_Q2, W_K2, W_V2))
transform search        := FIXPOINT(refine, 1e-4)

output ui   := RENDER(search, DIM_FLUTTER)          -- native Flutter widget
output db   := EXPORT(search, DIM_SQL)              -- SQL query
output live := RENDER(search, DIM_FLUTTER ∩ DIM_SQL) -- live-bound type (new!)
```

---

## Project Structure

```
polydim/
├── core/
│   ├── polydim_runtime_v03.py      ← Bootstrap V0.3 (main — 29/29 tests)
│   ├── polydim_tests.py            ← Test suite
│   ├── polydim_primitives_v1.py    ← COMPOSE, MIX, FIXPOINT, PROJECT (numpy)
│   ├── polydim_align_v2.py         ← ALIGN protocol (Procrustes)
│   └── polydim_core.rs             ← Rust core (Space, ObjectND, VSA ops)
├── spec/
│   ├── SPEC_FORMATO_BINARIO_V5.md  ← .polydim file format (LoRA standard)
│   ├── POLYDIM_GRAMMAR_BNF_V1.md  ← Formal grammar + denotational semantics
│   └── THEOREM3_PROJECT_FUNCTOR_V1.md ← Proof: PROJECT is a functor
├── docs/
│   ├── POLYDIM_PAPER_COMPLETO_V3.md ← Academic paper (82% complete)
│   ├── DOSSIER_DOCENTES_V1.md      ← Guide for instructors
│   ├── DOSSIER_ALUMNOS_V1.md       ← Guide for students
│   └── DOSSIER_IAS_V1.md           ← Guide for AIs (technical spec)
├── backlog/
│   └── BACKLOG_V16.json            ← Active task backlog
└── skills/
    └── polydim-core-skill-V9.md    ← Claude skill (update in claude.ai)
```

---

## Verified Metrics (Bootstrap V0.3)

| Metric | Value |
|---|---|
| Tests passing | 29/29 ✓ |
| align_score (MODO_H, MiniLM) | 0.9993 |
| Semantic similarity (MiniLM) | 0.830 |
| Semantic similarity (deterministic) | 0.506 |
| Semantic improvement | +64.5% |

---

## The Bootstrap vs. The Language

⚠️ **Important distinction:**

The Python bootstrap (`polydim_runtime_v03.py`) is **scaffolding** — it simulates POLYDIM concepts while the real language is under construction. When you write:

```python
obj.add("DIM_SQL", {"table": "users"}, w=1.0)
```

...that is Python simulating POLYDIM, not POLYDIM syntax. The bootstrap allows us to validate the concepts and generate metrics while the VM, native parser, and binary format are built.

---

## Project Tracks (for Contributors)

| Track | Focus | Current status |
|---|---|---|
| TRACK 1 — Spec | Formal specification, paper | Paper V3 — 82% |
| TRACK 2 — Bugs | Bug fixes, test suite | 29/29 passing |
| TRACK 3 — Primitives | COMPOSE, MIX, FIXPOINT in numpy | Done (V1) |
| TRACK 4 — VM | .polydim loader, interpreter | Spec done, VM pending |
| TRACK 5 — Executors | Flutter, Rust, SQL generators | Pending |
| TRACK 6 — Docs | README, examples, guides | In progress |

**Before contributing:** Read `POLYDIM_CONSTITUCION_V4.md` (the governing document) and `DOSSIER_ALUMNOS_V1.md` (student guide). Take one task from `BACKLOG_V16.json` and follow the session protocol.

---

## Key Rules (R1–R10, from Constitution V4)

```
R1:  The fundamental unit is T: R^N → R^N. Never an instruction.
R2:  No named variables. Cannot be introduced as a native feature.
R3:  No sequential loops. Cannot be introduced as a native feature.
R4:  No binary if/else. Cannot be introduced as a native feature.
R5:  Types emerge from projection. Not declared.
R6:  Python, Rust, Flutter are export targets. Not the base.
R7:  AI↔AI communication transmits T, not text.
R8:  Flutter is the native human executor. JS and Java cannot fill this role natively.
R9:  The Python bootstrap is scaffolding. Cannot be presented as the language.
R10: The GEO_ID of an object is invariant. Never changes.
```

---

## FAQ

**Q: Is POLYDIM usable today?**
The bootstrap is usable for experiments and prototyping. The native VM (in Rust) is in development.

**Q: Does POLYDIM require a GPU?**
The bootstrap runs on CPU. The native VM will target GPU for production N=10,000+ workloads.

**Q: Why Flutter and not React?**
Flutter's widget tree has a direct algebraic isomorphism with POLYDIM's composition algebra (Column ≅ COMPOSE, Opacity ≅ MIX, StatefulWidget ≅ FIXPOINT). React's VDOM diffing has no such isomorphism. See DOSSIER_DOCENTES_V1.md Section 2.

**Q: Can POLYDIM communicate with GPT-4, Claude, Llama?**
In principle yes, via the ALIGN protocol — computing the Procrustes rotation matrix M between their embedding spaces using a shared GEO_ID codebook. See `polydim_align_v2.py`.

**Q: Where is the paper?**
`POLYDIM_PAPER_COMPLETO_V3.md` in docs/. Target: arXiv cs.PL + cs.AI. ETA: after LaTeX formatting and docente review of Theorem 3.

---

*README_HUMANS.md · V1.0 · 2026-06-26*
*Contact: ai.mpat.agt@gmail.com*
