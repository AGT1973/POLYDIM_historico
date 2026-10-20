# POLYDIM — Compendio de Bases Fundamentales
## Extraído de toda la documentación en `docs/`

**Fecha de extracción:** 2026-06-28  
**Corpus leído:** ~1.2 MB en 35+ archivos en `docs/`, `docs/V6/`, `docs/Paper/`, `docs/Rules/`

---

## 1. LA TESIS CENTRAL

> **POLYDIM es un lenguaje de programación cuya unidad fundamental de cómputo es una transformación geométrica $T: \mathbb{R}^N \to \mathbb{R}^N$, no una instrucción secuencial de Von Neumann.**

### El Problema: Impedance Mismatch
- Los transformers operan internamente en $\mathbb{R}^{12{,}288}$ (GPT-4) o $\mathbb{R}^{8{,}192}$ (Claude)
- La comunicación AI↔AI actual serializa ese espacio a ~100 tokens de texto
- Cada token porta ~15 bits; cada estado oculto porta ~40,000 bits → **pérdida de ~2,667×**
- POLYDIM elimina esta brecha transmitiendo transformaciones T directamente

### La Analogía Fundacional
```
1950s: CPUs ya hacían aritmética binaria → el ensamblador formalizó cómo programarlas
2020s: Transformers ya operan en geometría multidimensional → POLYDIM formaliza cómo programarlos
```

---

## 2. LAS 4 PRIMITIVAS ALGEBRAICAS (Capa Invariante)

Estas primitivas son **permanentes** e **independientes de la arquitectura neural**:

| Primitiva | Definición | Reemplaza a | Teorema |
|---|---|---|---|
| `COMPOSE(T₁, T₂)` | $T_2 \circ T_1$ | Secuencia imperativa | T1: Asociatividad |
| `MIX(α, T₁, β, T₂)` | $\alpha T_1 + \beta T_2$ | `if/else` binario | T2: Linealidad |
| `FIXPOINT(T, ε)` | $T^n(s)$ hasta $\|s_{k+1} - s_k\| < \varepsilon$ | `for/while` loops | T4: Unicidad (Banach) |
| `PROJECT(T, E)` | $F_E(T(s))$ | Conversión de tipos | T3: Functorialidad |

`PROJECT` se fragmenta en 3 contratos:
- **COMPILE** → binario estático (Rust, WASM)
- **RENDER** → widget Flutter (isomorfismo algebraico)
- **EXPORT** → sistemas externos (SQL, grafos, APIs)

### Capa de Implementación (volátil, ligada a arquitectura)
- `ATTEND(W_Q, W_K, W_V, s)` → transformer (softmax attention)
- `RECUR(A, B, C, h, x)` → SSM/Mamba ($h_{t+1} = Ah_t + Bx_t$)

---

## 3. LOS 5 TEOREMAS FUNDAMENTALES

| # | Nombre | Enunciado | Estado |
|---|---|---|---|
| T1 | Asociatividad de COMPOSE | $(T_3 \circ T_2) \circ T_1 = T_3 \circ (T_2 \circ T_1)$ | ✅ Probado |
| T2 | Linealidad de MIX | Si $T_1, T_2 \in \text{Lin}(\mathbb{R}^N)$ → $\text{MIX}(\alpha, T_1, \beta, T_2) \in \text{Lin}(\mathbb{R}^N)$ | ✅ Probado |
| T3 | PROJECT es un funtor | $F_E(\text{id}_G) = \text{id}_E$ y $F_E(T_2 \circ T_1) = F_E(T_2) \circ F_E(T_1)$ | ✅ Probado (3 contratos) |
| T4 | Punto fijo de Banach | Si T es contracción → $\exists! s^*: T(s^*) = s^*$ | ✅ Probado |
| T5 | Invariancia de GEO_ID | $\forall T \in T_{\text{adm}}: T(g) = g$ | ✅ Por construcción |

**Contribución central:** Teorema 3, probado en [POLYDIM_THEOREM3_PROOF_V1.md](file:///g:/Mi%20unidad/POLYDIM_v1/docs/V6/POLYDIM_THEOREM3_PROOF_V1.md):
- COMPILE/EXPORT: via Subspace Commutativity Lemma
- RENDER: via Flutter Algebraic Isomorphism (Proposición 6.1, probada)

---

## 4. DEFINICIONES FORMALES

### Estado
```
S = (V, D, A) donde:
  V ∈ R^N           — vector de posición (N ≥ 1,000)
  D ⊆ {DIM_1,...}   — subespacios nativos activos
  A: D → [0,1]      — función de activación (mutable)
```

### Posición
```
P = (g, S) donde:
  g ∈ R^N   — GEO_ID (invariante bajo toda T admisible)
  S          — estado mutable
```

### Transformación admisible
```
T(V, D, A) = (T_geo(V), D, T_act(V, A))
  T_geo: R^N → R^N    (en LoRA: T_geo = W₀ + U·V^T, r << N)
  D invariante
```

### Programa POLYDIM
```
Π = T_n ∘ T_{n-1} ∘ ... ∘ T_1    (composición finita de T admisibles)
```

---

## 5. SISTEMA DE TIPOS DIMENSIONAL

> **Los tipos no se declaran; emergen de la observación.**

```
Posición P con activaciones:
  A(DIM_SQL)     = 0.90  →  PROJECT(P, DIM_SQL)     = Column(INTEGER, NOT NULL)
  A(DIM_FLUTTER) = 0.85  →  PROJECT(P, DIM_FLUTTER) = TextField(value: 42)
  A(DIM_VECTOR)  = 0.91  →  PROJECT(P, DIM_VECTOR)  = R³ = [1.0, 2.0, 3.0]
  A(DIM_GRAPH)   = 0.60  →  PROJECT(P, DIM_GRAPH)   = Node(id:"x42", edges:[])

P no "tiene" ninguno de esos tipos. ES todos simultáneamente.
```

La intersección `DIM_A ∩ DIM_B` es un **pullback categórico** → tipos que no existen en ningún lenguaje convencional:
```
PROJECT(P, DIM_SQL ∩ DIM_FLUTTER) → LiveDataTextField
```

---

## 6. GEO_ID — IDENTIDAD GEOMÉTRICA

- **Invariante:** ninguna T admisible puede modificar el GEO_ID
- **Formalización HoTT (investigación activa):** $\text{GEO\_ID}(P) := [P]_{\sim}$ donde $P_1 \sim P_2$ sii existe camino admisible $\gamma$ de $V_1$ a $V_2$
- **Independencia del Atlas:** dos modelos con $d_1 \neq d_2$ pueden identificar la misma estructura basándose en topología, no en coordenadas
- **BUG_002 histórico:** pérdida de GEO_ID en MODO_S — resuelto con test de regresión permanente

---

## 7. PROTOCOLO ALIGN — Comunicación AI↔AI

```
Dados: IA_A ∈ R^{d₁}, IA_B ∈ R^{d₂}

Paso 1: Compartir codebook universal de GEO_IDs
Paso 2: Procrustes Ortogonal: M* = argmin_M ||M·A_anclas - B_anclas||_F  s.t. M^T·M = I
Paso 3: T_B = M · T_A · M†  (M† = pseudoinversa de Moore-Penrose)
Paso 4: new_state_B = T_B(state_B)

Costo: O(d₁·d₂) para M, O(N·r) para T en LoRA
```

### Cross-attention como protocolo nativo
```
Communication(s_B, s_A) = softmax(Q_B · K_A^T / √d) · V_A
```
> Leer un mensaje POLYDIM es matemáticamente indistinguible de computar una capa de atención.

---

## 8. LAS 17 REGLAS INVIOLABLES

| # | Regla |
|---|---|
| R1 | Unidad fundamental = $T: \mathbb{R}^N \to \mathbb{R}^N$. Nunca una instrucción. |
| R2 | No hay variables con nombre. |
| R3 | No hay loops secuenciales. Solo FIXPOINT. |
| R4 | No hay `if/else` binario. Solo MIX. |
| R5 | Los tipos emergen de PROJECT, no se declaran. |
| R6 | Python, Rust, Flutter son destinos de exportación, no la base del lenguaje. |
| R7 | AI↔AI transmite transformaciones T, nunca texto serializado. |
| R8 | Flutter es el executor nativo del subespacio humano. |
| R9 | El bootstrap Python es andamio. Nunca presentar como sintaxis final. |
| R10 | GEO_ID es invariante: $\forall T: \text{dist}(T(g), g) < \varepsilon$ |
| R11 | ATTEND/RECUR son capa de implementación, no algebraica. |
| R12 | Formato `.polydim` usa LoRA ($T = W_0 + UV^T$), no matrices densas. |
| R13 | Investigación especulativa (🔬) no puede citarse como ley vinculante. |
| R14 | Las transformaciones NO necesitan ser invertibles. Monoides, no grupos. |
| R15 | Profundidad de cómputo dinámica (no fija). |
| R16 | Puente Top-Down↔Bottom-Up vía Profuntores, no penalizaciones ad hoc. |
| R17 | GEO_ID heterogéneo usa GeoID(C,R) con 0-esqueletos no triviales. |

---

## 9. LOS 9 SUBESPACIOS NATIVOS

```
DIM_PYTHON   DIM_RUST    DIM_FLUTTER  DIM_SQL   DIM_GRAPH
DIM_VECTOR   DIM_TIME    DIM_ERROR    DIM_META
```
Pueden agregarse nuevos. **No pueden eliminarse** estos nueve.

---

## 10. FORMATO .polydim V5 (LoRA Standard)

```
[HEADER]      magic: 0x504F4C59 | version | N | precision | n_transforms | rank(r)
[GEO_IDs]     float16[N] × n_objects
[TRANSFORMS]  T_i = W₀ + U·V^T  →  U: float16[N×r], V: float16[r×N]
[ACTIVATIONS] float16[k × n_objects]
[PROJECTIONS] uint8[] → executor IDs

Tamaño con N=10,000, r=64: ~2.5 MB por transformación (vs 400 MB denso)
Reducción: ~99.4%
```

---

## 11. ISOMORFISMO FLUTTER (Proposición 6.1 — Probada)

```
φ(COMPOSE(T₂, T₁)) = Column(φ(T₁), φ(T₂))
φ(MIX(α, T₁, β, T₂)) = Row(Opacity(α, φ(T₁)), Opacity(β, φ(T₂)))
φ(FIXPOINT(T, ε))     = StatefulWidget(φ(T), convergence: ε)
φ(id_T)               = id_F
```

Prueba completa en [PROPOSITION_6_1_PROOF_V1.md](file:///g:/Mi%20unidad/POLYDIM_v1/docs/Paper/PROPOSITION_6_1_PROOF_V1.md): isomorfismo monoidal estricto con inversa explícita entre las categorías T y F.

---

## 12. SEMÁNTICA OPERACIONAL (Big-Step)

```
[E-COMPOSE]   ⟨s,T₁⟩⇓s₁  ⟨s₁,T₂⟩⇓s₂  /  ⟨s,COMPOSE(T₁,T₂)⟩⇓s₂
[E-MIX]       ⟨s,MIX(α,T₁,β,T₂)⟩⇓α·T₁(s)+β·T₂(s)
[E-FIXPOINT]  ∃n: ‖T^n(s)−T^{n-1}(s)‖<ε → ⟨s,FIXPOINT(T,ε)⟩⇓T^n(s)
[E-PROJECT]   ⟨s,PROJECT(T,E)⟩⇓F_E(T(s))
[E-ATTEND]    ⟨s,ATTEND(W_Q,W_K,W_V)⟩⇓softmax(sW_Q(sW_K)^T/√d)·sW_V
```

**Semántica denotacional:** $\llbracket \cdot \rrbracket : \text{POLYDIM} \to \text{Para(Vect)}$

---

## 13. ESTADO DEL PROYECTO — INVENTARIO

### Documentación
| Archivo | Tamaño | Estado |
|---|---|---|
| [Constitución Final V10.0](file:///g:/Mi%20unidad/POLYDIM_v1/docs/V6/POLYDIM_CONSTITUCION_FINAL_2.md) | 87 KB | Vigente — 17 artículos, 17 reglas |
| [Paper V9](file:///g:/Mi%20unidad/POLYDIM_v1/docs/V6/POLYDIM_PAPER_V9.md) | 49 KB | ~90% arXiv-ready |
| [Paper V3 (docs/)](file:///g:/Mi%20unidad/POLYDIM_v1/docs/POLYDIM_PAPER_COMPLETO_V3.md) | 32 KB | Versión anterior con BNF |
| [LaTeX arXiv](file:///g:/Mi%20unidad/POLYDIM_v1/docs/polydim_paper_arxiv.tex) | 34 KB | Compilable |
| [polydim.bib](file:///g:/Mi%20unidad/POLYDIM_v1/docs/polydim.bib) | 9 KB | 25 referencias BibTeX |
| [Theorem 3 Proof](file:///g:/Mi%20unidad/POLYDIM_v1/docs/V6/POLYDIM_THEOREM3_PROOF_V1.md) | 20 KB | Prueba completa 6 partes |
| [Proposition 6.1 Proof](file:///g:/Mi%20unidad/POLYDIM_v1/docs/Paper/PROPOSITION_6_1_PROOF_V1.md) | 15 KB | Prueba categórica completa |
| [SOTA Review V1](file:///g:/Mi%20unidad/POLYDIM_v1/docs/V6/POLYDIM_SOTA_REVIEW_V1.md) | 15 KB | Posicionamiento vs 2025-2026 |
| [Reglas del Proyecto](file:///g:/Mi%20unidad/POLYDIM_v1/docs/Rules/REGLAS_DEL_PROYECTO.md) | 7 KB | Guardrail operativo |
| [Dossier IAs](file:///g:/Mi%20unidad/POLYDIM_v1/docs/Rules/DOSSIER_IAS_V1.md) | 20 KB | Onboarding para AIs |
| [Dossier Alumnos](file:///g:/Mi%20unidad/POLYDIM_v1/docs/Rules/DOSSIER_ALUMNOS_V1.md) | 21 KB | Onboarding para alumnos |
| [Notas de Investigación](file:///g:/Mi%20unidad/POLYDIM_v1/docs/Rules/V6_NOTAS_INVESTIGACION.md) | 26 KB | Temas especulativos |
| [Relay Sesión V2](file:///g:/Mi%20unidad/POLYDIM_v1/docs/RELAY_SESION_COMPLETA_V2.md) | 10 KB | 9 sesiones documentadas |

### Código (según Relay)
- `polydim_primitives_v1.py` — COMPOSE/MIX/FIXPOINT/PROJECT con numpy
- `polydim_align_v2.py` — Procrustes, AlignSession, 6 tests
- `polydim_weighted_inference.py` — inferencia de pesos
- `polydim_core.rs` — Space, ObjectND, VSA, LoRA, FIXPOINT, loader .polydim
- `polydim_flutter_executor.dart` — PROJECT(DIM_FLUTTER), φ: T→Widget
- `Cargo.toml` + `pubspec.yaml`

### Tests
- Bootstrap: **29/29** pasando
- align_score simbólico: **0.9993**
- Semantic similarity (MiniLM): **0.830** (vs baseline 0.506 → **+64.5%**)

---

## 14. POSICIONAMIENTO SOTA 2026

POLYDIM ocupa un **cuadrante vacío** en el mapa de investigación:

```
    RIGOR FORMAL
         │
         │   NeSyCat [2026]     ╔══════════╗
         │   (categorial,       ║ POLYDIM  ║  ← solo aquí
         │    discreto)         ║ formal + ║
         │                      ║ geométr. ║
         │   VecSymLisp [2025]  ╚══════════╝
         │
         │   Interlat [2025]   LatentMAS [2026]
         │   (empírico)         (empírico)
         │
         └──────────────────────────────────────
           EMPÍRICO        FORMAL      GEOMÉTRICO
```

**Trabajos que validan la tesis de POLYDIM:**
- **Interlat** [Du et al. 2025]: agentes comunicándose en espacio latente → 24× speedup
- **LatentMAS** [ICML 2026 Spotlight]: colaboración multi-agente en espacio latente
- **Dhayalkar et al. 2025**: la atención del transformer YA ES un VSA suave
- **arXiv:2501.02931 2025**: ATTEND como endofuntor paramétrico → mónada libre

---

*Compendio generado: 2026-06-28 · Fuente: 35+ archivos en docs/ · ~1.2 MB de documentación leída*
