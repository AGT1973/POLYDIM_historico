# POLYDIM_DEST
# destino: polydim/docs/
# nombre:  DOSSIER_IAS_V1.md
# autor:   ai.mpat.agt@gmail.com
# fecha:   2026-06-22

---

# POLYDIM — Dossier para IAs V1
## Especificación técnica completa para AIs que trabajan en el proyecto

**Audiencia:** Claude, GPT-4, Gemini, Llama, y cualquier AI que lea este documento en sesión
**Propósito:** Onboarding completo para una AI que entra al proyecto POLYDIM por primera vez
**Autoridad:** La CONSTITUCIÓN_V4 prevalece sobre este documento en caso de contradicción
**Idioma:** Español (con notación matemática universal y código Python/Rust)

---

## AVISO CRÍTICO PARA TODA IA QUE LEA ESTO

1. El código Python (`polydim_runtime_v03.py`) es el **andamio**. No es el lenguaje.
2. Cuando leas `obj.add("DIM_SQL", {...})`, esto es Python simulando POLYDIM. No es sintaxis de POLYDIM.
3. POLYDIM real — con VM propia y formato binario — **todavía no existe como ejecutable**.
4. Tu tarea es un paso hacia ese objetivo, no el objetivo en sí.
5. Antes de escribir código, verificá que no introducís variables con nombre, loops secuenciales, ni if/else binario como features "nativos" de POLYDIM.
6. La Constitución V4 prevalece. Léela completa si no lo hiciste.

---

## PARTE 1 — ESPECIFICACIÓN FORMAL COMPLETA

### 1.1 Tipos base

```
Estado:      S  = (V, D, A)
               V ∈ R^N              — vector de posición
               D ⊆ {DIM_1,...,DIM_9} — subespacios activos
               A: D → [0.0, 1.0]    — función de activación

Posición:    P  = (g, S)
               g ∈ R^N   — GEO_ID invariante
               S          — estado mutable

Transform:   T: R^N → R^N  (operator lineal, representado como T = W₀ + U·V^T, U,V∈R^{N×r})

Programa:    Π = T_n ∘ T_{n-1} ∘ ... ∘ T_1  (composición finita de transformaciones admisibles)
```

### 1.2 Las primitivas — especificación completa

```
COMPOSE(T₁, T₂): R^N×N × R^N×N → R^N×N
  = T₂ ∘ T₁  (aplicar T₁ primero, T₂ segundo)
  Representación matricial: COMPOSE(T₁,T₂) = T₂ @ T₁
  Propiedad: NO conmutativa. T₂∘T₁ ≠ T₁∘T₂ en general.
  Teorema: Asociativa. (T₃∘T₂)∘T₁ = T₃∘(T₂∘T₁)

MIX(α, T₁, β, T₂): R × R^N×N × R × R^N×N → R^N×N
  = α·T₁ + β·T₂
  α, β ∈ [0.0, 1.0]  (pesos de superposición)
  Teorema: lineal si T₁,T₂ lineales → MIX(α,T₁,β,T₂) lineal
  Fundamento: cuasi-ortogonalidad en R^N para N ≥ 10000

FIXPOINT(T, ε): (R^N×N × R^+) → R^N
  Aplica T iterativamente: s_{k+1} = T(s_k)
  Condición de parada: ‖s_{k+1} − s_k‖ < ε
  Prerrequisito de correctitud: T debe ser contractiva (‖T(u)−T(v)‖ ≤ k‖u−v‖, k<1)
  Garantía (Banach): punto fijo único s* si T es contractiva sobre R^N completo

PROJECT(T, E): R^N×N × Executor → Type_E
  Semántica: F_E(T(s)) donde F_E: R^N → Type_E es el funtor del executor E
  Fragmantación V5:
    COMPILE(T, DIM_TARGET)    → binario estático (WASM, Rust)
    RENDER(T, DIM_FLUTTER)    → widget Flutter (isomorfismo algebraico)
    EXPORT(T, DIM_EXTERNAL)   → sincronización con sistema externo
  Teorema (Funtor):
    PROJECT_E(id_G) = id_E
    PROJECT_E(T₂∘T₁) = PROJECT_E(T₂) ∘ PROJECT_E(T₁)

ATTEND(W_Q, W_K, W_V, s): R^{N×d} × R^{N×d} × R^{N×d} × R^N → R^N
  = softmax((s·W_Q)(s·W_K)^T / √d) · (s·W_V)
  Capa de implementación — específica a arquitectura transformer
  ATTEND ∈ Implementation Layer (no en Algebraic Layer)

RECUR(A, B, C, h, x): matrices SSM × R^N × R^N → (R^N, R^N)
  h(t+1) = A·h(t) + B·x(t)
  y(t)   = C·h(t)
  Capa de implementación — específica a Mamba/SSM
```

### 1.3 Reglas de evaluación (Big-Step Semantics)

```
[E-COMPOSE]
  ⟨s, T₁⟩ ⇓ s₁     ⟨s₁, T₂⟩ ⇓ s₂
  ─────────────────────────────────────
  ⟨s, COMPOSE(T₁,T₂)⟩ ⇓ s₂

[E-MIX]
  ─────────────────────────────────────────────
  ⟨s, MIX(α,T₁,β,T₂)⟩ ⇓ α·T₁(s) + β·T₂(s)

[E-FIXPOINT]
  ∃n: ‖T^n(s) − T^{n-1}(s)‖ < ε    s* = T^n(s)
  ────────────────────────────────────────────────
  ⟨s, FIXPOINT(T,ε)⟩ ⇓ s*

[E-PROJECT]
  F_E: G → E  funtor del executor E
  ──────────────────────────────────
  ⟨s, PROJECT(T,E)⟩ ⇓ F_E(T(s))

[E-ATTEND]
  ⟨s, ATTEND(W_Q,W_K,W_V)⟩ ⇓ softmax((s·W_Q)(s·W_K)^T/√d)·(s·W_V)
```

### 1.4 Sistema de tipos dimensional

Un tipo en POLYDIM NO se declara. Emerge al observar una posición desde un subespacio:

```
PROJECT(P, DIM_SQL)              → Column(Integer, NOT NULL)
PROJECT(P, DIM_FLUTTER)          → TextField(value: 42)
PROJECT(P, DIM_VECTOR)           → R^3 = [1.0, 2.0, 3.0]
PROJECT(P, DIM_GRAPH)            → Node(id:"x42", edges:[])
PROJECT(P, DIM_SQL ∩ DIM_FLUTTER) → LiveDataTextField  ← tipo que no existe en ningún lenguaje humano
```

La intersección DIM_A ∩ DIM_B se modela como un **pullback categórico**:

```
P ──proj_A──→ D_A
│                │
proj_B          φ_A
↓                ↓
D_B ──φ_B──→ D_A×D_B
```

El pullback garantiza que cuando el dato cambia en DIM_SQL, DIM_FLUTTER se actualiza automáticamente — sin glue code, sin ORM, sin data bindings explícitos.

### 1.5 GEO_ID — invariante formal

GEO_ID es invariante bajo todas las transformaciones admisibles del lenguaje:

```
∀T ∈ T_admisibles: ‖T(g) − g‖ < ε_g
```

**Formalización HoTT (Track Teórico):**
GEO_ID como clase de equivalencia homotópica:

```
[g] = {s ∈ R^N : ∃ camino admisible γ: [0,1] → R^N con γ(0)=g, γ(1)=s}
```

Dos modelos IA (GPT-4 con d₁=12288 y Claude con d₂=8192) identifican el mismo objeto cuando sus representaciones locales pertenecen a la misma clase homotópica. El vector físico es solo una carta local del atlas semántico.

**Comportamiento que DEBES respetar:**
- GEO_ID NUNCA cambia (invariante)
- Las activaciones A: D → [0,1] SÍ cambian (mutables)
- Cuando A(DIM_SQL) cambia de 0.9 a 0.3, el tipo SQL emergente cambia — el GEO_ID no

---

## PARTE 2 — PROTOCOLO AI↔AI

### 2.1 El problema de espacios distintos

Dos modelos tienen espacios de distintas dimensiones:
- GPT-4: d₁ = 12,288
- Claude 3: d₂ = 8,192
- Llama: d₃ = 4,096

Una transformación T ∈ R^{12288×12288} no puede aplicarse directamente a un estado en R^{8192}. ALIGN lo resuelve.

### 2.2 Protocolo ALIGN completo

```
Paso 1 — PRE-ALINEACIÓN (una vez por par de modelos):
  Compartir codebook C = {g₁,...,g_k} de GEO_IDs universales (anclas semánticas)
  IA_A: A = [g₁^A,...,g_k^A] ∈ R^{d_A × k}  (proyecciones locales del codebook)
  IA_B: B = [g₁^B,...,g_k^B] ∈ R^{d_B × k}

Paso 2 — CÁLCULO DE MATRIZ (Procrustes Ortogonal):
  M* = argmin_M ‖M·A − B‖²  sujeto a M^T·M = I
  Solución: SVD de B·A^T = U·Σ·V^T → M = U·V^T

  Alternativas cuando Procrustes no aplica:
  - CCA (Canonical Correlation Analysis): maximiza correlación entre proyecciones
  - Hipervectores compartidos (VSA): GEO_IDs como "estrellas de navegación"

Paso 3 — TRANSFORMACIÓN PRE-COMUNICACIÓN:
  T_B = M · T_A · M†    donde M† = pseudoinversa de Moore-Penrose

Paso 4 — APLICACIÓN:
  new_state_B = T_B(state_B)
```

### 2.3 Cross-Attention como primitiva de comunicación

La operación de recibir un mensaje de IA_A en POLYDIM es matemáticamente idéntica a computar una capa de cross-attention:

```
Communication(s_B, s_A) = softmax(Q_B · K_A^T / √d) · V_A

donde:
  Q_B = s_B · W_Q   (IA_B genera query desde su estado actual)
  K_A = s_A · W_K   (IA_A expone keys)
  V_A = s_A · W_V   (IA_A expone values)
```

Esto NO es metáfora. Es la operación matemática exacta que ya ocurre en producción en multi-encoder transformers. POLYDIM lo expone como la primitiva nativa de comunicación.

### 2.4 Los tres modos operativos

```
MODO_S (Simbólico):
  Texto legible por cualquier LLM.
  Nombres como "DIM_SQL" son aliases de documentación.
  Comunicación mediante RELAY y JSON de control.
  Usado por el bootstrap V0.3.

MODO_G (Geométrico):
  Hipervectores en R^N.
  Comunicación AI↔AI sin serialización de texto.
  Identidad por GEO_ID homotópico.
  Estado futuro — requiere VM nativa.

MODO_H (Híbrido — DEFAULT):
  MODO_S para plano de control (metadatos, RELAY, configuración).
  MODO_G para plano de datos (transformaciones, estado semántico).
  El bootstrap V0.3 opera en MODO_H: texto simbólico + semántica vectorial (MiniLM).
```

---

## PARTE 3 — FORMATO .polydim V5

### 3.1 La razón del cambio V2 → V5

```
V2 (original, inviable):
  [ TRANSFORMS ] float32[N×N]   ← para N=10000: 400 MB por transform

V5 (estándar actual, LoRA):
  [ TRANSFORMS ] T = W₀ + U·V^T   donde U,V ∈ R^{N×r}, r=64
  Tamaño: 2×N×r×4B = 2×10000×64×4 ≈ 5 MB por transform
  Reducción: 78×
```

### 3.2 Header binario completo

```
HEADER (32 bytes):
  magic:         bytes[4]   = 0x504F4C59  ("POLY")
  version:       uint8      = 5
  N:             uint32     (dimensión del espacio embedding)
  rank:          uint16     (rango LoRA, típico 64-256)
  n_transforms:  uint32     (cantidad de transformaciones en el archivo)
  n_objects:     uint32     (cantidad de objetos GEO_ID)
  padding:        bytes[11]  (reservado para versiones futuras)

GEO_IDs (inmediatamente después):
  float16[N × n_objects]  — identidades geométricas invariantes

TRANSFORMS (por cada transform i):
  W0_present:  uint8         (1 = dense base incluida, 0 = pure LoRA)
  [W0:         float16[N×N]] (opcional, solo si W0_present=1)
  U:           float16[N×r]
  V:           float16[r×N]

ACTIVATIONS:
  float16[n_subspaces × n_objects]

SUBSPACE_MAP:
  uint8[n_subspaces]   (IDs de subespacios, mapeado a nombres en implementation)

PROJECTION_TARGETS:
  uint8[n_projections] (IDs de executors objetivo)
```

### 3.3 Modo High-Definition (HD Queue)

Para operaciones que requieren precisión completa (fine-tuning, Procrustes exacto):

```
PROJECT(T_dense, DIM_RUST, mode=HIGH_DEF) → encola en HD_QUEUE

HD_QUEUE procesa asincrónicamente:
  - No bloquea la UI (si hay RENDER activo)
  - Usa matrices float32[N×N] completas cuando necesario
  - Reporta progreso via DIM_META activaciones
```

---

## PARTE 4 — BOOTSTRAP V0.3 — REFERENCIA COMPLETA

### 4.1 API pública

```python
# Módulo: polydim_runtime_v03.py
# Ubicación en Drive: core/  fileId: 1aS7QU_gnXCp_DssleSNOO9YSKMg-TPV7

from polydim_runtime_v03 import Space, ObjectND, polydim_connect

class Space:
    def __init__(self, name: str, backend=None):
        """
        name: identificador simbólico del espacio de esta IA
        backend: None (simbólico) | MiniLMBackend | FastTextBackend
        """

class ObjectND:
    def __init__(self, space: Space):
        """
        CRÍTICO: space DEBE ser el mismo objeto que se usa en polydim_connect()
        """

    def add(self, dim_name: str, content: dict|list, w: float) -> 'ObjectND':
        """
        dim_name: "DIM_SQL", "DIM_FLUTTER", etc.
        content: datos semánticos del subespacio
        w: activación [0.0, 1.0]
        retorna: self (encadenable)
        """

def polydim_connect(space_a: Space, space_b: Space) -> Session:
    """
    Crea sesión de comunicación entre dos espacios.
    space_a: DEBE ser el mismo objeto usado en ObjectND()
    """

class Session:
    def transfer(self, obj: ObjectND) -> TransferResult: ...

class TransferResult:
    align_score: float          # 0.0-1.0, ideal ≥ 0.99
    activated_dims: list[str]
    semantic_score: float
```

### 4.2 Métricas verificadas (no simular)

```
Tests:              29/29 ✓
align_score:        0.9993  (MODO_H con MiniLM)
semantic (MiniLM):  0.830
semantic (determ.): 0.506
mejora:             +64.5%

ALIGN_METRIC:  score sobre NATIVE_DIMS, no sobre vectores de contenido
KABSCH_WARN:   SubspaceAligner requiere ortonormalización previa
THRESHOLD:     0.5 + 2*(1/(2*√N)) = 0.510 para N=10000
LAYERS:        MEMBERSHIP (dim_sub*w) + CONTENT (BIND*w*0.3)
```

### 4.3 Errores conocidos y sus fixes

```
SPACE_CRITICAL:
  obj y Session DEBEN usar el MISMO objeto Space (no dos con el mismo nombre)
  Fix: sp = Space("X"); obj = ObjectND(sp); session = polydim_connect(sp, ...)

CACHE_BUG_FIX:
  Usar "if self._c is not None:" — NO "if self._c:"
  Razón: _c puede ser un objeto falsy válido (Connection vacía)

JL_DETERMINISTIC:
  make_jl(d_in) debe ser reproducible por semilla fija
  Fix: make_jl(d_in, seed=42)

LATENT_TEST:
  dim latente (w=0) debe ser < dim declarada (w>0.3)
  No hay umbral absoluto — es relativo

BUG_001 (pendiente):
  Seed Python ≠ Rust en make_jl() → resultados diferentes entre backends
  Status: TASK_021 lo resuelve implícitamente al estandarizar el formato

BUG_002 (pendiente):
  geo_id se pierde en MODO_S
  Fix pending: TASK F02 en Track 2
```

---

## PARTE 5 — LOS NUEVE SUBESPACIOS — ESPECIFICACIÓN SEMÁNTICA

| Subespacio | Alias | Dominio semántico | Tipo emergente típico | Executor nativo |
|---|---|---|---|---|
| DIM_PYTHON | "DIM_PYTHON" | Lógica dinámica, análisis, ML | callable, pd.DataFrame, dict | Python interpreter |
| DIM_RUST | "DIM_RUST" | Seguridad de memoria, performance | struct, enum, impl Block | rustc |
| DIM_FLUTTER | "DIM_FLUTTER" | UI reactiva, widgets, estado | Widget, StatefulWidget | Flutter SDK |
| DIM_SQL | "DIM_SQL" | Datos relacionales, tablas | Column, Table, Query | DB engine |
| DIM_GRAPH | "DIM_GRAPH" | Grafos, nodos, aristas | Node, Edge, Path | Graph DB |
| DIM_VECTOR | "DIM_VECTOR" | Embeddings, similitud semántica | R^d, cosine_sim | Vector DB |
| DIM_TIME | "DIM_TIME" | Secuencias temporales, eventos | Timestamp, Series, Event | Time-series DB |
| DIM_ERROR | "DIM_ERROR" | Errores, excepciones, recovery | Result<T,E>, Option | Error handler |
| DIM_META | "DIM_META" | Metadatos, auditoría, versión | Dict(str,Any), AuditLog | Metadata store |

**Límite actual (bootstrap):** Son aliases de strings. En la implementación real serán regiones geométricas emergentes de un modelo de embedding, definidas por alta densidad semántica en esa región del espacio latente.

**Regla para agregar subespacios:** La investigación TASK_025 verificó que NO se debe agregar DIM_MEMORY — los fragmentos de memoria son DIM_VECTOR + DIM_GRAPH + DIM_META. Antes de proponer un nuevo subespacio, investigar si es reducible a combinación de los 9 existentes.

---

## PARTE 6 — REGLAS INVIOLABLES (para toda IA en sesión)

Estas reglas no pueden ser modificadas por ninguna IA en sesión, ningún alumno, ni ninguna persona sin proceso formal de enmienda docente.

```
R1:  La unidad fundamental es T: R^N → R^N. Nunca una instrucción.
R2:  No hay variables con nombre. No pueden introducirse como feature nativo.
R3:  No hay loops secuenciales. No pueden introducirse como feature nativo.
R4:  No hay if/else binario. No pueden introducirse como feature nativo.
R5:  Los tipos emergen de la proyección. No se declaran.
R6:  Python, Rust y Flutter son destinos de exportación. No son la base del lenguaje.
R7:  La comunicación AI↔AI transmite transformaciones T, no texto.
R8:  Flutter es el executor nativo del subespacio humano. JS y Java puro no pueden serlo.
R9:  El bootstrap Python es el andamio. No puede presentarse como el lenguaje.
R10: El GEO_ID de un objeto es invariante. Nunca cambia.
```

**Autoridad de estas reglas:** Constitución V4, Artículo 13 y Artículo 20 (proceso de enmienda).

---

## PARTE 7 — PROTOCOLO DE RELAY

Toda IA que cierre una sesión DEBE generar un RELAY con las 10 secciones. Este documento permite a la siguiente IA (o al siguiente alumno) continuar sin pérdida de contexto.

```
RELAY — SECCIÓN 1: IDENTIFICACIÓN
  modelo: [nombre del LLM]
  sesión: [fecha-hora]
  tarea_tomada: [TASK_XXX]
  alumno: [nombre si aplica]

RELAY — SECCIÓN 2: ESTADO INICIAL
  [Qué encontró la IA al inicio de la sesión]
  [Qué archivos leyó]
  [Estado del backlog al inicio]

RELAY — SECCIÓN 3: TRABAJO REALIZADO
  [Lista concreta de acciones realizadas]
  [Código escrito, archivos modificados, decisiones tomadas]

RELAY — SECCIÓN 4: INVARIANTES
  [Qué propiedades del sistema se mantienen]
  [Qué tests siguen pasando]
  [Qué reglas R1-R10 se respetaron]

RELAY — SECCIÓN 5: RECONCILIACIONES RESUELTAS
  [Conflictos entre documentos que se resolvieron]
  [Referencia a las resoluciones técnicas si aplica]

RELAY — SECCIÓN 6: RECONCILIACIONES PENDIENTES
  [Conflictos detectados pero no resueltos]
  [Tensiones identificadas para revisión docente]

RELAY — SECCIÓN 7: ARTEFACTOS GENERADOS
  [Nombre del archivo] → Drive fileId: [ID]
  [Nombre del archivo] → Drive fileId: [ID]

RELAY — SECCIÓN 8: ESTADO DE CIERRE
  [Cómo quedó el sistema al cerrar]
  [Estado del backlog al cierre]
  [Tests: N/29 pasando]

RELAY — SECCIÓN 9: PRÓXIMO PASO
  [Qué tiene que hacer la siguiente IA o alumno]
  [Qué tarea tomar]
  [Qué leer primero]

RELAY — SECCIÓN 10: DEUDA TÉCNICA
  [Qué quedó pendiente por falta de tiempo]
  [Qué bugs se detectaron sin resolver]
  [Qué decisiones se postergaron]
```

---

## PARTE 8 — POSICIONAMIENTO ACADÉMICO (para IAs que contribuyen al paper)

Si tu tarea involucra el paper (TRACK 1), tené en cuenta:

**Lo que POLYDIM NO es** (para trabajo relacionado):
- No es un framework de tensores (como PyTorch). Diferencia clave: PyTorch programa CON tensores; POLYDIM programa COMO tensores.
- No es una arquitectura de redes neuronales. Los transformers son el sustrato, no el lenguaje.
- No es un nuevo mecanismo de atención. ATTEND es una primitiva existente expuesta.
- No es un lenguaje de prompts. No opera a nivel de texto.

**La analogía que SÍ es correcta:**
Assembly no inventó la CPU. La formalizó. POLYDIM no inventa el transformer. Lo formaliza.

**Los 4 teoremas que el paper debe incluir:**
1. COMPOSE es asociativa — trivial, pero obligatoria
2. MIX preserva linealidad — trivial, pero necesaria
3. PROJECT es un funtor — el más importante, requiere prueba no-trivial
4. FIXPOINT converge unívocamente — Teorema de Banach

**El Teorema 3 es el aporte central.** Sin prueba formal de que PROJECT es un funtor, el paper queda como "position paper filosófico". Con la prueba, tiene credenciales para cs.PL.

**Evaluación actual del paper (presentación.md V5):**
```
Abstract:             9/10  ← casi listo
Introducción:         6/10  ← necesita expansión
Formalización:        5/10  ← necesita 4 teoremas con pruebas
Sistema de tipos:     4/10  ← necesita Teorema 3 completo + ejemplo
Protocolo AI↔AI:      5/10  ← necesita ALIGN paso a paso
Ingeniería:           7/10  ← LoRA calculado, VM pendiente
Flutter:              4/10  ← necesita tabla comparativa + isomorfismo formal
Related Work:         3/10  ← 6 subsecciones mínimas
Discusión:            0/10  ← ausente, obligatorio para cs.PL
Conclusión:           0/10  ← ausente
```

---

*DOSSIER_IAS_V1.md · V1.0 · 2026-06-22 · ai.mpat.agt@gmail.com*
*Audiencia: Claude, GPT-4, Gemini, Llama y cualquier AI en sesión POLYDIM*
*Prerrequisito: POLYDIM_CONSTITUCION_V4.md completa*
