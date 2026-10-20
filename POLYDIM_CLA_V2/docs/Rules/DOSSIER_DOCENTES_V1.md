# POLYDIM_DEST
# destino: polydim/docs/
# nombre:  DOSSIER_DOCENTES_V1.md
# autor:   ai.mpat.agt@gmail.com
# fecha:   2026-06-21

---

# POLYDIM — Dossier para Docentes V1
## Guía completa de supervisión, evaluación y dirección del proyecto

**Audiencia:** Docentes, directores de tesis, investigadores senior
**Fuentes:** CONSTITUCIÓN_V2, PAPER_ESQUELETO_V1, MEJORAS_CHATGPT, MEJORAS_GEMINI, SÍNTESIS_V1
**Nivel:** Técnico-académico

---

## SECCIÓN 1 — VISIÓN GENERAL DEL PROYECTO

### 1.1 Qué es POLYDIM

POLYDIM es un lenguaje de programación cuya unidad fundamental no es la instrucción sino la transformación geométrica:

```
T : R^N → R^N
```

La motivación es atacar el "impedance mismatch" entre cómo piensan los transformers (en geometría multidimensional) y cómo se los programa actualmente (con secuencias de instrucciones diseñadas para humanos).

La analogía histórica que el proyecto usa para posicionarse académicamente:

```
1950s:  la CPU ya hacía aritmética binaria
        → el ensamblador formalizó cómo programarla

2020s:  los transformers ya operan en geometría multidimensional
        → POLYDIM formaliza cómo programarlos deliberadamente
```

Esta analogía es atractiva académicamente y bien recibida por los revisores externos (ChatGPT y Gemini la aceptan sin cuestionarla). El riesgo es que quede como metáfora sin formalización. El trabajo de los docentes es asegurarse de que la metáfora se convierta en matemática.

### 1.2 Estado actual del proyecto (2026-06-21)

| Componente | Estado | Nivel |
|---|---|---|
| Constitución V2 | VIGENTE y ratificada | Filosófico + especificación |
| Bootstrap Python V0.3 | 29/29 tests pasando | Prueba de concepto |
| align_score (MODO_H) | 0.9993 | Métrica verificada |
| Semántica MiniLM | 0.830 vs 0.506 determinístico | +64.5% sobre baseline |
| Paper esqueleto V1 | ~40% del contenido disponible | Posición académica |
| Revisiones externas | ChatGPT (15 prioridades) + Gemini (5+ recomendaciones) | Validación parcial |
| VM mínima | No existe | Pendiente |
| Formato .polydim binario | Especificado, no implementado | Pendiente |
| PROJECT(DIM_FLUTTER) | No existe | Pendiente |

### 1.3 Los tres usos inseparables (según Constitución V2)

1. **Comunicación AI↔AI:** Dos IAs intercambian transformaciones T, no mensajes de texto.
2. **Comunicación AI↔Humano:** Una IA proyecta su estado geométrico a un widget Flutter nativo.
3. **Ejecución multiplataforma:** Un único `.polydim` se ejecuta en cualquier plataforma.

---

## SECCIÓN 2 — MAPA CONCEPTUAL DEL LENGUAJE

### 2.1 Lo que no existe en POLYDIM (ausencias definitorias)

Esto es central para evaluar si un alumno entendió el lenguaje:

| Concepto ausente | Por qué | Reemplazo |
|---|---|---|
| Variables con nombre | Hay posiciones en R^N | GEO_ID + activaciones |
| Loop for/while | Hay convergencia a punto fijo | T^n(v) hasta ‖s_{k+1}-s_k‖ < ε |
| if/else binario | Hay gate continuo | α·T₁ + (1-α)·T₂ |
| Tipo declarado | El tipo emerge de la proyección | PROJECT(P, DIM_SQL) → tipo SQL |
| Instrucción secuencial | Hay composición algebraica | T₂ ∘ T₁ |
| Mensaje de texto AI↔AI | Hay transmisión de T | T aplicado a estado receptor |

### 2.2 Las cuatro primitivas

#### ATTEND(Q, K, V, s)
```
ATTEND(WQ, WK, WV, s) = softmax( (sWQ)(sWK)ᵀ / √d ) · (sWV)
```
Relevancia contextual. Mapea directamente al mecanismo de atención de un transformer. Es la primitiva más específica a la arquitectura actual.

**Nota docente:** ChatGPT señala que ATTEND está acoplada a los transformers, lo cual crea fragilidad arquitectural si los transformers son reemplazados. Ver Tensión 1 más adelante.

#### COMPOSE(T₁, T₂)
```
COMPOSE(T₁, T₂) = T₂ ∘ T₁
```
Composición algebraica. No es "primero T₁, luego T₂" — es una única transformación. No es conmutativa: T₂ ∘ T₁ ≠ T₁ ∘ T₂. Esta no-conmutatividad es el mecanismo por el que el orden causal se codifica topológicamente.

#### MIX(α, T₁, β, T₂)
```
MIX(α, T₁, β, T₂) = α·T₁ + β·T₂
```
Superposición continua. Reemplaza el if/else binario. **Por qué funciona:** En espacios de alta dimensionalidad (R^10000), dos vectores aleatorios son casi ortogonales con alta probabilidad (propiedad VSA). Esto permite que α·T₁ + β·T₂ ejecute ambas "ramas" simultáneamente sin interferencia destructiva. La "elección" ocurre en PROJECT, donde el executor colapsa el estado a lo largo del eje dominante.

#### PROJECT(T, executor)
```
PROJECT(T, DIM_FLUTTER)  →  widget nativo
PROJECT(T, DIM_RUST)     →  código Rust
PROJECT(T, DIM_SQL)      →  query SQL
PROJECT(T, DIM_PYTHON)   →  código Python
```
El compilador y el intérprete de POLYDIM. Formalmente, es un **funtor** entre categorías:
```
PROJECT : C_POLYDIM → D_executor
```
Donde C_POLYDIM es la categoría de posiciones y transformaciones, y D_executor es la categoría de tipos del executor.

### 2.3 Los nueve subespacios nativos

```
DIM_PYTHON    lógica dinámica, análisis, ML, scripts
DIM_RUST      seguridad de memoria, performance, ownership
DIM_FLUTTER   UI reactiva, widgets, estado (executor humano nativo)
DIM_SQL       datos relacionales, tablas, constraints
DIM_GRAPH     grafos, nodos, aristas, relaciones
DIM_VECTOR    embeddings, similitud semántica, VSA
DIM_TIME      secuencias temporales, eventos, orden
DIM_ERROR     estados de error, excepciones, recuperación
DIM_META      metadatos, auditoría, versión, origen
```

**Límite actual:** Son aliases de strings en el bootstrap. En el lenguaje real serán regiones emergentes de un modelo de embedding real. Los aliases son nombres temporales.

**Nota docente:** La investigación TASK_025 verificó que DIM_MEMORY NO debe agregarse. Los fragmentos MPAT5 son DIM_VECTOR + DIM_GRAPH + DIM_META. No agregar subespacios nuevos sin investigación análoga.

### 2.4 El sistema de tipos dimensional

```
Posición P con activaciones:
  DIM_SQL     → 0.9
  DIM_FLUTTER → 0.85
  DIM_VECTOR  → 0.9
  DIM_GRAPH   → 0.6

PROJECT(P, DIM_SQL)     →  Column(Integer)
PROJECT(P, DIM_FLUTTER) →  ListView
PROJECT(P, DIM_VECTOR)  →  R^3 = [1.0, 2.0, 3.0]
PROJECT(P, DIM_GRAPH)   →  Path(nodes)
```

**El mismo objeto. Cuatro tipos simultáneos. Sin conversión.**

Intersección de subespacios:
```
PROJECT(P, DIM_SQL ∩ DIM_FLUTTER)
```
Produce un widget enlazado en tiempo real a su fuente de datos. Este tipo no existe en ningún lenguaje humano.

Formalmente, la intersección debe modelarse como un **pullback categórico**: garantiza que las actualizaciones en SQL y en Flutter provienen de la misma raíz geométrica, sin código de glue ni data bindings.

---

## SECCIÓN 3 — LAS SEIS TENSIONES Y SUS RESOLUCIONES

Estas tensiones identificadas en la Síntesis V1 son los puntos de decisión más importantes del proyecto. El docente debe guiar al equipo en cómo resolverlas.

---

### TENSIÓN 1 — El nivel de ATTEND como primitiva

**La tensión:** ATTEND está en la misma capa que COMPOSE, MIX y PROJECT. Pero ATTEND está acoplada a la arquitectura transformer. Las otras tres son álgebra pura.

**Crítica de ChatGPT (P2):** Si dentro de 20 años los transformers son reemplazados, POLYDIM no debería colapsar con ellos.

**Resolución recomendada para el docente:** Proponer enmienda al Art. IV de la Constitución V2 que formalice dos capas sin violar R1-R10:

```
CAPA ALGEBRAICA (fundamento permanente):
  COMPOSE(T₁, T₂)           = T₂ ∘ T₁
  MIX(α, T₁, β, T₂)         = α·T₁ + β·T₂
  FIXPOINT(T, ε)             = T^n(v) hasta ‖s_{k+1}-s_k‖ < ε
  PROJECT(T, executor)       = proyección al subespacio del executor

CAPA DE IMPLEMENTACIÓN (específica a arquitectura):
  ATTEND(Q, K, V, s)         = softmax(QKᵀ/√d)·V   [transformer]
  RECUR(A, B, C, h, x)       = A·h+B·x, C·h         [Mamba/SSM]
```

ATTEND no desaparece — se reubica. FIXPOINT se añade como primitiva algebraica explícita.

**Acción docente:** Convocar proceso formal de enmienda al Art. IV. Requiere votación y registro en Constitución.

---

### TENSIÓN 2 — Flutter: exclusividad vs referencia canónica

**La tensión:** La Constitución V2 Art. VIII y Regla R8 establecen que "JS y Java no pueden ocupar el rol de executor nativo". ChatGPT (P3) señala que esto es académicamente difícil de sostener.

**La justificación matemática de V2 que sí es correcta:** El árbol de widgets de Flutter mapea algebraicamente:
```dart
Column( children: [ Text("x"), Button(acción) ] )
```
Esto es `T_columna ∘ ( T_texto ⊕ T_botón )`. La estructura de Flutter es isomorfa al álgebra de composición de POLYDIM. Esta isomorfía ES real y verificable.

**La debilidad:** Los términos absolutos "incompatible" son más difíciles de defender que términos de grado.

**Resolución recomendada para el docente:**

```
Flutter: executor nativo — isomorfismo algebraico directo
React / SwiftUI / Qt: executors externos — requieren capa de adaptación
JS puro: no puede ser executor nativo (fragmentación de ecosistema impide garantizar comportamiento)
Java puro: no puede ser executor nativo (modelo OOP contradice composición geométrica)
```

**Acción docente:** Revisar paper Sec. 5.3 para reformular con lenguaje académico de grado, no de exclusión binaria.

---

### TENSIÓN 3 — El formato .polydim y el problema N×N

**La tensión:** La Constitución V2 Art. IX especifica:
```
[ TRANSFORMS ]  float32[N×N]
```
Para N=10000: 10000×10000 = 100 millones de float32 ≈ 400 MB **por transformación**. Esto es inviable.

**Crítica de ChatGPT (P6):** Este diseño impide escalar. La solución son representaciones comprimidas.

**Variantes de representación (ver DOSSIER_MEJORAS_IMPLEMENTABLES_V1.md para detalle completo):**

| Variante | Representación | Tamaño para N=10000, r=64 | Caso de uso |
|---|---|---|---|
| Low-rank | T = UVᵀ, U,V ∈ R^{N×r} | 2×64×10000×4B ≈ 5 MB | Caso general |
| LoRA | T = W₀ + UVᵀ | 5 MB + overhead | Fine-tuning style |
| Sparse | índices + valores ≠ 0 | Variable | Subespacios locales |
| Tensor train | T = G₁ × G₂ × ... × Gₖ | ~MB | Estructuras jerárquicas |
| Graph of transforms | DAG de transformaciones | Variable | Programas complejos |

**Recomendación docente:** Actualizar Art. IX de la Constitución V2 para especificar representación low-rank como default. Generar TASK_021b: especificación binaria actualizada.

**Acción docente:** Esto es URGENTE. Bloquea TASK_021, TASK_023 (polydim_core.rs), y la VM.

---

### TENSIÓN 4 — La semántica formal: dónde está el vacío

**La tensión:** POLYDIM tiene filosofía y Constitución pero no tiene **semántica operacional formal**. Esto impide que sea un lenguaje en el sentido académico de cs.PL.

**Lo que falta:**

Definiciones formales (no existen aún):
```
S = (V, D, A) donde:
  V ∈ R^N          el vector de estado
  D = {DIM_i}      conjunto de subespacios
  A = {αi ∈ [0,1]} activaciones

P = (GEO_ID, V)    posición con identidad invariante

T : S → S          transformación del estado completo
```

Reglas de evaluación (no existen):
```
⟨s, T⟩ ⇓ s'                  (evaluación general)

⟨s, COMPOSE(T₁,T₂)⟩ ⇓ s₂
  cuando ⟨s, T₁⟩ ⇓ s₁  y  ⟨s₁, T₂⟩ ⇓ s₂

⟨s, MIX(α,T₁,β,T₂)⟩ ⇓ α·T₁(s) + β·T₂(s)

⟨s, FIXPOINT(T,ε)⟩ ⇓ s*
  cuando ‖T(s*) - s*‖ < ε
```

**Acción docente:** Asignar a un alumno (o a un AI de TRACK 1) la tarea de redactar el artículo de semántica formal (Sec. 3.2 del paper). Esto es la Prioridad 1 del paper según ChatGPT y Gemini.

---

### TENSIÓN 5 — El GEO_ID: de intuición a invariante formal

**La tensión:** GEO_ID está definido en todos los documentos como "hipervector base invariante que nunca cambia". Pero esta invariancia no está formalizada matemáticamente.

**Formalización propuesta:**

```
Sea T el grupo de transformaciones POLYDIM permitidas.
GEO_ID es invariante respecto a T si:
  ∀T ∈ T_permitidas : dist(T(GEO_ID), GEO_ID) < ε

O más fuerte:
  Inv(T, GEO_ID) ≡ T(GEO_ID) = GEO_ID
```

**Extensión filosófica (ChatGPT, secciones XXXVI-XLVIII):** La pregunta más profunda es si GEO_ID puede entenderse como una clase de equivalencia homotópica en el sentido de HoTT (Homotopy Type Theory). Si dos caminos en el espacio semántico son equivalentes (llevan al mismo significado), representan el mismo GEO_ID. Esta perspectiva conecta POLYDIM con:
- **Grothendieck:** haces, toposes, geometría de observadores
- **Lawvere:** categorías como fundamento de la lógica
- **Voevodsky / HoTT:** identidad como camino, tipos como espacios

**Decisión docente:** Esta extensión debe ir en Sec. 8.3 (Open Problems), no en el abstract. Es demasiado especulativa para el paper inicial pero debe documentarse para evitar que se pierda.

---

### TENSIÓN 6 — El protocolo AI↔AI: la realidad de los espacios distintos

**La tensión:** La Constitución V2 Art. VII dice que "IA_A transmite T a IA_B que la aplica a su estado". Pero GPT, Gemini, Claude y Llama tienen espacios latentes distintos (d₁ ≠ d₂, semánticas distintas).

**El protocolo ALIGN actual (bootstrap V0.3):** Implementa un score de alineación sobre los NATIVE_DIMS. Pero esto es simbólico — no es alineación de espacios vectoriales reales.

**Formalización del ALIGN real (Gemini):**
```
IA_A tiene espacio R^{d₁}, IA_B tiene espacio R^{d₂}
ALIGN calcula M : R^{d₁} → R^{d₂}  (rotación ortogonal)

IA_A transmite T_A
IA_B recibe y aplica:  T_B = M · T_A · M⁻¹
```

**Métodos posibles para M:**
- **Procrustes:** minimiza ‖M·A - B‖² sujeto a M ortogonal
- **CCA (Canonical Correlation Analysis):** maximiza correlación entre proyecciones
- **Hipervectores compartidos:** GEO_IDs universales como "anclas" de navegación
- **Codebook:** espacio universal compartido con proyecciones locales

**Acción docente:** Esto bloquea la demostración del protocolo AI↔AI real. Asignar como tarea de investigación: implementar ALIGN con Procrustes usando embeddings reales de dos modelos.

---

## SECCIÓN 4 — TABLA DE DECISIONES PENDIENTES

Esta tabla define las decisiones activas del proyecto. El docente debe aprobarlas formalmente.

| # | Decisión | Urgencia | Bloquea |
|---|---|---|---|
| D01 | Separar ATTEND en CAPA_ALGEBRAICA vs CAPA_IMPL en Art. IV | Media | Paper 3.2, longevidad |
| D02 | Reformular Art. VIII: "executor nativo" vs "executor externo" | Media | Paper 5.3 |
| D03 | Actualizar Art. IX: low-rank en lugar de float32[N×N] | **ALTA** | TASK_021, 023, VM |
| D04 | Definir PROJECT como funtor formalmente (Teorema 3) | **ALTA** | Paper 3.3 |
| D05 | Decidir si GEO_ID se formaliza con nivel homotópico | Baja | Paper 8.3 |
| D06 | Registrar bifurcación filosófica en research/ | Baja | Track Teoría |
| D07 | Integrar texto LaTeX de Gemini (3.5, 4.3, 5.2, 9) al paper | **ALTA** | Paper completo |
| D08 | Definir MIX con justificación VSA (ortogonalidad en alta dim) | **ALTA** | Paper 3.2 |
| D09 | Formalizar GEO_ID con invariante explícito | Media | Paper (Teorema 5) |
| D10 | Implementar ALIGN real (Procrustes / CCA) | Media | Paper 4.2, demo |

---

## SECCIÓN 5 — EVALUACIÓN DE LA CALIDAD DEL PROYECTO

### 5.1 Rúbrica de evaluación según revisores externos

| Aspecto | ChatGPT (hoy) | Gemini (implied) | Para 9/10 |
|---|---|---|---|
| Originalidad | 9.5/10 | ~9/10 | Ya está |
| Coherencia interna | 9/10 | 9/10 | Ya está |
| Formalización matemática | **6/10** | 7/10 | Ver TASK_A |
| Implementabilidad | **5/10** | 6/10 | Ver TASK_B,C |
| Potencial académico | 8.5/10 | 9/10 | Casi |
| Potencial práctico | 7/10 | 7.5/10 | Requiere VM |

### 5.2 Los cuatro teoremas mínimos para el paper

El paper no puede publicarse en cs.PL sin teoremas formales. Los cuatro mínimos:

**Teorema 1 — Asociatividad de COMPOSE:**
```
(T₃ ∘ T₂) ∘ T₁ = T₃ ∘ (T₂ ∘ T₁)
Prueba: propiedad estándar de composición de funciones.
```

**Teorema 2 — Linealidad de MIX:**
```
Si T₁, T₂ son transformaciones lineales,
entonces MIX(α, T₁, β, T₂) = α·T₁ + β·T₂ es también lineal.
Prueba: por definición de combinación lineal.
```

**Teorema 3 — PROJECT es un funtor (el más importante):**
```
PROJECT preserva composición:
  PROJECT(COMPOSE(T₁,T₂), E) = COMPOSE(PROJECT(T₁,E), PROJECT(T₂,E))
PROJECT preserva identidad:
  PROJECT(id_C, E) = id_E
```
Con esto PROJECT deja de ser metáfora. Es matemática.

**Teorema 4 — Unicidad del punto fijo:**
```
Si T es una contracción (‖T(u)-T(v)‖ ≤ k·‖u-v‖, k<1),
entonces ∃! punto fijo (Teorema de Banach).
Condición: los programas POLYDIM que "reemplazan loops" deben ser contracciones.
```

**Evaluación docente de los teoremas:** Verificar que los alumnos demuestren Teorema 3 con rigor. Es el aporte matemático más significativo del paper.

### 5.3 La estrategia de cuatro tareas para 9/10

Según análisis de ChatGPT:

**TASK_A: Semántica operacional completa**
- Publicar definiciones formales de S, P, T, D, A
- Escribir reglas de evaluación para las cuatro primitivas
- Demostrar los cuatro teoremas

**TASK_B: Sistema de tipos categórico demostrable**
- Formalizar PROJECT como funtor
- Demostrar Teorema 3
- Formalizar intersección dimensional como pullback

**TASK_C: Formato .polydim basado en LoRA**
- Cambiar float32[N×N] → (U, V) donde T = UVᵀ
- Especificar el header binario actualizado
- Implementar el loader/parser mínimo

**TASK_D: VM mínima ejecutando COMPOSE/MIX/PROJECT**
- Sin Flutter, sin SQL, sin AI↔AI
- Solo: estado → transformación → nuevo estado
- Implementar en Python inicialmente, migrar a Rust

---

## SECCIÓN 6 — ESTRUCTURA DE TRACKS

### 6.1 Tracks del proyecto

```
TRACK 1 — PAPER (cs.PL + cs.AI)
  Owner: alumnos + docentes
  Output: paper arXiv publicado
  Bloquea: publicación
  Prioridad del paper: 3.2 → 3.3 → 4.3 → 7 → Abstract

TRACK 2 — MÉTRICAS Y TESTS
  Owner: alumnos ingeniería
  Output: suite de tests extendida
  Depende de: Bootstrap V0.3 (completo)

TRACK 3 — VM MÍNIMA
  Owner: alumnos sistemas/Rust
  Output: .polydim → loader → IR → runtime
  Depende de: TASK_C (formato .polydim actualizado)

TRACK 4 — SELF-HOSTING PARCIAL
  Owner: alumnos avanzados
  Output: POLYDIM genera POLYDIM (parcial)
  Depende de: TRACK 3 completo
```

### 6.2 El papel del bootstrap V0.3

El bootstrap NO es el lenguaje. Es el andamio. Métricas verificadas:

```
Tests:                  29/29 ✓
align_score (MODO_H):   0.9993
MiniLM semantic:        0.830
Deterministic:          0.506
Mejora real:            +64.5%
```

API actual (3 líneas):
```python
from polydim_runtime_v03 import Space, ObjectND, polydim_connect
sp  = Space("MI_IA")
obj = ObjectND(sp).add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)
dims = polydim_connect(sp, Space("OTRA_IA")).transfer(obj)
```

**Error crítico a vigilar:** obj y Session deben usar el MISMO objeto Space. "if self._c is not None:" NOT "if self._c:".

---

## SECCIÓN 7 — COMPARATIVA CON TRABAJO RELACIONADO

Esta sección es crucial para el paper. El docente debe guiar a los alumnos en entender por qué POLYDIM no es simplemente "otro framework de tensores".

### 7.1 POLYDIM vs PyTorch / JAX / TensorFlow

| Característica | PyTorch / JAX | POLYDIM |
|---|---|---|
| Rol de los tensores | Son los datos que fluyen a través de un programa | Son el programa en sí mismo |
| Flujo de control | Secuencial / Externo (Python for, if/else) | Geométrico / Interno (Convergencia, Superposición) |
| Interacción humana | Logs, TensorBoard, UIs externas acopladas con REST | PROJECT(DIM_FLUTTER): estado geométrico genera UI nativamente |
| Paradigma | Von Neumann (control secuencial) | Geométrico (transformación del espacio) |

### 7.2 POLYDIM vs MCP (Model Context Protocol)

MCP sigue dependiendo de serialización de texto estructurado (JSON-RPC). Sufre el "impedance mismatch" que POLYDIM erradica:

```
MCP:     IA_A → JSON → [serialización] → JSON → IA_B
POLYDIM: IA_A → T   → [transmisión]   → T    → IA_B
         (T aplicada directamente al estado de IA_B)
```

### 7.3 POLYDIM vs A2A (Google Agent-to-Agent)

A2A también usa serialización de texto estructurado. No resuelve el problema del espacio latente.

### 7.4 POLYDIM vs FIPA (lenguajes de agentes clásicos)

FIPA define una semántica de actos de habla. POLYDIM no tiene lenguaje: tiene geometría.

---

## SECCIÓN 8 — PROBLEMAS ABIERTOS Y DIRECCIÓN DE INVESTIGACIÓN

### 8.1 Problemas técnicos activos (backlog)

```
PEND_001:  formato binario de transmisión (.polydim con low-rank)
PEND_002:  inferencia automática ponderada (TASK_022)
PEND_003:  consistencia distribuida cuando múltiples IAs modifican el mismo objeto
PEND_008:  detección de deriva semántica en sesión
PEND_010:  ALIGN para N distintos entre AIs (d₁ ≠ d₂)
```

### 8.2 Problemas de investigación de largo plazo

**¿Cómo emergen los subespacios de un modelo de embedding real sin definirlos manualmente?**
Opción técnica: Vector Quantization (VQ-VAE), Self-Organizing Maps (SOM), clustering jerárquico del espacio de activaciones.

**¿Consistencia distribuida entre AIs?**
Analogía: CRDTs (Conflict-free Replicated Data Types) adaptados a topología vectorial (Geometric-CRDTs donde las mutaciones convergen por promedio ponderado o proyección a subespacios comunes).

**¿Costo computacional de PROJECT en tiempo real?**
Para PROJECT(DIM_FLUTTER), el costo depende de la representación de T. Con low-rank: O(N·r) donde r << N.

**¿POLYDIM self-hosting?**
Meta de largo plazo: el runtime se compila a sí mismo. Indicador: COMPOSE/MIX/PROJECT generados por el propio runtime.

### 8.3 La bifurcación filosófica (para futura investigación)

ChatGPT plantea una hipótesis profunda: ¿existe una geometría del significado independiente de las representaciones, donde los programas sean caminos, las identidades sean clases homotópicas y los tipos sean observaciones locales?

Si esta geometría existe, entonces:
- Las 9 dimensiones de POLYDIM son **observadores**, no coordenadas
- Los tipos son **vecindades topológicas**, no etiquetas
- La igualdad es un **camino** (HoTT), no una relación binaria
- PROJECT es un **funtor de olvido**, no un compilador
- R^N es un **atlas específico** del espacio semántico, no el espacio mismo

Esta perspectiva conecta POLYDIM con Grothendieck (haces, toposes), Lawvere (categorías como fundamento) y Voevodsky/HoTT (identidad como camino).

**Decisión pedagógica:** Registrar en research/ pero no incluir en el paper actual. Si resulta correcta, modifica retroactivamente el significado de todo el lenguaje.

---

## SECCIÓN 9 — GUÍA DE SUPERVISIÓN

### 9.1 Cómo evaluar el trabajo de un alumno en POLYDIM

**Indicadores de comprensión correcta:**
- Describe T: R^N → R^N como unidad fundamental (no instrucción)
- Explica las cuatro ausencias definitorias (variables, loops, if, tipos declarados)
- Distingue claramente bootstrap de lenguaje real
- Puede derivar por qué MIX reemplaza if/else (ortogonalidad VSA)
- Entiende por qué PROJECT es un funtor (no solo una función)

**Indicadores de confusión frecuente:**
- Confundir la API de Python (`.add()`, `.transfer()`) con sintaxis de POLYDIM
- Tratar los subespacios como types Python en lugar de regiones geométricas
- Pensar que COMPOSE es "primero T₁, luego T₂" en vez de un único operador
- No entender la diferencia entre GEO_ID (invariante) y activaciones (mutables)

### 9.2 Cómo evaluar una implementación

**Criterios mínimos para TRACK 2:**
1. Tests existentes pasan: 29/29
2. align_score ≥ 0.99 en MODO_H
3. GEO_ID se mantiene invariante tras mutaciones

**Criterios mínimos para TRACK 3 (VM):**
1. Un estado s ∈ R^N puede evolucionar bajo T
2. COMPOSE de dos T produce resultado algebraicamente correcto
3. FIXPOINT converge bajo T contractiva

### 9.3 Cómo evaluar una sección del paper

Usar el mapa de secciones de SINTESIS_V1 Parte 5. Cada sección tiene estado (Pendiente / Esqueleto / Parcial / Disponible). Para que una sección sea "lista para revisión", debe:
1. Tener formalización matemática (no solo descripción)
2. Citar las referencias correctas
3. No contener afirmaciones sin demostración o sin citation

---

## SECCIÓN 10 — REGLAS INVIOLABLES (REFERENCIA DOCENTE)

Estas reglas no pueden modificarse por alumnos ni por IAs en sesión. Solo por proceso formal de enmienda con aprobación del docente.

| Regla | Contenido |
|---|---|
| R1 | La unidad fundamental es T: R^N → R^N. Nunca una instrucción. |
| R2 | No hay variables con nombre. No pueden introducirse como feature nativo. |
| R3 | No hay loops secuenciales. No pueden introducirse como feature nativo. |
| R4 | No hay if/else binario. No pueden introducirse como feature nativo. |
| R5 | Los tipos emergen de la proyección. No se declaran. |
| R6 | Python, Rust y Flutter son destinos de exportación. No son la base. |
| R7 | La comunicación AI↔AI transmite transformaciones T, no texto. |
| R8 | Flutter es el executor del subespacio humano. JS y Java no pueden ocupar ese rol de forma nativa. |
| R9 | El bootstrap Python es el andamio. No puede presentarse como el lenguaje. |
| R10 | El GEO_ID de un objeto es invariante. Nunca cambia. |

---

*DOSSIER_DOCENTES_V1.md · V1.0 · 2026-06-21 · ai.mpat.agt@gmail.com*
*Basado en: CONSTITUCIÓN_V2, PAPER_ESQUELETO_V1, MEJORAS_CHATGPT, MEJORAS_GEMINI, SÍNTESIS_V1*
*Audiencia: Docentes, directores, investigadores senior*
