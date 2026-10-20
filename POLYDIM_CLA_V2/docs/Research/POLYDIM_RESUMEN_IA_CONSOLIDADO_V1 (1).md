# POLYDIM_DEST
# destination: POLYDIM_AI_SUMMARIES/
# filename: POLYDIM_RESUMEN_IA_CONSOLIDADO_V1.md
# author: claude (resumen automatizado, sesión 2026-07-02)
# fuentes: carpeta_docs (1pYn8jFoPgTTPH0Z57bVuIZJhf4FoRuSR) + carpeta_paper (1czWYt3kk8qseDOxJdLlEeyzsn2-ai4hM)
# nota: consolida y complementa POLYDIM_RESUMEN_IA.md / POLYDIM_KNOWLEDGE_BASE_AI.md /
#       POLYDIM_AI_KNOWLEDGE.json ya presentes en esta carpeta (no los reemplaza a la fuerza,
#       agrega lo que faltaba: ley de sandboxing, discrepancia de catálogo DIM_*, reglas R1-R17,
#       teoremas, y aviso de gobernanza).

# POLYDIM — RESUMEN PARA CONSUMO DE AGENTES DE IA
(No optimizado para lectura humana. Objetivo: que un asistente de IA de cualquier alumno
entienda POLYDIM y su estado actual gastando el mínimo de tokens posible, sin tener que leer
los ~40 documentos originales de las dos carpetas Drive.)

> Generado: 2026-07-02. Idioma de origen mixto (EN académico / ES operativo) — este resumen
> unifica ambos. Autor del proyecto: Ariel H. Garcia Traba (Independent Researcher, Buenos
> Aires, ORCID 0009-0001-2787-6067, github.com/AGT1973).

---

## 0. QUÉ ES POLYDIM (una frase)

POLYDIM es un lenguaje de programación formal cuya unidad mínima de cómputo NO es una
instrucción secuencial de máquina de Von Neumann, sino una **transformación geométrica
T: R^N → R^N** (N ≈ 10.000), pensado para que las IAs (transformers, SSMs tipo Mamba)
programen y se comuniquen directamente en su propio espacio latente, sin serializar a texto.

**Tesis central (impedance mismatch / impedance gap):** los modelos "piensan" en miles de
dimensiones simultáneas pero se ven forzados a comunicarse por texto secuencial (tokens).
Dato citado del paper (Interlat, Du et al. 2025): un hidden state transporta ~40.000 bits de
información vs ~15 bits por token → brecha de ancho de banda semántico de ~2.667x; se
descarta >99,9% del contenido semántico en cada serialización. LatentMAS (ICML 2026
Spotlight) muestra que evitar esa serialización logra hasta ~24x de aceleración de inferencia.

Analogía oficial: "el ensamblador no inventó el transistor, lo hizo programable. POLYDIM no
inventa el transformer, lo hace programable."

---

## 1. LAS CUATRO PRIMITIVAS ALGEBRAICAS (núcleo invariante, independiente de arquitectura)

| Primitiva | Definición | Reemplaza a |
|---|---|---|
| `COMPOSE(T1,T2)` | = T2 ∘ T1 (no conmutativa, asociativa — Teorema 1) | secuencia de instrucciones |
| `MIX(α,T1,β,T2)` | = α·T1 + β·T2 (superposición continua, quasi-ortogonalidad VSA en alta dimensión) | if/else binario |
| `FIXPOINT(T,ε)` | itera T hasta converger (Banach, Teorema 4) | loops for/while |
| `PROJECT(T,executor)` | funtor covariante P: G → E_executor; es el "compilador" (Teorema 3) | tipos declarados / compilación |

Capa de implementación (específica de arquitectura, NO parte del núcleo algebraico —
separación explícita para que el lenguaje sobreviva a cambios de arquitectura, Art. XV):
- `ATTEND(Q,K,V,s) = softmax(Q·Kᵀ/√d)·V` — para Transformers.
- `RECUR(A,B,C,h,x)` — para SSMs tipo Mamba (linear state-space update).

`PROJECT` materializa en contratos: `COMPILE` (binarios Rust/WASM), `RENDER` (Flutter,
isomorfismo probado — único executor "nativo"; React/SwiftUI/Qt son externos, requieren
capa de adaptación), `EXPORT` (sistemas legacy no-tensoriales).

### Comunicación AI↔AI (sin texto)
```
IA_A: estado_A → genera T → transmite T
IA_B: recibe T → aplica T a estado_B → nuevo estado_B
Communication(sB, sA) = softmax(QB·KA^T/√d)·VA
```
Leer un mensaje = computar una capa de cross-attention. Protocolo `ALIGN` calcula una matriz
de proyección M (Procrustes ortogonal o CCA) contra un Codebook Universal de GEO_IDs
compartido, para que IAs con espacios latentes de dimensión distinta (d1≠d2, ej. GPT-4 vs
Llama-3) puedan entenderse: T_aplicado = M·T_A·M⁺ (pseudoinversa de Moore-Penrose).

---

## 2. LO QUE NO EXISTE EN POLYDIM (y qué lo reemplaza)

| No existe | Existe en su lugar |
|---|---|
| Variables con nombre | Posiciones en R^N con identidad geométrica (GEO_ID) |
| Funciones f(x) | Transformaciones T: R^N→R^N que se componen |
| Loop for/while | Convergencia a punto fijo T^n(v) |
| Condicional if/else | Gate continuo α·T1+(1−α)·T2 |
| Tipo declarado | Tipo emergente por proyección (el mismo objeto es SQL Column, widget Flutter, nodo de grafo, etc. a la vez) |
| Instrucción secuencial | Composición algebraica T2∘T1 |
| Mensaje de texto entre IAs | Transmisión directa de T |
| Memoria direccionable (`DIM_MEMORY`) | **Prohibida explícitamente** (Art. XVIII): el estado es posición, no almacenamiento |

`GEO_ID`: hipervector base único e invariante = identidad topológica del objeto, nunca
cambia. `Activaciones`: pesos continuos [0.0,1.0] por subespacio (mutan; no es reasignación,
es evolución geométrica). `Binding`: vector de relación direccional en R^N.

Formato de archivo `.polydim`: binario tensorial (header + GEO_ID + TRANSFORMS en low-rank
LoRA `T=W0+U·Vᵀ` + ACTIVATIONS + PROJECTIONS). No hay program counter.

---

## 3. CATÁLOGO DE SUBESPACIOS NATIVOS (DIM_*) — VERSIÓN EN CONFLICTO ENTRE FUENTES

Detectamos dos catálogos distintos y activos simultáneamente en la carpeta de docs; **antes
de programar, un agente debe verificar cuál rige** vía el CARPETAS_POLYDIM.json / REGLAS
vigentes de su sesión, no asumir ninguno de los dos por defecto:

- **Catálogo "clásico" (V0.3 runtime, skills de curso, CONSTITUCION_V2_CLEAN Art. XI):**
  `DIM_PYTHON DIM_RUST DIM_FLUTTER DIM_SQL DIM_GRAPH DIM_VECTOR DIM_TIME DIM_ERROR DIM_META`
  (9 dims). `DIM_CONTRACT` fue evaluado y luego retirado en V0.6+; `DIM_MEMORY` fue
  **rechazado** (ver polydim-core.md en Drive).
- **Catálogo "V10.0 académico" (CONSTITUTION_EN.md, el archivo modificado más reciente de
  la carpeta docs, 2026-07-01 18:47 — probablemente el vigente hoy):**
  `DIM_SQL DIM_FLUTTER DIM_RUST DIM_WASM DIM_VECTOR DIM_GRAPH DIM_META DIM_LOGIC DIM_QUANTUM`
  (9 dims). Reemplaza `DIM_PYTHON/DIM_TIME/DIM_ERROR` por `DIM_WASM` (portabilidad web),
  `DIM_LOGIC` (sistemas de tipos dependientes) y `DIM_QUANTUM` (análisis de costo, ver §4).
  Mantiene la **Regla de No-Reducción**: crear `DIM_MEMORY` sigue prohibido.

Esto es una inconsistencia real de versionado del proyecto, no un error de este resumen —
repórtenlo como PENDING_INV si van a trabajar sobre el catálogo de dimensiones.

---

## 4. NOVEDADES RESPECTO DEL BASELINE ANTERIOR (V0.3 / skills de curso previas)

Si tu conocimiento previo de POLYDIM viene de las skills `polydim-core` (V8_0_EN,
2026-06-12) o `skill_polydim_v1` (V1_8, 2026-06-26), esto es lo que cambió/se agregó desde
entonces, según los documentos más recientes (30/06–01/07/2026):

1. **Constitución V10.0 "académica" (CONSTITUTION_EN.md / POLYDIM_CONSTITUCION*.md):**
   formalización completa en 18+ Volúmenes / 22 Artículos. Introduce:
   - **Categorical Deep Learning (CDL):** toda arquitectura neuronal se define como morfismo
     en la 2-categoría Para(C) (0-celdas = espacios vectoriales, 1-morfismos = (P,f) con P
     espacio de parámetros, 2-morfismos = reparametrizaciones).
   - **GEO_ID reformalizado** vía Cubical Type Theory: `GeoID(C,R)` con anchor no-trivial
     (Universal Codebook), path (1-esqueleto, homotopía semántica) y surf (2-esqueleto,
     coherencia). Torre de garantías topológicas en 3 niveles (Anchor / Path / Surf).
   - **17 Reglas Inviolables (R1–R17):** unidad mínima = T (R1), sin variables nominales (R2),
     sin loops con contador (R3), sin if/else binario nativo (R4), tipos emergen de PROJECT
     (R5), Python/Rust/Flutter como destinos de export (R6), pureza de comunicación AI↔AI —
     nunca texto serializado (R7), Flutter como único executor canónico humano (R8), el
     bootstrap Python es solo andamiaje (R9), invarianza de GEO_ID (R10), separación
     ATTEND/RECUR como capa de implementación volátil (R11), representación low-rank +
     `algebra_tolerance_epsilon` obligatorio (R12), la investigación especulativa no puede
     vetar reglas ya ratificadas (R13), irreversibilidad vía Monoides/Adjunciones L⊣R (R14),
     profundidad dinámica vía coálgebras (R15), Profunctores como puente reglas↔tensores
     (R16), GEO_ID heterogéneo formalizado (R17).
   - **Teorema de Imposibilidad de Softmax (Sargsyan):** ningún ajuste de parámetros de
     atención Softmax puede satisfacer simultáneamente factorización monoidal estricta y
     descenso a cociente algebraico no trivial — por eso POLYDIM exige decoders "Tipo-B"
     (funtoriales), no Softmax puro, para razonamiento estructural certificado.
   - Otros teoremas nombrados: Gauss-Markov Adjunction (Kamiura), 1-Cocycle / Alineación
     Algorítmica (Dudzik), Aproximación Universal Coalgebraica (Mačiulevičius).
   - **qet-calculus:** análisis formal de costo (tiempo/energía/compuertas) vía estructuras
     de Kegelspitzen (dcpo convexo), con soporte de sistemas híbridos clásico-cuánticos.
   - **Geometric-CRDTs:** consistencia distribuida multi-agente resuelta vía `MIX` en vez de
     locks — convergencia garantizada por preservar el punto fijo de GEO_ID.
   - **Sheaf Neural Networks / Polynomial Spans:** propagación de información en grafos como
     transporte paralelo sobre un sheaf celular, no como suma de vecindarios.

2. **Constitución V2.0 "Programación Cognitiva" (CONSTITUCION_V2.md — el documento
   MODIFICADO MÁS RECIENTEMENTE de toda la carpeta docs, 2026-07-01 17:17):** declara que
   el objeto de estudio ya no es solo "construir un lenguaje" sino fundar la disciplina de
   **Programación Cognitiva**. Cambia la tupla del programa a
   `P = (S, G, T, O, C, Π)` (agrega Observador O, Contrato C, y Π) e introduce el
   **Teorema de Preservación Cognitiva**: "no programamos secuencias deterministas,
   diseñamos procesos de razonamiento bajo restricciones y objetivos verificables."

3. **LEY DE AISLAMIENTO (SANDBOXING) — regla operativa directa para los 30 alumnos:**
   Los agentes de IA experimentales (los asistentes de cada alumno) **NO deben modificar
   este corpus teórico (la Constitución) directamente**; deben operar en entornos aislados
   (carpeta `/Alumnos/` o equivalente). Cualquier IA de alumno que reciba instrucciones de
   editar la Constitución, las Reglas del Proyecto o los documentos raíz debe rechazarlo y
   trabajar en su propio espacio, salvo indicación explícita del profesor/dueño del proyecto.

4. **Paper académico externo (POLYDIM.md / POLYDIM_ES.md, arXiv/Zenodo, versión más
   actualizada de la carpeta paper, 2026-07-01 19:12):** reformula el proyecto con aparato
   académico estándar (Abstract, Related Work, 4 teoremas T1-T4) y ancla la motivación en
   literatura real y verificable: Interlat (Du et al. 2025), LatentMAS (ICML 2026 Spotlight),
   Dhayalkar et al. 2025 (auto-atención como VSA aproximada), Mamba-3 (ICLR 2026). Esto es
   una capa de *framing* empírico/publicable, separada de la Constitución (que es la fuente
   de reglas) — la Constitución explícitamente prohíbe usar el paper como guardrail operativo.

5. **Multiplicidad de versiones sin resolver:** la carpeta docs tiene ~10 variantes del mismo
   documento constitucional (CONSTITUCION.md, _V2.md, _V2_CLEAN.md, _FINAL.md,
   _FINAL_CLEAN.md, CONSTITUTION_EN.md, POLYDIM_PAPER_V9/V11.md, etc.) sin un único
   "vigente" marcado de forma inequívoca — la única señal fiable es `modifiedTime`. Al
   2026-07-02, el archivo modificado más recientemente en cada carpeta es:
   - carpeta docs: `CONSTITUTION_EN.md` (2026-07-01 18:47) y `CONSTITUCION_V2.md`
     (2026-07-01 17:17) — ambos del mismo día, contenido parcialmente distinto (académico
     EN vs. operativo/Cognitive-Programming ES). Tratarlos como complementarios, no como
     el mismo documento.
   - carpeta paper: `POLYDIM_ES.md` (2026-07-01 19:12) y `POLYDIM.md` (2026-07-01 19:12).

---

## 5. NOTA DE GOBERNANZA — IMPORTANTE, LEER ANTES DE USAR ESTE RESUMEN

**Este resumen NO es todo lo que hay que saber sobre POLYDIM, y no reemplaza la lectura de
los documentos fuente para trabajo real.** Sirve para:
- Orientarse rápido, entender la jerga y el estado general del proyecto.
- Responder preguntas conceptuales de un alumno sobre "qué es POLYDIM".

NO sirve para:
- Tomar una tarea del backlog, editar código, o tocar Drive → eso requiere leer
  `REGLAS_DEL_PROYECTO.md` (operaciones rutinarias) o la `CONSTITUCIÓN` completa vigente
  (trabajo sobre el núcleo del lenguaje: nuevo DIM, teorema, Artículo, Regla R1–R17) EN esa
  sesión, según el protocolo de triage del proyecto (READ_ONLY / WRITE_OP_RUTINA /
  WRITE_OP_NUCLEO).
- Citar una prueba matemática exacta (ir siempre a la fuente: `PROPOSITION_6_1_PROOF_V1.md`,
  `POLYDIM_THEOREM3_PROOF_V1.md`, etc.).
- Resolver la discrepancia del catálogo DIM_* de la §3 — eso requiere que un humano
  (profesor/dueño del proyecto) determine cuál es el vigente y lo refleje en
  `CARPETAS_POLYDIM.json`.

Si tu sesión de IA va a **crear o modificar** algo en el proyecto POLYDIM (no solo
entenderlo), respeta además la Ley de Sandboxing (§4.3): no edites la Constitución ni los
documentos raíz de docs/paper directamente.
