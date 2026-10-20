# POLYDIM_DEST
# destino: polydim/docs/
# nombre:  REGLAS_DEL_PROYECTO.md
# fecha:   2026-06-25
# fuente:  extraído de POLYDIM_CONSTITUCION_V7.md (Artículos III, IV.1, VI.1, XIII, XVI, XX)
# rol:     guardrail liviano de lectura obligatoria para TODA sesión WRITE_OP rutinaria.
#          NO reemplaza la Constitución. Es un subconjunto operativo, no una versión reducida.
#          Para teoremas, demostraciones, brechas, mecanismos en desarrollo e investigación
#          especulativa → leer POLYDIM_CONSTITUCION_V7.md completa (solo si la tarea es de
#          núcleo del lenguaje: nueva DIM, nuevo teorema, resolver brecha, tocar Artículo
#          IV/VI/VIII/XI/XVI, o modificar estas mismas reglas).
#          El paper / esqueleto académico es un documento DERIVADO de la Constitución para
#          publicación externa — no se lee en sesiones de trabajo, nunca es fuente de reglas.

# REGLAS DEL PROYECTO — POLYDIM

### Subconjunto operativo de la Constitución V7. Vinculante. No reemplaza el Artículo XX (proceso de enmienda) ni autoriza a modificar la Constitución directamente.

---

## 1 — LO QUE NO EXISTE (no lo generes, ni en código, ni en specs, ni en skills)

| No existe | En su lugar |
| - | - |
| Variable con nombre | Posición = (GEO_ID, activaciones) |
| Función `f(x)` declarada | Transformación T: R^N → R^N + COMPOSE |
| Loop `for` / `while` | FIXPOINT (convergencia, no contador) |
| `if` / `else` binario | MIX (superposición continua, no bifurcación) |
| Tipo declarado (`int`, `str`, clase) | El tipo emerge de PROJECT, no se anota |
| Instrucción secuencial | COMPOSE: T₂ ∘ T₁ (no conmutativo) |
| Mensaje de texto / JSON entre IAs | Transmisión directa de T (ALIGN / cross-attention) |

Ninguna versión futura puede introducir estos conceptos como nativos del lenguaje (R2, R3, R4).

---

## 2 — LAS CUATRO PRIMITIVAS ALGEBRAICAS (capa invariante, Artículo IV.1)

```
COMPOSE(T₁, T₂) = T₂ ∘ T₁                          (orden importa, no conmutativo)
MIX(α, T₁, β, T₂) = α·T₁ + β·T₂                     (reemplaza if/else)
FIXPOINT(T, ε): iterar s_{k+1}=T(s_k) hasta ‖Δ‖ < ε  (reemplaza loops)
PROJECT(T, executor) → vista del executor            (no convierte, observa)
```
PROJECT se fragmenta en implementación: COMPILE (binario), RENDER (Flutter), EXPORT (sistemas no tensoriales: SQL, grafos, APIs).

ATTEND y RECUR son capa de implementación (ligada a transformer/SSM), no capa algebraica invariante (R11).

---

## 3 — GEO_ID (Artículo VI.1)

Toda posición = (GEO_ID, activaciones). El GEO_ID es invariante: nunca cambia bajo ninguna transformación permitida (R10). Las activaciones sí mutan — eso no es "reasignación", es evolución continua del estado.

---

## 4 — LOS NUEVE SUBESPACIOS NATIVOS (Artículo XIII)

```
DIM_PYTHON   DIM_RUST    DIM_FLUTTER  DIM_SQL   DIM_GRAPH
DIM_VECTOR   DIM_TIME    DIM_ERROR    DIM_META
```
Pueden agregarse subespacios nuevos. **No pueden eliminarse** estos nueve.

---

## 5 — LAS 17 REGLAS INVIOLABLES (Artículo XVI, completo)

Ninguna de estas reglas puede modificarse por decisión de un alumno, de una IA en sesión, ni de ninguna persona individual sin el proceso formal de enmienda (sección 6 de este archivo / Artículo XX completo).

| Regla | Contenido |
| - | - |
| **R1** | La unidad fundamental es T: R^N → R^N. Nunca una instrucción. |
| **R2** | No hay variables con nombre. No pueden introducirse como feature nativo del lenguaje. |
| **R3** | No hay loops secuenciales imperativos. La convergencia se expresa con FIXPOINT. |
| **R4** | No hay `if`/`else` binario nativo. La rama condicional se expresa con MIX. |
| **R5** | Los tipos emergen de la proyección (PROJECT/COMPILE/RENDER/EXPORT). No se declaran. |
| **R6** | Python, Rust y Flutter son destinos de exportación/ejecución del bootstrap. No son la base del lenguaje. |
| **R7** | La comunicación AI→AI transmite transformaciones T (vía cross-attention/ALIGN), nunca texto serializado, como mecanismo nativo. |
| **R8** | Flutter es el único executor calificado como nativo del subespacio humano (isomorfismo algebraico verificable). |
| **R9** | El bootstrap Python (`polydim_runtime_v03.py`) es el andamio. No puede presentarse como la sintaxis o semántica final del lenguaje sin la etiqueta explícita "bootstrap". |
| **R10** | El GEO_ID de un objeto es invariante: ∀T permitida, dist(T(GEO_ID), GEO_ID) < ε. La extensión GeoID(C,R) (Artículo VI.3bis) es vinculante V7. |
| **R11** | ATTEND y RECUR pertenecen a la capa de implementación (IV.2), no a la capa algebraica invariante (IV.1). |
| **R12** | El formato `.polydim` no especifica matrices densas N×N. Representación de bajo rango por defecto: T = W₀ + U·Vᵗ. El header incluye `algebra_tolerance_epsilon`. |
| **R13** | Ninguna formalización marcada 🔬 investigación especulativa puede citarse como ley vinculante. |
| **R14** | Las transformaciones de POLYDIM NO necesitan ser invertibles. Opera sobre monoides, no sobre grupos. |
| **R15** | POLYDIM soporta profundidad dinámica de cómputo. Ninguna implementación puede hardcodear una profundidad fija como límite del lenguaje. |
| **R16** | El puente Top-Down ↔ Bottom-Up se implementa mediante Profuntores, no mediante penalizaciones suaves ad hoc. |
| **R17** | El GEO_ID se define sobre GeoID(C,R) cuando opera entre modelos heterogéneos. El punto base no es trivial: porta contenido semántico del codebook del dominio. |

---

## 6 — SI ALGO DE ESTO PARECE INSUFICIENTE O CONTRADICTORIO

NO lo resuelvas por tu cuenta y NO sigas con la tarea asumiendo una interpretación propia. Tres caminos, en este orden:

1. ¿Es ambigüedad de lectura, no de contenido? → Releer la sección correspondiente de POLYDIM_CONSTITUCION_V7.md completa (ahí está el mecanismo, la demostración o la brecha que falta aquí por diseño).
2. ¿Sigue sin resolverse? → Es candidato a PENDING_INV. Registralo en el relay (sección 6: PENDING_RECONCILIATIONS) y seguí con lo que sí es claro.
3. ¿Es un cambio real a una regla? → NUNCA se decide en sesión. Artículo XX: escribir PROPUESTA_CONSTITUCION_NNN.md en backlog/, justificar matemáticamente, esperar validación del docente.

---

*REGLAS_DEL_PROYECTO.md · derivado de POLYDIM_CONSTITUCION_V7.md · 2026-06-25*
*Constitución completa (teoremas, mecanismos, brechas, investigación): POLYDIM_CONSTITUCION_V7.md en docs/*
*Paper académico (output, no input): carpeta paper — nunca se lee como guardrail de sesión*
