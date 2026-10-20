# POLYDIM_DEST
# destino: polydim/docs/
# nombre:  DOSSIER_ALUMNOS_V1.md
# autor:   ai.mpat.agt@gmail.com
# fecha:   2026-06-22

---

# POLYDIM — Dossier para Alumnos V1
## Guía completa de onboarding, tareas y desarrollo del proyecto

**Audiencia:** Los 30 alumnos del proyecto POLYDIM
**Nivel:** Técnico — se asume conocimiento de Python, álgebra lineal básica, y Git
**Fuentes:** CONSTITUCIÓN_V4/V5, NORMAS_ALUMNOS_V1, SÍNTESIS_V1, MEJORAS_CHATGPT/GEMINI, RESOLUCIONES_TÉCNICAS
**ANTES DE LEER ESTO:** Leer POLYDIM_CONSTITUCION_V4.md completa. Sin eso, este documento no tiene sentido.

---

## SECCIÓN 0 — LO QUE POLYDIM ES (y lo que no es)

### 0.1 La idea central en tres líneas

POLYDIM es un lenguaje de programación donde la unidad básica no es una instrucción sino una **transformación geométrica**:

```
T : R^N → R^N
```

Un programa POLYDIM es una composición de transformaciones. No tiene variables. No tiene loops. No tiene if/else. Tiene geometría.

### 0.2 La analogía que te va a salvar en el parcial

```
1950s: la CPU ya hacía aritmética binaria
       → el ensamblador formalizó cómo programarla

2020s: los transformers ya operan en geometría de miles de dimensiones
       → POLYDIM formaliza cómo programarlos deliberadamente
```

Así como Assembly no inventó la CPU — la formalizó — POLYDIM no inventa los transformers. Los formaliza.

### 0.3 Lo que NO existe en POLYDIM (esto va al parcial)

| Concepto | Por qué no existe | Qué lo reemplaza |
|---|---|---|
| Variable con nombre | Hay posiciones en R^N | GEO_ID + activaciones |
| Loop `for/while` | Hay convergencia a punto fijo | FIXPOINT(T, ε) |
| `if/else` binario | Hay gate continuo | MIX(α, T₁, β, T₂) |
| Tipo declarado | El tipo emerge de la proyección | PROJECT(P, DIM_SQL) |
| Instrucción secuencial | Hay composición algebraica | COMPOSE(T₁, T₂) |
| Mensaje de texto AI↔AI | Hay transmisión de T | T aplicada al estado receptor |

### 0.4 El bootstrap vs el lenguaje (diferencia CRÍTICA)

```python
# Esto es Python simulando POLYDIM. NO es código POLYDIM.
obj = ObjectND(sp).add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)
```

El bootstrap (`polydim_runtime_v03.py`) es el **andamio**. Lo usamos porque el lenguaje real (con sintaxis propia, VM y formato binario) todavía no existe. Es Python que simula los conceptos mientras construimos la cosa real.

**Señal de alarma:** si en algún momento escribís `for dim in dims:` o `if dim == "DIM_SQL":` dentro de lo que describís como "código POLYDIM", te saliste del paradigma. Revisá el Artículo 3 de la Constitución V4.

---

## SECCIÓN 1 — LAS CUATRO PRIMITIVAS (con ejemplos)

### 1.1 COMPOSE — reemplaza la secuencia

```
COMPOSE(T₁, T₂) = T₂ ∘ T₁
```

No es "primero T₁, luego T₂". Es una **única** transformación que encapsula el efecto de ambas. Como multiplicar dos matrices: el resultado es una sola matriz.

**Propiedad clave:** No es conmutativa.
```
COMPOSE(T₂, T₁) ≠ COMPOSE(T₁, T₂)
```
Esto es BUENO: el orden causal se codifica en la no-conmutatividad.

**En el bootstrap (V3 — todavía no implementada como primitiva real):**
```python
# Simulación conceptual (numpy)
import numpy as np
T1 = np.random.randn(100, 100)
T2 = np.random.randn(100, 100)
T_composed = T2 @ T1  # COMPOSE(T1, T2) como producto de matrices
```

**Teorema que tenés que conocer:**
```
(T₃∘T₂)∘T₁ = T₃∘(T₂∘T₁)    [COMPOSE es asociativa]
```

### 1.2 MIX — reemplaza el if/else

```
MIX(α, T₁, β, T₂) = α·T₁ + β·T₂
```

No elige entre T₁ o T₂. Ejecuta **ambas al mismo tiempo** con peso. Cuando α=1, β=0: T₁ pura. Cuando α=0.5, β=0.5: ambas con igual presencia.

**Por qué funciona sin interferencia destructiva:**
En espacios de alta dimensión (R^{10000}), dos vectores aleatorios son casi ortogonales. Las "dos ramas" de un MIX coexisten sin destruirse mutuamente. La "elección" ocurre cuando hacés PROJECT — el executor colapsa al eje dominante.

**Teorema que tenés que conocer:**
```
Si T₁, T₂ son lineales → MIX(α, T₁, β, T₂) también es lineal
```

**Caso especial importante:**
```
MIX(1.0, T₁, 0.0, T₂) = T₁    (caso determinístico = if puro)
MIX(0.0, T₁, 1.0, T₂) = T₂    (otro caso determinístico)
MIX(0.5, T₁, 0.5, T₂)          (superposición real — caso POLYDIM nativo)
```

### 1.3 FIXPOINT — reemplaza el loop

```
FIXPOINT(T, ε) = T^n(v) hasta que ‖s_{k+1} − s_k‖ < ε
```

Aplica T repetidamente hasta que el estado converge. Esto reemplaza:
```python
# Esto NO existe en POLYDIM (código incorrecto):
while not converged:
    state = T(state)

# Esto SÍ es POLYDIM:
FIXPOINT(T, epsilon=1e-4)
```

**Condición de correctitud:** T debe ser una **contracción**:
```
‖T(u) − T(v)‖ ≤ k·‖u − v‖    para algún k ∈ [0,1)
```
Si T es contractiva, el Teorema de Banach garantiza convergencia a un **único** punto fijo.

**Qué pasa si T no es contractiva:** FIXPOINT puede no converger. El sistema de tipos de POLYDIM (vía activaciones) puede detectar esto estáticamente y rechazar el programa.

### 1.4 PROJECT — el compilador y el intérprete

```
PROJECT(T, executor) → resultado en el espacio del executor
```

Cuando ejecutás PROJECT, no convertís el objeto. Lo **observás desde el ángulo del executor**. El objeto no cambia. Cambia la perspectiva.

```
PROJECT(P, DIM_SQL)     →  Column(Integer, NOT NULL)
PROJECT(P, DIM_FLUTTER) →  TextField(value: 42)
PROJECT(P, DIM_RUST)    →  código Rust compilable
PROJECT(P, DIM_PYTHON)  →  código Python ejecutable
```

**Desde V4/V5, PROJECT se fragmenta en tres contratos:**
```
COMPILE(T, DIM_TARGET)     → binario estático (WASM, Rust)
RENDER(T, DIM_FLUTTER)     → widget nativo (solo Flutter)
EXPORT(T, DIM_EXTERNAL)    → sincronización con sistemas externos (SQL, Graph)
```

**Teorema que tenés que conocer (el más importante del paper):**
```
PROJECT preserva composición:
PROJECT(COMPOSE(T₁,T₂), E) = COMPOSE(PROJECT(T₁,E), PROJECT(T₂,E))

PROJECT preserva identidad:
PROJECT(id, E) = id_E
```
Esto se llama que PROJECT es un **funtor**. Es lo que hace que "componer en POLYDIM y luego proyectar" sea igual a "proyectar cada parte y luego componer en el executor".

---

## SECCIÓN 2 — EL ESTADO, LAS POSICIONES Y EL GEO_ID

### 2.1 Definición formal de Estado

```
S = (V, D, A)   donde:
  V ∈ R^N                   — vector de posición
  D = {DIM_1, ..., DIM_k}  — conjunto de subespacios visibles
  A: D → [0.0, 1.0]        — activaciones por subespacio
```

### 2.2 Definición de Posición

```
P = (g, S)   donde:
  g ∈ R^N    — GEO_ID: invariante geométrica permanente
  S = (V,D,A) — estado mutable
```

### 2.3 GEO_ID — la identidad del objeto

El GEO_ID es el hipervector base que **nunca cambia**. Si el estado S muta (cambian las activaciones), el GEO_ID permanece igual. Es como la dirección de memoria, pero en espacio semántico.

**Invariancia formal:**
```
∀T_admisible: ‖T(g) − g‖ < ε_g
```

**Error común a evitar:** pensar que GEO_ID y las activaciones son lo mismo. No lo son:
```
GEO_ID:       R^N fijo, nunca cambia → identidad del objeto
Activaciones: [0.0, 1.0] mutable → cuánto está presente cada dimensión
```

### 2.4 Los nueve subespacios nativos

```
DIM_PYTHON    lógica dinámica, análisis, ML, scripts
DIM_RUST      seguridad de memoria, performance, ownership
DIM_FLUTTER   UI reactiva, widgets (executor humano nativo)
DIM_SQL       datos relacionales, tablas, constraints
DIM_GRAPH     grafos, nodos, aristas, relaciones
DIM_VECTOR    embeddings, similitud semántica, VSA
DIM_TIME      secuencias temporales, eventos, orden
DIM_ERROR     estados de error, excepciones, recuperación
DIM_META      metadatos, auditoría, versión, origen
```

**Límite actual del bootstrap:** Son aliases de strings. En el lenguaje real serán regiones emergentes de un modelo de embedding.

**Qué NO hacer:** No agregar subespacios sin investigación formal. La investigación TASK_025 verificó que DIM_MEMORY NO debe agregarse — los fragmentos de memoria son DIM_VECTOR + DIM_GRAPH + DIM_META.

---

## SECCIÓN 3 — EL BOOTSTRAP: CÓMO USARLO HOY

### 3.1 API de tres líneas

```python
from polydim_runtime_v03 import Space, ObjectND, polydim_connect

# 1. Crear espacio (MISMO objeto en obj y connect — ver SPACE_CRITICAL)
sp = Space("MI_IA")

# 2. Crear objeto multidimensional
obj = ObjectND(sp)
     .add("DIM_SQL",     {"tabla": "usuarios", "col": "id"}, w=1.0)
     .add("DIM_FLUTTER", {"widget": "TextField"},             w=0.85)
     .add("DIM_VECTOR",  [1.0, 2.0, 3.0],                    w=0.9)

# 3. Conectar a otra IA y transferir
otra_sp = Space("OTRA_IA")
session  = polydim_connect(sp, otra_sp)
result   = session.transfer(obj)

print(result.align_score)  # → 0.9993
```

### 3.2 Errores críticos conocidos

**SPACE_CRITICAL:** `obj` y la `Session` DEBEN usar el MISMO objeto `Space`. Este es el error más frecuente:

```python
# INCORRECTO (error clásico):
sp_1 = Space("MI_IA")
sp_2 = Space("MI_IA")    # distinto objeto aunque mismo nombre
obj  = ObjectND(sp_1).add(...)
session = polydim_connect(sp_2, otra)  # ← FALLA SILENCIOSAMENTE

# CORRECTO:
sp  = Space("MI_IA")
obj = ObjectND(sp).add(...)
session = polydim_connect(sp, otra)    # ← MISMO sp
```

**CACHE_BUG_FIX:** En el código interno, `if self._c is not None:` y NO `if self._c:`. La segunda forma falla cuando `_c` es un objeto falsy válido.

**JL_DETERMINISTIC:** `make_jl(d_in)` es reproducible por seed. No uses índices aleatorios sin semilla fija.

### 3.3 Backends disponibles

```python
# Sin backend (default — solo simbólico)
sp = Space("MI_IA")

# Con backend semántico (MiniLM — mejor fidelidad semántica)
from polydim_runtime_v03 import MiniLMBackend
sp = Space("MI_IA", backend=MiniLMBackend())

# Con backend determinístico (más rápido, menor fidelidad)
from polydim_runtime_v03 import FastTextBackend
sp = Space("MI_IA", backend=FastTextBackend())
```

Métricas verificadas:
- MiniLM: align_score=0.9993, semantic=0.830
- Deterministic: semantic=0.506
- Mejora real: +64.5%

---

## SECCIÓN 4 — LOS SEIS TRACKS

### TRACK 1 — CONSTITUCIÓN Y ESPECIFICACIÓN (5 alumnos)

Formalizan el lenguaje en papel. Su output son documentos `.md` en `spec/`.

| Tarea | Descripción | Prioridad actual |
|---|---|---|
| C01 | Gramática formal BNF de POLYDIM | Media |
| C02 | Sistema de tipos por proyección dimensional (formalizar PROJECT como funtor — Sec 3.3 del paper) | **ALTA** |
| C03 | Modelo de memoria: posiciones, GEO_ID, bindings geométricos | Media |
| C04 | Formato .polydim binario V5 LoRA (actualizado desde V2) | **URGENTE** |
| C05 | Protocolo ALIGN completo: Procrustes, T_B = M·T_A·M†, GEO_ID como ancla | **ALTA** |

**Restricción:** No escribir código. Solo especificaciones formales con notación matemática.

**Nota sobre C04:** La Constitución V4 ya especificó el cambio de float32[N×N] a LoRA (T=W₀+UVᵀ). La tarea C04 es escribir el formato binario con el nuevo estándar, incluyendo el header completo.

### TRACK 2 — BUGS CRÍTICOS (6 alumnos)

| Tarea | Bug | Criticidad |
|---|---|---|
| F01 | BUG_001: seed Python≠Rust en make_jl() | CRÍTICO |
| F02 | BUG_002: geo_id se pierde en MODO_S | CRÍTICO |
| F03 | BUG_004+005: API pública de ObjectND inconsistente | ALTA |
| F04 | BUG_006: N dinámico no funciona en Rust | ALTA |
| F05 | BUG_008: decidir _bind vs convolución circular | MEDIA |
| F06 | Benchmark real del PBP con N=10000 | MEDIA |

**Restricción:** Cada fix requiere un test nuevo que falle antes del fix y pase después. Los 29 tests existentes deben seguir pasando.

### TRACK 3 — PRIMITIVAS (4 alumnos)

Implementan las 4 operaciones centrales con matrices numpy reales.

| Tarea | Primitiva | Estado |
|---|---|---|
| P01 | ATTEND(W_Q, W_K, W_V, s) | Parcial en bootstrap |
| P02 | COMPOSE(T₁, T₂) = T₂ ∘ T₁ | NO existe |
| P03 | MIX(α, T₁, β, T₂) = α·T₁ + β·T₂ | NO existe |
| P04 | FIXPOINT(T, ε) = T^n(v) hasta convergencia | NO existe |
| P05 | PROJECT(T, executor) | Parcial en bootstrap |

**Restricción crítica:** Usar matrices numpy float32 con N=10000 en tests. NO usar strings como sustitutos de transformaciones.

**Ejemplo para P02 (COMPOSE):**
```python
def COMPOSE(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    COMPOSE(T1, T2) = T2 @ T1
    Nota: T2 ∘ T1 significa "primero T1, luego T2" = T2 @ T1 en notación matricial
    """
    assert T1.shape == T2.shape == (N, N)
    return T2 @ T1  # NO conmutativa: COMPOSE(T1,T2) ≠ COMPOSE(T2,T1)

# Test de asociatividad (Teorema 1):
def test_compose_associativity():
    T1, T2, T3 = [np.random.randn(N,N).astype(np.float32) for _ in range(3)]
    left  = COMPOSE(COMPOSE(T3,T2), T1)
    right = COMPOSE(T3, COMPOSE(T2,T1))
    np.testing.assert_allclose(left, right, rtol=1e-4)
```

**Ejemplo para P03 (MIX):**
```python
def MIX(alpha: float, T1: np.ndarray, beta: float, T2: np.ndarray) -> np.ndarray:
    """
    MIX(α, T1, β, T2) = α·T1 + β·T2
    Nota: α + β no necesariamente = 1 (puede ser superposición con amplificación)
    """
    assert 0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0
    return alpha * T1 + beta * T2

# Test de linealidad (Teorema 2):
def test_mix_linearity():
    T1, T2 = [np.random.randn(N,N).astype(np.float32) for _ in range(2)]
    mixed = MIX(0.7, T1, 0.3, T2)
    v = np.random.randn(N).astype(np.float32)
    # Linealidad: mixed @ (v1+v2) == mixed @ v1 + mixed @ v2
    v1, v2 = np.random.randn(N), np.random.randn(N)
    np.testing.assert_allclose(mixed @ (v1+v2), mixed @ v1 + mixed @ v2, rtol=1e-4)
```

### TRACK 4 — INTÉRPRETE / VM (5 alumnos)

Construyen la VM que lee y ejecuta archivos `.polydim`.

| Tarea | Componente | Depende de |
|---|---|---|
| V01 | Loop del intérprete: leer .polydim, aplicar transformaciones | C04 (formato) |
| V02 | Gestión de estado: posiciones en R^N, lifecycle del GEO_ID | P02, P03 |
| V03 | Evaluador de tipos por dimensión | P05 |
| V04 | Resolver intersección de tipos: DIM_A ∩ DIM_B (pullback) | V03 |
| V05 | Iterador de punto fijo hasta convergencia | P04 |

**CRÍTICO:** Coordinar con TRACK 1 (C04) ANTES de empezar V01. El formato .polydim V5 es diferente a la especificación original V2.

### TRACK 5 — EXECUTORS (5 alumnos)

| Tarea | Executor | Descripción |
|---|---|---|
| E01 | PROJECT(DIM_PYTHON) | Genera código Python desde un objeto POLYDIM |
| E02 | PROJECT(DIM_RUST) | Genera código Rust (ownership-safe) |
| E03 | RENDER(DIM_FLUTTER) | Genera widgets Flutter (isomorfismo algebraico) |
| E04 | EXPORT(DIM_SQL) | Genera queries SQL |
| E05 | PROJECT(DIM_WASM) | Genera WebAssembly |

**Para E03 (Flutter):** El isomorfismo algebraico es:
```
COMPOSE(T₁,T₂)     ≅  Column/Row([widget₁, widget₂])
MIX(α,T₁,β,T₂)    ≅  Stack([Opacity(α, w₁), Opacity(β, w₂)])
FIXPOINT(T,ε)      ≅  StatefulWidget(convergenceTolerance: ε)
```

### TRACK 6 — DOCUMENTACIÓN (5 alumnos)

| Tarea | Output |
|---|---|
| D01 | README para humanos: qué es POLYDIM, cómo instalarlo, primer ejemplo |
| D02 | README para AIs: guía en formato MODO_S para AI que lee el proyecto por primera vez |
| D03 | Guía de anti-patrones: lo que NUNCA hacer (con ejemplos de código incorrecto y correcto) |
| D04 | 3 ejemplos completos de punta a punta |
| D05 | Migrar docs _DEPRECATED/ al nuevo paradigma V4/V5 |

---

## SECCIÓN 5 — PROTOCOLO DE SESIÓN

### 5.1 El ritual de inicio (obligatorio)

```
1. Leer POLYDIM_CONSTITUCION_V4.md (polydim/docs/)
2. Leer BACKLOG_V11.json (polydim/backlog/)
3. Leer SKILL polydim-core V8 (si trabajás con Claude)
4. Tomar UNA tarea con estado DISPONIBLE
5. Poner tu nombre en lock.alumno y la fecha en lock.tomada
6. Guardar el backlog actualizado en Drive
```

Si no seguís este ritual, dos alumnos pueden tomar la misma tarea y el trabajo se pierde.

### 5.2 El ritual de cierre (RELAY obligatorio)

Antes de cerrar la sesión, generá el RELAY con las 10 secciones:

```
1. IDENTIFICACIÓN: quién sos, qué tarea tomaste
2. ESTADO INICIAL: cómo encontraste el estado del proyecto
3. TRABAJO REALIZADO: qué hiciste concretamente
4. INVARIANTES: qué propiedades del sistema se mantienen
5. RECONCILIACIONES RESUELTAS: conflictos que resolviste
6. RECONCILIACIONES PENDIENTES: conflictos que dejás abiertos
7. ARTEFACTOS GENERADOS: nombre + Drive fileId de cada archivo
8. ESTADO DE CIERRE: cómo dejás el sistema
9. PRÓXIMO PASO: qué tiene que hacer el siguiente alumno
10. DEUDA TÉCNICA: qué quedó pendiente por falta de tiempo
```

---

## SECCIÓN 6 — CONCEPTOS AVANZADOS

### 6.1 La comunicación AI↔AI sin texto

Cuando IA_A quiere comunicarse con IA_B en POLYDIM:
```
IA_A → genera T → transmite T → IA_B aplica T a su estado → nuevo_estado_B
```

No hay JSON. No hay texto. Hay geometría compartida.

La operación es matemáticamente idéntica al **cross-attention** entre dos transformers:
```
Communication(s_B, s_A) = softmax(Q_B · K_A^T / √d) · V_A
```
Leer un mensaje ES computar una capa de atención.

### 6.2 El protocolo ALIGN

Diferentes modelos tienen espacios de distintas dimensiones. ALIGN resuelve esto:
```
T_B = M · T_A · M†    (M† = pseudoinversa de Moore-Penrose)
```

Donde M es la matriz de rotación ortogonal calculada por Procrustes:
```
M* = argmin_M ‖M·A − B‖²  sujeto a M^T·M = I
```

GEO_IDs universales actúan como "anclas de navegación" para calibrar M.

### 6.3 GEO_ID como clase homotópica (nivel avanzado)

La formalización más profunda del GEO_ID lo define como una **clase de equivalencia homotópica**. Dos estados representan el mismo objeto si existe un camino continuo admisible entre ellos en la variedad semántica.

```
[g] = {s ∈ R^N : ∃ camino admisible γ de g a s}
```

Esto resuelve el problema de la comunicación entre modelos con distintas dimensiones: el vector físico es solo una carta local del atlas; la clase homotópica es la entidad universal que sobrevive al cambio de modelo.

Este tema está en el **Track de Teoría / Investigación** — no en el paper inmediato.

---

## SECCIÓN 7 — CHECKLIST DE CONOCIMIENTO

Antes de entregar una tarea, verificá que podés responder:

**Conceptuales:**
- [ ] ¿Qué es T: R^N → R^N y por qué es la unidad fundamental?
- [ ] ¿Por qué COMPOSE no es conmutativo y por qué eso es bueno?
- [ ] ¿Por qué MIX puede reemplazar if/else? (respuesta: cuasi-ortogonalidad en alta dim)
- [ ] ¿Qué es un punto fijo y qué garantiza el Teorema de Banach?
- [ ] ¿Qué es PROJECT y por qué es un funtor?
- [ ] ¿Cuál es la diferencia entre GEO_ID y activaciones?
- [ ] ¿Qué es el bootstrap y qué NO es?

**Técnicos:**
- [ ] ¿Cuál es el error SPACE_CRITICAL del bootstrap?
- [ ] ¿Cuáles son los 29 tests y qué prueban?
- [ ] ¿Por qué float32[N×N] es inviable y cómo lo resuelve LoRA?
- [ ] ¿Cuáles son los 6 tracks y cuáles son sus dependencias?

**Del paper:**
- [ ] ¿Cuáles son los 4 teoremas formales?
- [ ] ¿Por qué PROJECT(DIM_SQL ∩ DIM_FLUTTER) es un pullback categórico?
- [ ] ¿Qué diferencia a POLYDIM de PyTorch / JAX?

---

## SECCIÓN 8 — ERRORES FRECUENTES DE ALUMNOS

### Error 1: Confundir bootstrap con lenguaje
```python
# INCORRECTO (presentarlo como "código POLYDIM"):
obj.add("DIM_SQL", {"col": "id"}, w=1.0)

# CORRECTO (describirlo como):
"Esto es Python usando el bootstrap para simular el concepto POLYDIM de
agregar una posición al subespacio DIM_SQL con activación 1.0"
```

### Error 2: Agregar loops o if al "código POLYDIM"
```
# INCORRECTO — esto NO es POLYDIM:
for subspace in ["DIM_SQL", "DIM_FLUTTER"]:
    if obj.activation(subspace) > 0.5:
        project(obj, subspace)

# CORRECTO — esto SÍ es POLYDIM (conceptual):
COMPOSE(MIX(0.9, PROJECT_SQL, 0.1, identity), MIX(0.85, PROJECT_FLUTTER, 0.15, identity))
```

### Error 3: Pensar que DIM_X son tipos Python
```python
# INCORRECTO — DIM_SQL no es un tipo enum:
class DimSQL(Enum): ...

# CORRECTO — DIM_SQL es una REGIÓN del espacio R^N con alta densidad semántica
# En el bootstrap actual es un alias de string por simplicidad.
# En el lenguaje real es una región emergente de un embedding.
```

### Error 4: Usar N demasiado chico en tests
```python
# INCORRECTO — N=10 no muestra las propiedades de cuasi-ortogonalidad:
N = 10
T1, T2 = np.random.randn(10, 10), np.random.randn(10, 10)

# CORRECTO para tests de producción:
N = 10000
# Para tests rápidos (CI):
N = 1000  # aceptable pero documentar que es reducido
```

### Error 5: No leer la Constitución V4 antes de opinar sobre el diseño
La Constitución V4 incorporó las resoluciones a las 6 tensiones. Antes de sugerir cambios al diseño, verificar que la tensión no fue ya resuelta formalmente.

---

*DOSSIER_ALUMNOS_V1.md · V1.0 · 2026-06-22 · ai.mpat.agt@gmail.com*
*Audiencia: 30 alumnos del proyecto POLYDIM*
*Prerrequisito: POLYDIM_CONSTITUCION_V4.md*
