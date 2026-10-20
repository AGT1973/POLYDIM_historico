# POLYDIM_DEST
# destination: polydim/research/
# filename:    RESEARCH_MEMORIA_10_ALUMNOS.md
# author:      curso04.mithril@gmail.com

# RESEARCH_MEMORIA_10_ALUMNOS.md
## Arquitectura de Memoria Compartida para 10 IAs Alumnas — POLYDIM

**Fecha:** 2026-06-17
**Tarea:** TASK_033 (propuesta)
**Estado:** ANÁLISIS COMPLETO

---

## 1. EL PROBLEMA

Hay 10 IAs alumnas trabajando en POLYDIM en sesiones independientes.
Cada sesión empieza desde cero: lee la skill, lee el BACKLOG, arranca.

**Lo que se pierde entre sesiones:**
- Decisiones de diseño y por qué se tomaron
- Anti-patrones descubiertos (ej: "no uses MD5 como semántica")
- Hallazgos de research (ej: MiniLMBackend existe en V0.3 pero no en V0.5)
- Evaluaciones externas (ej: evaluacion_proyecto_2026_06_17.md)
- Conocimiento acumulado de sesiones anteriores de otros alumnos

**El resultado actual:** cada alumno potencialmente repite errores,
redescubre lo que ya se sabe, o ignora contexto crítico porque no está
en la skill ni en el BACKLOG.

---

## 2. DIAGNÓSTICO DEL ESTADO ACTUAL

### Skill polydim-core (SKILL.md) — V8_0_EN

**Problema 1: Desactualizada.**
Dice que V0.3 es el runtime principal. Estamos en V0.5 con 10 NATIVE_DIMS,
Mesh, META_PERMISSIONS, etc. Ningún alumno que lea la skill sabe esto.

**Problema 2: No captura decisiones.**
La skill dice QUÉ es POLYDIM pero no POR QUÉ se tomaron las decisiones
clave. Ejemplo: DIM_CONTRACT existe desde TASK_024, pero la skill no
menciona por qué se agregó ni qué debate hubo.

**Problema 3: No captura evaluaciones externas.**
La evaluacion_proyecto_2026_06_17.md (Antigravity) es el documento más
importante del proyecto hoy. No está referenciada en ningún lugar que un
alumno vea al startup.

**Problema 4: No es semánticamente buscable.**
Si un alumno necesita saber qué se decidió sobre props geometricas, no
puede buscar — tiene que leer todo o adivinar el nombre del archivo.

---

## 3. TRES OPCIONES ANALIZADAS

---

### OPCIÓN A: Skills en Markdown (evolución del sistema actual)

**Cómo funciona:**
Mantener la skill MD pero convertirla en un sistema estructurado:
- `polydim-core/SKILL.md` → protocolo y estado actual (actualizada)
- `polydim-decisions/SKILL.md` → decisiones de diseño + POR QUÉ
- `polydim-antipatterns/SKILL.md` → qué NO hacer y por qué
- `polydim-evaluations/SKILL.md` → resúmenes de evaluaciones externas

**Pros:**
- Cero infraestructura extra
- Funciona hoy, sin código nuevo
- Compatible con todos los AI instances sin dependencias
- Versionable en Drive
- Fácil de mantener manualmente

**Contras:**
- Estático — no actualiza solo con el conocimiento de las sesiones
- No buscable semánticamente — hay que saber qué archivo leer
- Crece linealmente — en 6 meses puede ser demasiado para leer en startup
- Duplicación — varios alumnos pueden escribir lo mismo en distintos archivos
- No captura conocimiento tácito (lo que un alumno "sabe" pero no escribió)

**Veredicto:** Solución inmediata válida. No es la solución final.

---

### OPCIÓN B: Knowledge Fragments en Drive (JSON indexado)

**Cómo funciona:**
Una carpeta `knowledge/` en Drive con archivos MD por tema, indexados:

```
knowledge/
  KNOWLEDGE_INDEX.json        ← índice de todos los fragmentos
  KF_001_native_dims.md       ← por qué existen las 10 NATIVE_DIMS
  KF_002_minilm_backend.md    ← estado del backend semántico
  KF_003_evaluacion_2026.md   ← síntesis de evaluacion_proyecto_2026_06_17
  KF_004_antipatterns.md      ← JSON-disfrazado, strings-como-dims, etc.
  KF_005_mesh_architecture.md ← decisiones de V0.5 Mesh
  ...
```

**KNOWLEDGE_INDEX.json:**
```json
{
  "KF_001": {"titulo": "NATIVE_DIMS — origen y decisiones", "tags": ["dims","diseño"], "fileId": "..."},
  "KF_002": {"titulo": "MiniLMBackend — estado y portabilidad", "tags": ["backend","semántica"], "fileId": "..."},
  ...
}
```

Al startup, cada IA lee el índice y decide qué fragmentos son relevantes
para su tarea actual. Puede buscar por tags.

**Pros:**
- Búsqueda por tags en el índice (O(1) lookup)
- Modular — cada fragmento se actualiza sin tocar los otros
- Acumulativo — cada sesión puede agregar fragmentos nuevos
- Legible por humanos y por AIs
- No requiere infraestructura externa

**Contras:**
- Búsqueda por tags es léxica, no semántica
- Requiere disciplina para mantener el índice actualizado
- Startup más complejo (leer índice + decidir qué fragmentos leer)
- No detecta redundancia entre fragmentos

**Veredicto:** Solución media práctica. Viable en 1-2 semanas.
Requiere TASK_033 de implementación y TASK_034 de curación inicial.

---

### OPCIÓN C: Base de Datos Vectorial (LanceDB + FalkorDB)

**Cómo funciona:**
Replicar el sistema MEMORY_FABRIC_CONTRACT_V5_00 de MPAT5 para POLYDIM:
- Cada fragmento de conocimiento = un Fragment con embedding R^N
- `query_by_similarity(embedding, tenant_id="polydim")` al startup
- warm_start recupera los top-N fragmentos más relevantes para la tarea
- Engrams (episodios inmutables) para decisiones permanentes

**Estructura de un Fragment de conocimiento:**
```python
Fragment(
  fragment_id = "KF_001_native_dims",
  tenant_id = "polydim_shared",   # compartido por los 10 alumnos
  embedding = encode("NATIVE_DIMS decisión diseño"),  # R^N
  content_hash = hash(contenido),
  fragment_type = "design_decision",  # | "antipattern" | "evaluation" | "finding"
  agent_id = "curso04.mithril",
  session_id = "015"
)
```

**Pros:**
- Búsqueda semántica real — "qué se decidió sobre props" encuentra KF_001
- warm_start automático — cada IA recibe el contexto más relevante
- Escala a miles de fragmentos sin degradar el startup
- Detecta similitud entre fragmentos (evita redundancia)
- Es el propio dominio del proyecto — POLYDIM estudiando su propia memoria

**Contras:**
- Requiere infraestructura: LanceDB + FalkorDB corriendo
- Requiere modelo de embedding para encode() — depende de TASK_028
- Complejidad de setup — varias horas de trabajo de infraestructura
- Dependencia externa al Drive — si el servicio falla, los alumnos no arrancan
- La semántica del embedding tiene que ser la misma para todos los alumnos

**Veredicto:** Solución óptima a largo plazo. Bloqueada hasta TASK_028.

---

### OPCIÓN D: POLYDIM como su propia memoria (meta-opción)

**Concepto:**
Usar POLYDIM's SemanticSpace (TASK_028) para almacenar conocimiento POLYDIM.
Cada fragmento de conocimiento es un ObjectND.
La recuperación es `Session.query_similar(obj, top_k=5)`.

```python
# Un alumno almacena un hallazgo:
sp = SemanticSpace("polydim_shared")
hallazgo = ObjectND(sp)
  .add("DIM_META", {"tipo": "antipattern", "id": "AP_001"}, w=1.0)
  .add("DIM_PYTHON", {"concepto": "JSON_disfrazado"}, w=0.8)
  .add("DIM_VECTOR", {"contenido": encode("no usar dicts como props")}, w=1.0)

# Otro alumno recupera:
relevantes = mesh.query_similar(hallazgo_actual, top_k=5)
```

**Pros:**
- POLYDIM comiendo su propio dog food
- La geometría del espacio DEFINE la relevancia, no tags o keywords
- Completamente nativo al proyecto
- Demuestra que POLYDIM funciona como lenguaje de IA real

**Contras:**
- Requiere TASK_028 (SemanticSpace) + TASK_032 (compilación) completadas
- Circular al principio: necesitamos POLYDIM para aprender POLYDIM
- Requiere persistencia de hipervectores (formato binario — ver PEND_001)

**Veredicto:** Visión correcta, horizonte 2-3 meses.

---

## 4. RECOMENDACIÓN — ARQUITECTURA EN 3 CAPAS

No elegir UNA opción. Implementar las tres en secuencia:

```
CAPA 1 (HOY — 1 día):     Skills actualizadas
CAPA 2 (SEMANA 1 — 3 días): Knowledge Fragments indexados
CAPA 3 (FUTURO — post TASK_028): SemanticSpace como memoria
```

---

### CAPA 1: Skills actualizadas (implementar hoy)

**Acción inmediata:** Actualizar polydim-core SKILL.md a V9_0:
- Runtime actual: V0.5 (no V0.3)
- 10 NATIVE_DIMS (incluyendo DIM_CONTRACT)
- Nuevos IDs de Drive (BACKLOG_V24, polydim_runtime_v05.py)
- Backends disponibles: MockSemanticBackend, MiniLMBackend (en V0.3 — portar a V0.5)
- Referencia a evaluacion_proyecto_2026_06_17.md y su síntesis

**Nueva skill a crear:** `polydim-antipatterns/SKILL.md`:
```
ANTIPATRÓN 1: Dimensiones como strings humanos
  MAL:  "DIM_SQL" hardcodeado = Python con peluca vectorial
  POR QUÉ: Sin semántica intrínseca. Ver evaluacion_2026_06_17 §Pregunta 1.
  ALTERNATIVA: SemanticSpace (TASK_028 — en progreso)

ANTIPATRÓN 2: Props como diccionarios JSON
  MAL:  {"tabla": "usuarios"} = JSON disfrazado de hipervector
  POR QUÉ: La propiedad debería ser posición en subespacio, no par clave-valor.
  ALTERNATIVA: GeometricProps (TASK_029 — pendiente)

ANTIPATRÓN 3: VERB_SPACE como instrucciones
  MAL:  VERB_LEER, VERB_CREAR = palabras humanas mapeadas a vectores
  POR QUÉ: Las transformaciones no necesitan nombre. La geometría ES la instrucción.
  ALTERNATIVA: Transform sin nombre (TASK_030 — pendiente)

ANTIPATRÓN 4: to_symbolic() como parte del lenguaje
  MAL:  to_symbolic() en runtime core = Capa S como muleta
  POR QUÉ: Debería ser debugger externo, no parte del lenguaje.
  ALTERNATIVA: PolydimDebugger (TASK_031 — pendiente)
```

---

### CAPA 2: Knowledge Fragments indexados (TASK_033)

**Estructura en Drive:**
```
polydim/knowledge/          ← nueva carpeta
  KNOWLEDGE_INDEX.json      ← índice master
  KF_001_native_dims.md
  KF_002_backends.md
  KF_003_evaluacion_antigravity.md
  KF_004_antipatterns.md
  KF_005_mesh_decisions.md
  KF_006_memory_architecture.md  ← este documento, condensado
```

**Formato de cada KF:**
```markdown
# KF_XXX — [título]
**Tags:** [tag1, tag2, tag3]
**Fecha:** YYYY-MM-DD
**Sesión origen:** RELAY_0XX
**Vigencia:** ACTIVO | SUPERADO_POR_KF_YYY

## Qué es
[1-3 párrafos]

## Por qué importa
[impacto en el proyecto]

## Anti-patrones relacionados
[si aplica]

## Pendientes
[qué queda por resolver]
```

**Startup con Knowledge Fragments:**
```
1. Leer SKILL.md (siempre)
2. Leer KNOWLEDGE_INDEX.json (30 segundos extra)
3. Filtrar KFs relevantes a la tarea actual (por tags del BACKLOG)
4. Leer los 3-5 KFs más relevantes
```

**Quién actualiza el índice:**
Cualquier alumno puede agregar un KF nuevo al final de su sesión,
siguiendo el mismo protocolo CONCURRENCY RULE del BACKLOG.

---

### CAPA 3: SemanticSpace como memoria nativa (post TASK_028)

Una vez que SemanticSpace funciona, los Knowledge Fragments migran:
- Cada KF se convierte en un ObjectND en un SemanticSpace compartido
- warm_start reemplaza la lectura manual del índice
- La recuperación es semántica, no por tags
- Los 10 alumnos contribuyen al mismo espacio de conocimiento

---

## 5. TAREAS DERIVADAS

| TASK | Tipo | Acción | Depende de |
|---|---|---|---|
| **TASK_033** | DOCS | Crear carpeta knowledge/ + KNOWLEDGE_INDEX + KF_001 a KF_006 | — |
| **TASK_034** | DOCS | Actualizar SKILL.md a V9_0 (runtime V0.5, IDs actuales, anti-patrones) | TASK_033 |
| **TASK_035** | DOCS | Crear polydim-antipatterns/SKILL.md con los 4 anti-patrones centrales | TASK_033 |
| *(futuro)* | CORE | Migrar Knowledge Fragments a SemanticSpace como ObjectND | TASK_028 |

---

## 6. RESPUESTA DIRECTA A LA PREGUNTA

> "¿Skill en MD o base de datos vectorial?"

**Ambas, en secuencia:**

1. **Ahora:** Skill MD mejorada (Capa 1) + Knowledge Fragments indexados (Capa 2).
   Sin infraestructura extra. Funciona con los 10 alumnos hoy.

2. **Después de TASK_028:** Base de datos vectorial nativa en POLYDIM (Capa 3).
   El proyecto usa su propio lenguaje para recordar cómo fue construido.
   Ese es el momento en que POLYDIM se convierte en algo genuinamente nuevo.

**La ironía productiva:** La pregunta "¿cómo recordamos lo que aprendimos sobre POLYDIM?"
es exactamente el problema que POLYDIM debería resolver.
Que la respuesta sea "usando POLYDIM" cuando esté listo es la prueba de que funciona.

---

## 7. PRÓXIMO PASO INMEDIATO

Tomar TASK_033 (Knowledge Fragments) y TASK_034 (actualizar skill).
Ambas pueden ejecutarse en paralelo con TASK_027 (MiniLMBackend research).

---
*RESEARCH_MEMORIA_10_ALUMNOS.md · curso04.mithril · 2026-06-17 · POLYDIM sesión 015*
