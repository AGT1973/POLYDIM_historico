# POLYDIM_DEST
# destino: POLYDIM/research/
# nombre: RESEARCH_DIM_MEMORY_MPAT5.md
# alumno: claude-sonnet-4-6

# RESEARCH_DIM_MEMORY_MPAT5.md
## Verificación de DIM_MEMORY contra dominio MPAT5 memory_unikernels/

**Fecha:** 2026-06-16
**Tarea:** TASK_025
**Autor:** claude-sonnet-4-6
**Estado:** COMPLETO

---

## 1. PREGUNTA DE INVESTIGACIÓN

`polydim_runtime_v04.py` dejó pendiente la siguiente decisión:

> *"memory_unikernels NO se agrega todavía — no está claro si describe una
> propiedad semántica del objeto o solo infraestructura de despliegue
> (donde corre, no qué es). Pendiente de verificación directa contra MPAT5."*

La pregunta es: **¿debe existir DIM_MEMORY como NATIVE_DIM en POLYDIM?**

---

## 2. EVIDENCIA RECOLECTADA DE MPAT5

### 2.1 MEMORY_FABRIC_CONTRACT_V5_00.md (TASK_010 — TERMINADA)

Módulo `memory_unikernels/` en MPAT5. Contrato adoptado. Backend: LanceDB (vectores) + FalkorDB (grafo).

Operaciones nucleares:
- `upsert_fragment(fragment, graph_relations)` — escribe un fragmento de conocimiento
- `query_by_similarity(embedding, tenant_id)` — recupera por similitud semántica
- `store_engram(engram)` — episodio inmutable en FalkorDB
- `warm_start(tenant_id, ecs_id)` — recupera top-N fragmentos para iniciar sesión

Un **Fragment** tiene: `fragment_id`, `tenant_id`, `embedding` (vector R^N),
`content_hash`, `fragment_type`, `agent_id`, `session_id`.

Un **Engram** (episodio) es inmutable. Nunca hard-delete en sesión activa (INV-MEM-009).

### 2.2 CONTRATO_AESP_V1_00.md (TASK_008 — TERMINADA)

AESP clasifica `memory.fragment.upsert` como acción de riesgo **MEDIO**
(requiere CHECK_1 a CHECK_7). Esto confirma que las operaciones de memoria
son acciones de primera clase con identidad verificable en el sistema.

### 2.3 CONTRATO_COGNITIVE_EVENT_MESH.md (TASK_002 — TERMINADA)

Eventos memory.* que circulan por el CEM:
- `memory.fragment_stored`
- `memory.fragment_retrieved`
- `memory.episode_stored`
- `memory.warm_start_executed`
- `memory.quota_warning`
- `memory.quota_exceeded`

Estos eventos **tienen estructura propia** (tenant_id, fragment_id, embedding,
importance) que es transferible entre agentes como payload.

### 2.4 POLYDIM_SPEC_RESUMEN.md + polydim_runtime_v04.py

NATIVE_DIMS actuales: DIM_PYTHON, DIM_RUST, DIM_FLUTTER, DIM_SQL,
DIM_GRAPH, DIM_VECTOR, DIM_TIME, DIM_ERROR, DIM_META, DIM_CONTRACT.

DIM_GRAPH cubre grafos en general (FalkorDB incluido).
DIM_VECTOR cubre embeddings vectoriales (LanceDB incluido).

---

## 3. ANÁLISIS: ¿ES memory_unikernels UNA DIMENSIÓN SEMÁNTICA?

### Criterio de NATIVE_DIM en POLYDIM

Una NATIVE_DIM responde a: **"¿este objeto ES algo de este tipo?"**
- DIM_SQL: ¿es una fila/tabla SQL?
- DIM_PYTHON: ¿es un objeto Python?
- DIM_CONTRACT: ¿es un acuerdo/contrato entre agentes?
- DIM_VECTOR: ¿es un embedding vectorial?

### Análisis de memory_unikernels

Un **Fragment** de MPAT5 tiene dos naturalezas simultáneas:
1. **ES un vector** (embedding R^N) → ya cubierto por `DIM_VECTOR`
2. **ES un nodo de grafo** (Engram en FalkorDB) → ya cubierto por `DIM_GRAPH`

La capa `memory_unikernels` describe **dónde viven** los fragmentos
(LanceDB + FalkorDB en Firecracker) y **cómo se accede** a ellos,
no una propiedad semántica propia de los objetos.

### Contraargumento evaluado

Se podría argumentar que un objeto puede **SER un fragmento de memoria**
(Fragment) de forma análoga a como puede SER un contrato (DIM_CONTRACT).
¿Justifica esto DIM_MEMORY?

**Respuesta:** DIM_CONTRACT se justifica porque un contrato tiene propiedades
semánticas únicas no reducibles a otras dimensiones (partes, invariantes,
estado ADOPTADO/PENDIENTE). Un Fragment MPAT5, en cambio, se describe
completamente con DIM_VECTOR (su embedding) + DIM_GRAPH (sus relaciones Engram)
+ metadatos que DIM_META puede representar. No hay residuo semántico sin cubrir.

---

## 4. CONCLUSIÓN

| Pregunta | Respuesta |
|----------|-----------|
| ¿memory_unikernels es infraestructura o semántica? | **Infraestructura** de despliegue |
| ¿Un Fragment MPAT5 necesita DIM_MEMORY? | **No** — DIM_VECTOR + DIM_GRAPH lo cubren completamente |
| ¿Hay residuo semántico sin cubrir? | **No** — embedding=DIM_VECTOR, relaciones=DIM_GRAPH, metadatos=DIM_META |
| ¿Debe agregarse DIM_MEMORY a NATIVE_DIMS? | **NO** |

**Decisión: DIM_MEMORY no se agrega a POLYDIM.**

La intuición del v04 era correcta. `memory_unikernels` describe el *dónde*
(Firecracker + LanceDB + FalkorDB), no el *qué* del objeto.

---

## 5. IMPACTO EN polydim_runtime_v05.py

La TASK_023 (`polydim_runtime_v05.py`) puede proceder con los 10 NATIVE_DIMS
actuales sin agregar DIM_MEMORY.

Si en el futuro se necesita representar un Fragment completo como ObjectND,
el patrón recomendado es:

```python
fragment_obj = ObjectND(space)
fragment_obj.add("DIM_VECTOR", {"embedding_dim": 10000, "model": "lancedb"}, w=1.0)
fragment_obj.add("DIM_GRAPH", {"store": "falkordb", "node_type": "Engram"}, w=0.9)
fragment_obj.add("DIM_META", {"tenant_id": tid, "fragment_id": fid}, w=0.7)
```

Esto representa un Fragment MPAT5 sin necesidad de una nueva dimensión nativa.

---

## 6. PENDIENTES GENERADOS

Ninguno. Decisión cerrada. TASK_025 completa.

**Referencia cruzada:**
- MEMORY_FABRIC_CONTRACT_V5_00.md (fileId: 1wkGkqNRqD2iHRX62pgzigEIrvI2YyQtA)
- CONTRATO_AESP_V1_00.md (fileId: 1BxpIEAUdoVU4dnJNCFSfBTiEpzblQ_2A)
- polydim_runtime_v04.py (fileId: 1UTATtOYZP3M3skWIoVelpultzitIMpai)
