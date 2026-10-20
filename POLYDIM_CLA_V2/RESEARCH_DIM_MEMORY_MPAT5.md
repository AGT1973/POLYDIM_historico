# POLYDIM_DEST
# destino: polydim/research/
# nombre: RESEARCH_DIM_MEMORY_MPAT5.md
# autor: ai.mpat.agt@gmail.com

# RESEARCH: Verificacion de DIM_VECTOR + DIM_GRAPH contra memory_unikernels (MPAT5)
## TASK_025 — Resolucion

**Fecha:** 2026-06-17
**Estado:** RESUELTO
**Tarea:** TASK_025 (prioridad 1, research/)
**Bloqueo original:** falta de acceso de lectura a Drive MPAT5 (contracts_aesp/, memory_unikernels/) — resuelto en esta sesion.

---

## 1. OBJETIVO

Verificar si los dos dimension subspaces nativos de POLYDIM relacionados con
memoria — DIM_VECTOR (embeddings, similitud semantica) y DIM_GRAPH (nodos,
aristas, relaciones) — son suficientes para representar la arquitectura real
de memoria persistente que MPAT5 implemento en `memory_unikernels/`, o si se
necesita un DIM_MEMORY nuevo.

---

## 2. ARQUITECTURA REAL EN MPAT5 (fuente: Drive MPAT5)

```
memory_unikernels/ implementa GraphRAG en dos capas:

CAPA VECTORIAL   → LanceDB   (embeddings, busqueda por similitud, schema Arrow)
CAPA GRAFO       → FalkorDB  (relaciones, traversal, Cypher)

Flujo obligatorio (INV-GRG-003):
  query → embedding → LanceDB.search() → candidate_ids
        → FalkorDB.expand() → context_nodes → respuesta

Escritura unica (INV-GRG-002 / INV-UPSERT-001):
  upsert_fragment() = LanceDB.merge_insert() + FalkorDB MERGE (Cypher)
  Prohibido DELETE+INSERT en ambas capas.

Runtime de aislamiento: Firecracker MicroVMs, unikernel-per-tenant
  (TASK_004/TASK_014). Persistencia delegada a Drive via snapshot
  Lance/Parquet al cierre de cada unikernel (INV-GR-006).

Capa de gobierno (contracts_aesp/): AESP — 8 comprobaciones obligatorias
  (identidad, autoridad, presupuesto, contexto, aislamiento, deriva,
  integridad, biometria) antes de toda accion ALTO/CRITICA, incluyendo
  escrituras de memoria. HallucinationGuard/Metric son capas adicionales
  sobre el output, no sobre el almacenamiento en si.
```

---

## 3. MAPEO CONTRA DIMENSIONES NATIVAS POLYDIM

| Componente MPAT5 | Dimension POLYDIM | Cobertura |
|---|---|---|
| LanceDB (embeddings, ANN) | DIM_VECTOR | COMPLETA — coincide 1:1 con la definicion nativa (embeddings, similitud semantica, VSA) |
| FalkorDB (nodos, Cypher, traversal) | DIM_GRAPH | COMPLETA — coincide 1:1 (grafos, nodos, aristas, relaciones) |
| upsert_fragment() atomico (merge_insert + MERGE) | DIM_VECTOR ⊕ DIM_GRAPH combinados | CUBIERTA, pero como **operacion sincronizada entre dos dims**, no una dimension nueva |
| Firecracker unikernel-per-tenant, snapshot a Drive | DIM_META (audit, version, origin) | PARCIAL — el aislamiento por tenant y el ciclo de vida del unikernel no tienen subspace propio; hoy se fuerza dentro de DIM_META por falta de mejor encaje |
| AESP (8 checks, presupuesto, biometria) | DIM_ERROR (recovery) + DIM_META (audit) | PARCIAL — AESP es mas un protocolo de control de acceso/economico que un subspace de contenido; no mapea limpio a ningun DIM existente |
| HallucinationGuard/Metric | DIM_ERROR | COMPLETA — encaja bien en "error states, exceptions, recovery" |

---

## 4. HALLAZGO PRINCIPAL

**No se necesita un DIM_MEMORY nuevo.** La arquitectura de memoria de MPAT5
(GraphRAG = vector store + graph store) es exactamente la combinacion que
POLYDIM ya modela de forma nativa como un objeto ND con dos dimensiones
activas simultaneas:

```python
obj = ObjectND(sp)
obj.add("DIM_VECTOR", {"embedding": [...], "table": "knowledge_fragment"}, w=1.0)
obj.add("DIM_GRAPH",  {"node_id": "...", "relations": [...]}, w=1.0)
```

El upsert atomico de MPAT5 (INV-GRG-002) es, en terminos POLYDIM, una
escritura SIMULTANEA en ambas dimensiones del mismo ObjectND — el principio
central de POLYDIM ("un objeto existe en N dimensiones a la vez") cubre el
caso sin necesidad de extension.

**Gap real identificado (no es DIM_MEMORY, es otro):** ni el ciclo de vida
del unikernel (boot/snapshot/destroy por tenant) ni el protocolo de control
de acceso AESP tienen un dimension subspace nativo claro. Esto es deuda
tecnica de POLYDIM, no de MPAT5.

---

## 5. RECOMENDACION

```
NO crear DIM_MEMORY — redundante con DIM_VECTOR + DIM_GRAPH.

SI evaluar a futuro (fuera de scope de TASK_025):
  - DIM_TENANT o extension de DIM_META para ciclo de vida de unikernel
    (boot, snapshot, destroy, aislamiento).
  - DIM_GOVERNANCE o extension de DIM_ERROR para protocolos tipo AESP
    (autorizacion, presupuesto, biometria) — actualmente forzado dentro
    de DIM_ERROR/DIM_META sin encaje limpio.

Accion inmediata: ninguna modificacion a los 9 dimension subspaces nativos.
TASK_025 se cierra como verificacion con resultado "cobertura confirmada,
sin cambios necesarios en el core".
```

---

## 6. FUENTES (Drive MPAT5, solo lectura)

```
mpat5/contracts_aesp/CONTRATO_AESP_V1_00.md       (1BxpIEAUdoVU4dnJNCFSfBTiEpzblQ_2A)
mpat5/contracts_aesp/CONTRATO_GraphRAG_FalkorDB.md (1SPaV4R6kknYhcEb9obuAu0I3qSw3v2pk)
mpat5/contracts_aesp/CONTRATO_HallucinationGuard_Metric.md (13fnYX1ZrTIalFE2dJi9WZF4EFzJBzsRU)
mpat5/memory_unikernels/MEMORY_FABRIC_CONTRACT_V5_00.md   (1wkGkqNRqD2iHRX62pgzigEIrvI2YyQtA)
mpat5/memory_unikernels/CONTRATO_GraphRAG_FalkorDB.md     (1wSMNdbGUq1h50Vfl9fo2UXTVlddYAkPo)
mpat5/memory_unikernels/RESEARCH_NanoVMs_vs_Firecracker.md (1nSKuEhb0wPoV7ORwblLYzj0337SK-EjH)
```

---
*RESEARCH_DIM_MEMORY_MPAT5.md · TASK_025 · 2026-06-17 · ai.mpat.agt@gmail.com*
