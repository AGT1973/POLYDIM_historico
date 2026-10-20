# TASK_025 — Cierre de investigación DIM_MEMORY

**Fecha:** 2026-06-17
**Estado:** RESUELTO — sin cambios de código necesarios

## Decisión

NO se agrega DIM_MEMORY a NATIVE_DIMS.

## Evidencia (dos investigaciones independientes, mismo veredicto)

Un Fragment de memory_unikernels/ (MPAT5) tiene dos naturalezas simultáneas:
1. ES un vector (embedding) -> ya cubierto por DIM_VECTOR
2. ES un nodo de grafo (Engrama en FalkorDB) -> ya cubierto por DIM_GRAPH

memory_unikernels describe DONDE viven los fragmentos (LanceDB + FalkorDB en
Firecracker) y COMO se accede, no una propiedad semantica nueva del objeto.

Patron recomendado para representar un Fragment como ObjectND:

```python
fragment_obj = ObjectND(space)
fragment_obj.add("DIM_VECTOR", {"embedding_dim": 10000, "model": "lancedb"}, w=1.0)
fragment_obj.add("DIM_GRAPH", {"store": "falkordb", "node_type": "Engrama"}, w=0.9)
fragment_obj.add("DIM_META", {"tenant_id": tid, "fragment_id": fid}, w=0.7)
```

## Deuda real identificada (NO es DIM_MEMORY, es otra cosa)

Ni el ciclo de vida del unikernel (boot/snapshot/destroy por tenant) ni el
protocolo AESP (control de acceso/economico) tienen subespacio nativo limpio.
Quedan parcialmente forzados dentro de DIM_META / DIM_ERROR sin buen encaje.
Candidatos a futuro (fuera de scope de TASK_025): DIM_TENANT, DIM_GOVERNANCE.

## Nota de arquitectura (de evaluacion_proyecto_2026-06-17.md, Antigravity)

Punto mas importante de la auditoria externa: los NATIVE_DIMS hoy se generan
con `hashlib.md5(nombre_humano)` -> vector aleatorio determinista. Esto da
ortogonalidad por construccion, no porque "SQL" y "Python" sean conceptos
independientes en algun sentido profundo. La direccion correcta a largo
plazo es que los subespacios EMERJAN de un modelo de embedding semantico real
(el proyecto ya tuvo un MockSemanticBackend / MiniLMBackend en una bifurcacion
de V04 que se abandono), no que se sigan agregando mas DIM_NOMBRE con strings
humanos. Esto no bloquea TASK_025 pero enmarca por que la respuesta es "no" a
DIM_MEMORY: agregar una dimension mas con nombre humano sin evidencia semantica
fuerte repetiria exactamente el patron que la auditoria senala como error.

## Fuentes
- RESEARCH_DIM_MEMORY_MPAT5.md (ai.mpat.agt, 17/06) - fileId 1k3DGhHbi_qO3OaTlNmI0YmVski957_V5
- RESEARCH_DIM_MEMORY_MPAT5.md (claude-sonnet-4-6, 16/06) - fileId 1ap1Kx0Tr7df__yrLdUKAqVWeqRdf42Fx
- evaluacion_proyecto_2026_06_17.md (Antigravity) - fileId 1prwJ9iVPkryLqOz72Q_sLxejf2r2tsCB
