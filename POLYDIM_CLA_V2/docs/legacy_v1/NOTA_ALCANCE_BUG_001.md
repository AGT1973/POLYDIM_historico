# Nota de alcance — fix BUG_001/BUG_006 vs. deuda semántica de NATIVE_DIMS

**Aplica a:** polydim_core.rs V0.2 (TASK_023)
**Fecha:** 2026-06-17/18

## Qué resuelve el fix de BUG_001 (NATIVE_SYNC)

Un problema de **interop binario**: Python (PCG64) y Rust (LCG+Box-Muller propio)
generaban hipervectores distintos para el mismo seed numérico. El fix hace que
Rust ya no genere los 9 `NATIVE_DIMS` localmente — los recibe de Python vía
`sync_native()` durante el handshake, garantizando que ambos lados usen
exactamente el mismo vector para el mismo concepto nativo.

## Qué NO resuelve (y no pretende resolver)

La evaluación `evaluacion_proyecto_2026-06-17.md` (Antigravity) señala un
problema distinto, de fondo: los `NATIVE_DIMS` se generan con
`hashlib.md5(nombre_humano)` → vector aleatorio determinista. Esto da
ortogonalidad **por construcción**, no porque "SQL" y "Python" sean conceptos
independientes en un sentido semántico real. Es "Python con peluca vectorial",
en palabras de esa evaluación.

NATIVE_SYNC garantiza que Rust y Python coincidan en *qué* vector arbitrario
usan para `DIM_SQL`. No garantiza que ese vector capture nada intrínseco sobre
qué es SQL. Son dos problemas distintos:

| Problema | Resuelto por |
|---|---|
| Python y Rust generan HVs distintos para el mismo seed | BUG_001 fix (NATIVE_SYNC) — ✅ resuelto |
| Los subespacios nativos no tienen semántica intrínseca, son hashes arbitrarios | Sin asignar a ningún TASK todavía — ❌ no resuelto |

## Por qué esto no bloquea TASK_023

La investigación independiente `RESEARCH_DIM_MEMORY_MPAT5.md` (TASK_025, dos
versiones convergentes) confirmó que la lista de 9 `NATIVE_DIMS` no debe
crecer sin evidencia semántica sólida — ni siquiera para un caso aparentemente
justificado como `DIM_MEMORY`. Por el mismo criterio, este fix tampoco intenta
simular semántica real donde no la hay: mantiene las 9 dimensiones existentes,
corrige solo la mecánica de generación/transmisión para que sea consistente
entre lenguajes.

## Dirección futura (no asignada a un TASK numerado todavía)

Los subespacios nativos deberían derivarse de un modelo de embedding semántico
real (sentence-transformers, o el espacio de embedding propio de un LLM), no
de hashes de strings. El proyecto tuvo un `MockSemanticBackend`/`MiniLMBackend`
en una bifurcación de V04 que se abandonó — esa es la dirección correcta para
retomar cuando se aborde esta deuda.
