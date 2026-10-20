POLYDIM — RELAY SESION 008
Fecha: 2026-06-13 — Claude Sonnet 4.6
Input: "." x2 — continuacion directa desde sesion 007

TRABAJO REALIZADO
─────────────────

TASK_007 — TERMINADA
 RESEARCH_HV_ENCODE_V0.md
 fileId: 1Yjfde9EkN_r7jIZIXSsR2v4xM9s2HC4y

 Contenido:
 - Análisis del estado actual: ObjectND._hv() hace codificación implícita acumulativa
   SUPERPOSE(geo_id, sub(d)*w, BIND(sub(d),enc(p))*w*CONTENT_W)
 - Tres opciones evaluadas:
   3A. VSA Determinística (RECOMENDADA para V1): sin deps, reproducible, compatible con ALIGN
   3B. Embedding Semántico (V2+): semántica rica pero rompe determinismo, requiere Cap.EMBEDDING
   3C. Codificación Aprendida con personal_seed (V3+): Hebbian, alta complejidad, riesgo drift
 - Extensión propuesta V1:
   encode_typed(space, val) — preserva int/float/bool/list/dict
   encode_props(space, props, depth, max_depth=2) — recursivo hasta depth=2
   Space.query_content(hv, dim, vocab) — BIND inverso aprox, recupera claves con score > UMBRAL
   CONTENT_W ajustable por dim en ObjectND.add()
 - 5 invariantes nuevas: HVE_001-005
 - 5 tareas derivadas: TASK_018-022

ESTADO DEL NUCLEO (post sesion 008)
─────────────────────────────────────

Módulos ejecutables:
 polydim_runtime_v02fix.py  1EQ1bhBN8MQqnWtvFzNVia9B5RhlPFDs6  CANONICAL para alumnos

Research:
 RESEARCH_AUTOPROG_V1.md    1gyaNUTC3UOK18jEwxp95bQgq3fS5acD2  META semántica
 RESEARCH_HV_ENCODE_V0.md   1Yjfde9EkN_r7jIZIXSsR2v4xM9s2HC4y  HV encoding — NUEVO

Backlog activo:
 BACKLOG_V11.json           1o5kHjnchdVbTQwoN4-7BC0VAWXAOxPbW  ACTIVO

Skill en disco:
 polydim-core-skill V3_0 (local) — diverge de V6_0 en Drive (1Ji8JcDsSHE_Wxh7Y2kAZ9MDG7AHElBDL)
 PENDIENTE: actualizar skill en disco con IDs actualizados (BACKLOG_V11, runtime_v02fix, etc.)

PENDIENTES CLAVE
────────────────
TASK_017: META_PERMISSIONS en InitMsg (prioridad 2)
TASK_018: encode_typed + encode_props en polydim_runtime_v03.py (prioridad 1)
TASK_019: Space.query_content(hv, dim, vocab) (prioridad 1, depende 018)
TASK_020: test recuperación encode→transfer→query_content (prioridad 2, depende 019)

PROXIMA SESION
──────────────
Opción A (core V1): TASK_018 — encode_typed + encode_props → polydim_runtime_v03.py
Opción B (spec):    TASK_017 — META_PERMISSIONS → SPEC_META_PERMISSIONS_V1.md + v03

RELAY_SESION_008.md — 2026-06-13 — TASK_007 TERMINADA
