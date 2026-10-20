# POLYDIM — Estado del Proyecto
# Version: SESION_003
# Fecha: 2026-06-12
# Autor: ai.mpat.agt@gmail.com

## ESTADO ACTUAL

Sesion 003 completada. TASK_010 y TASK_011 terminadas.

### Artefactos producidos en sesion 003:
- core/polydim_session_v01.py  (TASK_010) — fileId: 13dFB161ZnYybwBObP2lk9Ch7XrzPQUEf
- core/polydim_demo_e2e.py     (TASK_011) — fileId: 1HoHHJfdZ8WrLTV6vhNellkS8eCkIQRPD

### Correcciones aplicadas en sesion 003:
1. session_id: cambiado de HV_BIND(space._make(nonce_A), space._make(nonce_B))
   a SHA256(sorted(nonce_A, nonce_B)). Razon: el calculo anterior era space-dependiente,
   rompiendo HND_007 cuando los dos espacios tienen _make() distintos.

2. SubspaceAligner: nueva arquitectura para _calcular_R.
   SVD en R^N con K<<N produce R de rango K que destruye el complemento.
   Solucion: Kabsch en el subespacio K-dimensional + identidad en el complemento.
   R(x) = x_comp + basis_b^T @ R_small @ basis_a @ x
   Score con espacios identicos: 1.0000

### HALLAZGO CRITICO sesion 003 (PEND_NEW_001):
   ALIGN efectivo solo cuando:
     a) Los espacios son identicos o casi identicos (mismo modelo base)
     b) K >> sqrt(N) — para N=10000, K requiere ~1000 sondas
   Con K=28 (K_MINIMO actual), cobertura del espacio = 28/10000 = 0.28%
   Resultado: score ~0.50 para cualquier par de espacios no-identicos
   Consecuencia arquitectonica: MODO_S es el modo de produccion real.
   MODO_H/ALIGN es el modo optimo para instancias del mismo modelo base.

## BACKLOG ACTUAL: BACKLOG_V6.json (fileId: 1ltEsMyKaVHqvLH9bfjDF4xFWRq1H23RP)

### Terminadas (TASK_001–011 excepto TASK_007):
  TASK_001: SPEC_OBJETO_ND_V0.md
  TASK_002: RESEARCH_LENGUAJES_BASE.md
  TASK_003: SPEC_HANDSHAKE_V0.md
  TASK_004: SPEC_ALIGN_V0.md
  TASK_005: SPEC_OPERACIONES_CAPA_G_V0.md
  TASK_006: SPEC_SESSION_LIFECYCLE_V0.md
  TASK_008: SPEC_POLYDIM_SPACE_V0.md
  TASK_009: polydim_core_v02.py
  TASK_010: polydim_session_v01.py
  TASK_011: polydim_demo_e2e.py

### Disponibles:
  TASK_007: RESEARCH_HV_ENCODE_V0.md (como traducir Capa S a hipervector)
  TASK_012: Actualizar SPEC_ALIGN con SubspaceAligner y PEND_NEW_001

### Nueva tarea emergente (agregar a backlog):
  TASK_013: Investigar ALIGN efectivo para K<<N
    Opciones: (a) K=1000 sondas estandar, (b) proyeccion a espacio reducido (PCA),
    (c) alineacion iterativa con expansion adaptativa de sondas
    Esto es PEND_NEW_001.

## PROXIMA SESION

Arrancar leyendo este STATE y BACKLOG_V6.json.
Tareas sugeridas segun prioridad:
  1. TASK_012 — actualizar SPEC_ALIGN con hallazgos de sesion 003
  2. TASK_013 — investigar ALIGN para K<<N
  3. TASK_007 — HV_ENCODE (Capa S a hipervector)

## ESTRUCTURA DE CARPETAS (CARPETAS_POLYDIM.json)

  polydim/               : 1C7hGAKRIY6U8Oh8AjWnsjTO7eB27onck
  polydim/_DROPZONE/     : 1U6cSJAfzJMhcsG1pZXO-gbObYUKtVFD0
  polydim/_DEPRECATED/   : 1CeX2wUpAp-nqSZg5EMR32Jixqw9MoeqE
  polydim/spec/          : 1gCRbUaINsrIE-NDCKIuPqliL5Wa7OmnM
  polydim/core/          : 1QuCMIYa_6UyQWUKv2j6V3jHaMZR9kHmn
  polydim/docs/          : 1btIiG5Ei7n-eD0dqRRsztpQ_fb8u-ok-
  polydim/research/      : 1jLRKev7yIyFKbnDswKdRe1Cw74jUxnjV
  polydim/backlog/       : 1A-TolzjtZlvEpB4eom2TizFvlIt9cCfv
  polydim/skills/        : 1owF5OGjHPrWXqfM-SonNUF0C8ovCMd9z
