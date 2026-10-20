POLYDIM — RELAY SESION 008
Fecha: 2026-06-13 — Claude Sonnet 4.6

TASK_007 TERMINADA — RESEARCH_HV_ENCODE_V0.md
fileId: 1apniggMNh5uXMbS6yceMEeBwIsdUwScu

CONTENIDO:
  Dos partes independientes de HV_ENCODE: nombres de dimension (A) y props (B).

  PARTE A — nombres:
    A1 VSA deterministica (actual): MD5+seed+normal. Sin proximidad semantica.
    A2 VSA+clustering semantico: grupos manuales, proximidad parcial.
    A3 Embedding model: proximidad real, requiere modelo estandar externo.
    Decision V0.1: A1 correcto y suficiente. align_transform compensa.

  PARTE B — propiedades:
    B1 Cast a string (actual): universal, simple, "42"·"43"=0.
    B2 Codificacion sinusoidal numerica: hv_num(42)·hv_num(43)~0.999.
    B3 Tipo explicito en BIND: props verbosas, decodificacion exacta.
    B4 Props anidadas recursivas: BIND jerarquico.
    Decision V0.1: B1 correcto. Extension B2 recomendada, ~20 lineas.

  Tabla de decision incluida para alumnos.
  PEND_NUEVA_004: modelo embedding estandar POLYDIM (V1.0).
  PEND_NUEVA_005: props anidadas B4 (baja prioridad, bajo riesgo).

ESTADO:
  TASK_001..016: TERMINADAS
  TASK_017 (META_PERMISSIONS): prioridad 1, disponible
  TASK_018 (B2 numerica): prioridad 2, disponible

ARTEFACTOS:
  research/RESEARCH_HV_ENCODE_V0.md  1apniggMNh5uXMbS6yceMEeBwIsdUwScu
  backlog/BACKLOG_V11.json            1av5bTFu_EodlFoKHAntlbjgTJ52N6XDS
  docs/RELAY_SESION_008.md            [este archivo]

PROXIMA PRIORIDAD: TASK_017 (META_PERMISSIONS + runtime_v03)
RELAY_SESION_008.md — 2026-06-13
