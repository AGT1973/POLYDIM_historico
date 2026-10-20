# TASK_BUG_009 — Duplicado de BACKLOG_V18.json en Drive

**Fecha:** 2026-06-19
**Tipo:** Limpieza / deuda de proceso (mismo patron que BUG_007)

## Que paso

Se subieron dos versiones de BACKLOG_V18.json por la misma limitacion de
Drive que afecto al Rust deprecado: no hay `delete` ni `update` in-place,
solo `create_file` (siempre crea un archivo nuevo) y `copy_file`.

- fileId 1Cy7UwvuIE1cUg8ihjcvK2XOXZDfLnc4J — version VIEJA
  No tiene la ubicacion real del Rust deprecado (no menciona
  POLYDIM/_DEPRECATED/, solo la copia de advertencia en core/).

- fileId 1KrouvIZ7VeBNRX10jHl4Vw6DsFdKy620 — version VIGENTE
  Incluye copia_en_DEPRECATED apuntando a 12kypwGryqVBZ77Em-mcsYIY0dmEph2u0.

## Resolucion

VIGENTE = 1KrouvIZ7VeBNRX10jHl4Vw6DsFdKy620. Usar este fileId en cualquier
referencia futura a "el backlog V18". El fileId viejo (1Cy7UwvuIE1cUg8ihjcvK2XOXZDfLnc4J)
queda obsoleto; igual que con el Rust, el borrado fisico requiere accion
manual en Drive web (no hay permiso de delete via MCP en esta sesion).

## Patron general (aplica tambien a BUG_007: multiples BACKLOG_V11 sin deprecar)

Cada vez que un archivo necesite corregirse despues de subido, create_file
genera un fileId NUEVO, nunca sobrescribe. La convencion para esta sesion
en adelante: el RELAY de cada sesion debe decir explicitamente cual fileId
es el vigente para cada nombre de archivo logico (BACKLOG, RELAY, etc.),
igual que se hizo para polydim_core.rs en TASK_023_RUST_LEGACY.
