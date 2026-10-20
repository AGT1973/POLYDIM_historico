# NOTE_DT_TRANSFORM_001.md
# date: 2026-06-26
# author: ai.mpat.agt@gmail.com

## DT_TRANSFORM_001 — Duplicate polydim_transform.py in core/

### Situación

Hay DOS archivos llamados `polydim_transform.py` en `polydim/core/`:

| fileId | Autor | Versión | Estado |
|--------|-------|---------|--------|
| `1cabu-jm4MXdjtNKBgHWttUqnJ6-kb8y9` | polydim.ai.lenguage@ | V1.0 original (TASK_030) | **OBSOLETO** — deprecar |
| `1BFezxUvQMdqeqlJ5NsOcIiCqQJ9RYRmJ` | ai.mpat.agt@ | V1.1 (DT_EXEC_002 ✓) | **ACTIVO** |

La V1.0 ya fue copiada a `_DEPRECATED/` como `DEPRECATED_polydim_transform_v1_0.py`
(`1wXhTHXoOs5uxFnIUCUS7BMM-KYdfY0vh`).

### Problema

El MCP de Drive no tiene `delete_file` ni `move_file` — solo `copy_file`.
El archivo original V1.0 (`1cabu-jm4MXdjtNKBgHWttUqnJ6-kb8y9`) sigue en `core/`.

Un `import polydim_transform` en Python cargará el que aparezca primero en el sistema
de archivos. En Drive esto no es determinístico sin saber el orden de listado.

### Acción requerida — MANUAL (Drive web)

1. Abrir `polydim/core/` en Drive web
2. Identificar `polydim_transform.py` con fileId `1cabu-jm4MXdjtNKBgHWttUqnJ6-kb8y9`
   (autor: polydim.ai.lenguage@gmail.com, size ~7.7KB)
3. Moverlo a `polydim/_DEPRECATED/` con nombre `DEPRECATED_polydim_transform_v1_0_orig.py`
   (la copia ya existe como `DEPRECATED_polydim_transform_v1_0.py`)
4. Confirmar que solo queda `1BFezxUvQMdqeqlJ5NsOcIiCqQJ9RYRmJ` en `core/`

### Estado

- Copia en _DEPRECATED/: ✓ (`1wXhTHXoOs5uxFnIUCUS7BMM-KYdfY0vh`)
- Original en core/: ⚠️ PENDIENTE eliminación manual
- Impacto: bajo (ambos archivos tienen la misma API pública; V1.1 tiene los fixes)
- Clasificación: BLOCKED_MANUAL — no puede resolverse desde MCP sin delete_file
