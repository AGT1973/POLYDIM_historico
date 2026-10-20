# POLYDIM_DEST
# destination: polydim/research/
# filename:    RESEARCH_MINIML_BACKEND.md
# author:      ai.mpat.agt@gmail.com

# RESEARCH — Backend Semántico en POLYDIM (TASK_027)
**Fecha:** 2026-06-18
**Origen:** evaluacion_proyecto_2026_06_17.md §1 + POLYDIM_BASES_V1.md PASO_1
**Estado:** TERMINADA — backend portado a v08

---

## 1. EL PROBLEMA

`polydim_runtime_v03.py` introdujo `SemanticBackend` (con implementaciones
`MockSemanticBackend`, `MiniLMBackend`, `FastTextBackend`) que permitía a
`Space` generar subespacios semánticamente relacionados, en lugar de
hashes aleatorios sin relación entre sí.

Resultados verificados en V0.3 (documentados en el header del archivo):
```
usuario↔cliente  sem=0.830  det=0.506   (mismo grupo semántico)
usuario↔tabla    sem=0.512  det=0.493   (grupos distintos)
ALIGN sem↔det:   0.923      valido=True (IAs con backends distintos se alinean)
```

Esta feature se perdió en el camino: V0.4 (BUG fixes), V0.5 (Mesh),
V0.6 (merge), V0.7 (PCRG-1) — ninguno la portó. `Space()` quedó limitado
a generar subespacios desde hashes de nombres (MD5 en V0.3-V0.6, PCRG-1
desde V0.7), sin relación semántica real entre "DIM_SQL" y "DIM_VECTOR",
por ejemplo, salvo la que un humano declare a mano.

---

## 2. POR QUÉ IMPORTA (POLYDIM_BASES_V1.md)

> Los subespacios deben EMERGER de un modelo de embedding real.
> [...] DIM_SQL y DIM_PYTHON estarán relacionados semánticamente
> porque "SQL" y "Python" coexisten en el espacio del lenguaje real.
> La IA no necesita que nadie le diga que están relacionados.

Sin backend semántico, cada `DIM_NOMBRE` es ortogonal a las demás por
construcción del generador aleatorio — independientemente de si los
conceptos están relacionados en el mundo real. Activar un backend es
el primer paso medible hacia AXIOMA_5 (cuasi-ortogonalidad por diseño
geométrico real, no por accidente de hash) y hacia TASK_028
(SemanticSpace con NATIVE_DIMS emergentes).

---

## 3. QUÉ SE PORTÓ

| Componente | Origen | Cambio en v08 |
|------------|--------|----------------|
| `SemanticBackend` (ABC) | v03 | sin cambios |
| `MockSemanticBackend` | v03 | centroides de grupo: MD5+PCG64 → **PCRG-1** |
| `MiniLMBackend` | v03 | sin cambios (requiere `pip install sentence-transformers`) |
| `FastTextBackend` | v03 | sin cambios (requiere `pip install fasttext-wheel`) |
| `make_jl()` | v03 | **sin cambios** — MD5+PCG64 mantenido a propósito (ver §4) |
| `Space._mk()` con backend | v03 | perturbación por personal_seed: MD5 → **PCRG-1** |

---

## 4. DECISIÓN DE DISEÑO: ¿por qué `make_jl()` NO se migró a PCRG-1?

`make_jl(d_in, d_out)` genera una matriz de proyección Johnson-Lindenstrauss
(`d_out × d_in`, entradas ~N(0,1)/√d_in) para llevar el embedding del
backend (384 dims en MiniLM, 64 en Mock) al espacio POLYDIM de N=10000.

Se decidió **no** migrarla a PCRG-1 por dos razones:

1. **No es un subespacio nombrado.** PCRG-1 resuelve BUG_001: que
   `Space("X").sub("DIM_SQL")` sea igual en Python y en Rust. La matriz JL
   no es un subespacio por nombre — es maquinaria interna del backend
   semántico, exclusiva del lado Python.

2. **Rust (`polydim_core.rs` V1.0) no implementa backends semánticos.**
   No hay ningún escenario actual donde Rust necesite reproducir esta
   matriz. Migrar sin necesidad agrega complejidad sin beneficio medible.

Si en el futuro se porta un backend semántico a Rust, `make_jl()` deberá
migrar a PCRG-1 en ese momento — queda como deuda técnica documentada,
no como bug.

---

## 5. INTEGRACIÓN CON PCRG-1 (V0.7)

`Space._mk()` en v08 tiene dos ramas:

```python
def _mk(self, name: str) -> np.ndarray:
    if self.backend:
        hv = self._JL @ self.backend.encode(name)
        if self.ps:
            key = f"{self.ps}:{name}"
            p   = pcrg_make_hv(key, N)        # antes: MD5+PCG64
            hv  = 0.85 * hv + 0.15 * p
    else:
        key = f"{self.ps}:{name}" if self.ps else name
        hv  = pcrg_make_hv(key, N)             # PCRG-1 puro (sin cambios v07)
    ...
```

La perturbación por `personal_seed` (cuando hay backend) ahora usa PCRG-1
en lugar de MD5+PCG64 — consistente con el resto del runtime desde V0.7.

---

## 6. VERIFICACIÓN (tests en v08)

```
test_semantic_clustering_mock
  sim(usuario, cliente) > sim(usuario, tabla)   → mismo grupo > grupos distintos

test_semantic_backend_with_personal_seed
  La relación semántica se preserva con personal_seed activo,
  y dos IAs con seeds distintos generan vectores distintos para el mismo nombre.

test_align_with_different_backends
  Dos Sessions con MockSemanticBackend (personal_seeds distintos)
  logran ALIGN válido y transfieren un ObjectND correctamente.
```

16 tests heredados de v07 siguen pasando sin modificación — backend=None
es el comportamiento por defecto, sin regresión.

---

## 7. CÓMO USARLO

```python
from polydim_runtime_v08 import Space, ObjectND, MockSemanticBackend, MiniLMBackend

# Desarrollo/tests, sin internet ni GPU:
sp = Space("MI_IA", semantic_backend=MockSemanticBackend())

# Producción, embeddings reales (requiere pip install sentence-transformers):
sp = Space("MI_IA", semantic_backend=MiniLMBackend())

obj = ObjectND(sp).add("DIM_SQL", {"tabla": "usuarios"})
```

---

## 8. SIGUIENTE PASO

Con el backend semántico disponible, **TASK_028 (SemanticSpace)** puede
implementar `discover_dims(corpus)` para que `NATIVE_DIMS` emerjan de
clusters del espacio de embedding, en lugar de ser una lista hardcodeada
de 10 strings (`NATIVE = ["DIM_SQL", "DIM_PYTHON", ...]`).

Esto es PASO_4 de POLYDIM_BASES_V1.md — deprecar `NATIVE` hardcodeado.

---

*RESEARCH_MINIML_BACKEND.md · POLYDIM · TASK_027 · 2026-06-18*
