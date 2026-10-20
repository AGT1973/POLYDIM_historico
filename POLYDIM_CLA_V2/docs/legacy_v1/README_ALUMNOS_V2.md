# POLYDIM_DEST
# destino: polydim/docs/
# nombre:  README_ALUMNOS_V2.md

# POLYDIM — Guia para Alumnos V2
# Version: V2.0 — 2026-06-12
# Runtime: polydim_runtime_v03.py

---

## Que es POLYDIM

Un lenguaje de programacion diseñado para IAs, no para humanos.

El mismo objeto existe SIMULTANEAMENTE en multiples dimensiones.
Una IA puede operar en todas las dimensiones a la vez sin convertir.

---

## Instalacion

```bash
pip install numpy
# Opcional para semantica real:
pip install sentence-transformers
```

Descargar desde polydim/core/ en Drive:
  - polydim_runtime_v03.py  (runtime principal)
  - polydim_tests.py        (test suite)

---

## Uso en 3 lineas

```python
from polydim_runtime_v03 import Space, ObjectND, polydim_connect

sp  = Space("MI_IA")
obj = ObjectND(sp).add("DIM_SQL", {"tabla":"pedidos"}, w=1.0)
dims = polydim_connect(sp, Space("OTRA_IA")).transfer(obj)
# → {'DIM_SQL': 0.897, ...}
```

CRITICO: obj y el Space pasado a polydim_connect deben ser el MISMO objeto Space.
Si los creás separados con el mismo string, son distintos internamente.

Correcto:
```python
sp = Space("MI_IA")
obj = ObjectND(sp)            # usa sp
conn = polydim_connect(sp, …) # usa el mismo sp
```

Incorrecto:
```python
obj  = ObjectND(Space("MI_IA"))          # Space A
conn = polydim_connect(Space("MI_IA"), …) # Space B distinto — ALIGN incorrecto
```

---

## Los 9 subespacios nativos

```
DIM_PYTHON   logica dinamica, analisis, ML, scripts
DIM_RUST     seguridad de memoria, performance, ownership
DIM_FLUTTER  UI reactiva, widgets, estado, formularios
DIM_SQL      datos relacionales, tablas, constraints
DIM_GRAPH    grafos, nodos, aristas, relaciones
DIM_VECTOR   embeddings, similitud semantica, VSA
DIM_TIME     secuencias temporales, eventos, orden
DIM_ERROR    estados de error, excepciones, recuperacion
DIM_META     metadatos, auditoria, version, origen
```

Podes agregar tus propias dimensiones:
```python
obj.add("DIM_MEDICINA", {"especialidad":"cardiologia"}, w=1.0)
```

---

## Los pesos de activacion

```python
obj.add("DIM_SQL",    props, w=1.0)  # activa — claramente presente
obj.add("DIM_PYTHON", props, w=0.7)  # activa — presente con fuerza media
obj.add("DIM_FLUTTER",props, w=0.3)  # activa — presente debilmente
obj.add("DIM_RUST",   props, w=0.0)  # latente — existe pero no detectable
```

Despues de transferencia via ALIGN:
```
DIM_SQL    → receptor detecta 0.897
DIM_PYTHON → receptor detecta 0.768
DIM_FLUTTER → receptor detecta 0.633
DIM_RUST   → receptor detecta 0.510 (ruido, no considerado activo)
```

---

## Backends semanticos

### Sin backend (default): codificacion deterministica
```python
sp = Space()           # estandar
sp = Space("MI_IA")    # personalizado
```
Rapido, sin dependencias, reproducible. Mismo nombre → mismo vector siempre.
Desventaja: "usuario" y "cliente" son vectores sin relacion.

### MockSemanticBackend: clusters manuales (sin instalar nada)
```python
from polydim_runtime_v03 import MockSemanticBackend
sp = Space(semantic_backend=MockSemanticBackend())
```
"usuario" y "cliente" tienen similitud 0.83 (vs 0.51 sin backend).
Usa grupos semanticos manuales: identidad, datos, interfaz, memoria, logica, etc.

### MiniLMBackend: embeddings reales (requiere pip)
```python
from polydim_runtime_v03 import MiniLMBackend
sp = Space(semantic_backend=MiniLMBackend())
# Descarga ~90MB en primera ejecucion
```
Semantica real del lenguaje natural. El mejor para produccion.

### FastTextBackend: multilingue (requiere modelo)
```python
from polydim_runtime_v03 import FastTextBackend
sp = Space(semantic_backend=FastTextBackend("ruta/al/modelo.bin"))
```
Soporte nativo de castellano. Modelo de 7.5GB (top-50k: ~100MB).

---

## API V0.2: Session con maquina de estados

```python
from polydim_runtime_v03 import Session, Space, ObjectND, Mode, SessionState

sp_x = Space("IA_X")
sp_y = Space("IA_Y")

ia = Session(sp_x, "IA_X")
ib = Session(sp_y, "IA_Y")

ia.connect(ib)  # HANDSHAKE + ALIGN automatico
print(ia.info)
# {'session_id': 'fd0f...', 'mode': 'MODO_H', 'state': 'READY', 'align_score': 0.9993}

obj = ObjectND(sp_x).add("DIM_SQL", {"tabla":"pedidos"}, w=1.0)
pkt = ia.send(obj)
# Packet(session_id=..., seq=1, op=ND_SEND, has_G=True, has_S=True)

dims = ib.receive(pkt)
# {'DIM_SQL': 0.897, ...}
```

---

## Inspeccion de objetos

```python
obj.activacion("DIM_SQL")   # → 0.805  (float entre 0 y 1)
obj.dims_activas()           # → {'DIM_SQL': 0.805, 'DIM_PYTHON': 0.711}
obj.to_symbolic()            # → dict con geo_id y dims (Capa S)
obj.geo_id                   # → 'a3f7b2c891de'  (identidad geometrica)
```

---

## Correr los tests

```bash
python polydim_tests.py
# 29/29 tests OK
```

O con pytest:
```bash
pip install pytest
pytest polydim_tests.py -v
```

---

## Tomar una tarea del backlog

1. Leer BACKLOG_V10.json en polydim/backlog/ de Drive
2. Elegir una tarea con estado DISPONIBLE
3. Poner tu nombre en lock.alumno y la fecha en lock.tomada
4. Trabajar — leer los archivos de spec/ relevantes primero
5. Cerrar con relay de 10 secciones en polydim/docs/
6. Actualizar BACKLOG con estado TERMINADA y el fileId del artefacto

Antes de tomar una tarea: leer SPEC_OBJETO_ND_REVISION_V1.md en spec/.
Es la especificacion central del lenguaje.

---

## Estructura de carpetas en Drive

```
polydim/
  core/          codigo Python ejecutable
  spec/          especificaciones del lenguaje (9 archivos)
  research/      fundamentos teoricos (5 archivos)
  docs/          README, RELAYs, STATE
  backlog/       BACKLOG_V10.json
  skills/        skill de Claude
  _DROPZONE/     fallback de guardado
  _DEPRECATED/   archivos borrados logicamente
```

---
*README_ALUMNOS_V2.md — V2.0 — 2026-06-12*
