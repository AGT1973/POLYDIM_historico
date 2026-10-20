# POLYDIM_DEST
# destino: polydim/docs/
# nombre:  README_ALUMNOS.md

# POLYDIM — Guia para Alumnos
# Version: V0.1 — 2026-06-11

---

## Que es POLYDIM

Un lenguaje de programacion diseñado para IAs, no para humanos.

La idea central: en los lenguajes actuales un objeto es UNA cosa
(una fila SQL, O un dict Python, O un widget Flutter).
En POLYDIM, el mismo objeto ES todas esas cosas SIMULTANEAMENTE.
Cada "dimension" es una perspectiva distinta del mismo objeto.

---

## Instalacion

Solo necesitas numpy:

    pip install numpy

Despues descarga polydim_runtime_v01.py de la carpeta polydim/core/ en Drive.

---

## Uso en 3 lineas

    from polydim_runtime_v01 import Space, ObjectND, polydim_connect

    obj = ObjectND(Space("MI_IA")).add("DIM_SQL", {"tabla":"pedidos"}, w=1.0)
    dims = polydim_connect(Space("MI_IA"), Space("OTRA_IA")).transfer(obj)

Eso es todo. dims contiene las dimensiones que la otra IA detecto en el objeto.

---

## Los conceptos que necesitas entender

### Space
El espacio vectorial donde viven los objetos.
Cada IA tiene su propio Space. Dos IAs con el mismo personal_seed comparten simbolos.

    sp = Space()           # espacio estandar
    sp = Space("MI_IA")    # espacio propio de tu IA

### ObjectND
El objeto polidimensional. Se crea en un Space y puede tener N dimensiones.

    obj = ObjectND(sp)
    obj.add("DIM_SQL",    {"tabla":"usuarios", "pk":"id"},  w=1.0)
    obj.add("DIM_PYTHON", {"tipo":"dict"},                  w=0.7)
    obj.add("DIM_FLUTTER",{"widget":"Form"},                w=0.3)
    obj.add("DIM_RUST",   {"tipo":"struct"},                w=0.0)  # latente

El parametro w es el peso de activacion [0.0, 1.0].
  w=1.0 → dimension completamente activa
  w=0.5 → dimension parcialmente activa
  w=0.0 → dimension latente (existe pero no se detecta)

### polydim_connect
Conecta dos Spaces. Ejecuta HANDSHAKE + ALIGN automaticamente.

    conn = polydim_connect(Space("IA_A"), Space("IA_B"))
    print(conn.info)
    # {'session_id': 'fd0fd8...', 'mode': 'MODO_H', 'align_score': 0.9993, 'align_valid': True}

    dims = conn.transfer(obj)
    # {'DIM_SQL': 0.897, 'DIM_PYTHON': 0.766, 'DIM_FLUTTER': 0.633}

---

## Las 9 dimensiones nativas

    DIM_PYTHON   logica dinamica, analisis, ML
    DIM_RUST     seguridad de memoria, performance, ownership
    DIM_FLUTTER  UI reactiva, widgets, estado
    DIM_SQL      datos relacionales, tablas, constraints
    DIM_GRAPH    grafos, nodos, aristas, relaciones
    DIM_VECTOR   embeddings, similitud semantica
    DIM_TIME     secuencias temporales, eventos
    DIM_ERROR    estados de error, excepciones
    DIM_META     metadatos, auditoria, version

Podés agregar tus propias dimensiones:

    obj.add("DIM_MEDICINA", {"especialidad":"cardiologia"}, w=1.0)

---

## Como ver que dimensiones tiene un objeto

    print(obj.dims_activas())
    # {'DIM_SQL': 0.805, 'DIM_PYTHON': 0.709, 'DIM_FLUTTER': 0.595}

    print(obj.activacion("DIM_SQL"))
    # 0.805

    print(obj.geo_id)
    # 'a3f7b2c891de'  ← identidad geometrica invariante

---

## El ALIGN: por que importa

Cuando dos IAs tienen espacios distintos, el mismo concepto
vive en vectores distintos. Sin ALIGN, la transferencia es ruido.
Con ALIGN, la precision es ~0.999.

polydim_connect() hace el ALIGN automaticamente.
El score en conn.info te dice cuan bien alineados estan los espacios.
Score > 0.85 es valido. Score < 0.85 → el sistema degrada a MODO_S (simbolico).

---

## Ejemplo completo: dos IAs intercambian un campo de formulario

    from polydim_runtime_v01 import Space, ObjectND, polydim_connect

    # IA diseñadora crea el objeto
    sp_disenio = Space("IA_DISENIO")
    campo_precio = ObjectND(sp_disenio)
    campo_precio.add("DIM_SQL",    {"columna":"precio", "tipo":"DECIMAL_10_2"}, w=1.0)
    campo_precio.add("DIM_PYTHON", {"tipo":"float",     "rango":"0_999999"},    w=1.0)
    campo_precio.add("DIM_FLUTTER",{"widget":"TextField","validacion":"positivo"}, w=1.0)
    campo_precio.add("DIM_RUST",   {"tipo":"f64",        "overflow":"check"},   w=1.0)

    print("Dims en IA_DISENIO:", campo_precio.dims_activas())

    # IA backend recibe el objeto
    conn = polydim_connect(sp_disenio, Space("IA_BACKEND"))
    dims_recibidas = conn.transfer(campo_precio)
    print("Dims detectadas por IA_BACKEND:", dims_recibidas)
    print("Sesion:", conn.info)

---

## Tareas disponibles en el BACKLOG

Leer BACKLOG_V6.json en polydim/backlog/ de Drive.
Tomar una tarea DISPONIBLE, hacer lock con tu nombre, trabajar, cerrar con relay.

Antes de tomar una tarea: leer los archivos de spec relevantes en polydim/spec/.
El archivo central es SPEC_OBJETO_ND_REVISION_V1.md.

---

## Estructura de carpetas en Drive

    polydim/
      core/          codigo Python ejecutable
      spec/          especificaciones del lenguaje
      research/      investigacion de fundamentos
      docs/          documentacion (este archivo, RELAYs, STATE)
      backlog/       gestion de tareas
      skills/        skills de Claude
      _DROPZONE/     fallback de guardado
      _DEPRECATED/   archivos borrados logicamente

---
*README_ALUMNOS.md — V0.1 — 2026-06-11*
