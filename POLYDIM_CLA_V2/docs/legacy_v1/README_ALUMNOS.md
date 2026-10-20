# POLYDIM_DEST
# destino: polydim/docs/
# nombre:  README_ALUMNOS.md
# Version: V0.3 — 2026-06-13

# POLYDIM — Guia para Alumnos
# Version: V0.3 — 2026-06-13
# Modulo base: polydim_runtime_v03.py

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

Descarga polydim_runtime_v03.py de la carpeta polydim/core/ en Drive.
Es el unico archivo que necesitas. No tiene otras dependencias de POLYDIM.

---

## Uso en 3 lineas

    from polydim_runtime_v03 import Space, ObjectND, polydim_connect

    obj  = ObjectND(Space("MI_IA")).add("DIM_SQL", {"tabla":"pedidos"}, w=1.0)
    dims = polydim_connect(Space("MI_IA"), Space("OTRA_IA")).transfer(obj)

dims contiene las dimensiones que la otra IA detecto en el objeto.

---

## Los conceptos que necesitas entender

### Space
El espacio vectorial donde viven los objetos.
Cada IA tiene su propio Space. Dos IAs con el mismo personal_seed comparten simbolos.

    sp = Space()           # espacio estandar (sin personal_seed)
    sp = Space("MI_IA")    # espacio propio de tu IA

### ObjectND
El objeto polidimensional. Se crea en un Space y puede tener N dimensiones.

    obj = ObjectND(sp)
    obj.add("DIM_SQL",    {"tabla":"usuarios", "pk":"id"},  w=1.0)
    obj.add("DIM_PYTHON", {"tipo":"dict"},                  w=0.7)
    obj.add("DIM_FLUTTER",{"widget":"Form"},                w=0.3)
    obj.add("DIM_RUST",   {"tipo":"struct"},                w=0.0)  # latente

El parametro w es el peso de activacion [0.0, 1.0]:
  w=1.0  dimension completamente activa
  w=0.5  dimension parcialmente activa
  w=0.0  dimension latente (existe pero no se detecta)

Encadenamiento: obj.add() devuelve self, se puede encadenar:
    obj = ObjectND(sp).add("DIM_SQL", {"tabla":"x"}, w=1.0).add("DIM_PYTHON", {}, w=0.7)

### Props numericas (nuevo en V0.3)
Los valores int y float en props usan codificacion sinusoidal automaticamente.
Esto significa que valores cercanos producen vectores cercanos:

    sim({"n": 42}, {"n": 43}) ≈ 0.999   <- muy similares
    sim({"n": 42}, {"n": 100}) ≈ 0.67   <- moderadamente similares
    sim({"n": 42}, {"n": 542}) ≈ 0.50   <- esencialmente distintos

    obj.add("DIM_SQL", {"n_filas": 1000, "pk":"id"}, w=1.0)  # n_filas es numerico

### polydim_connect
Conecta dos Spaces. Ejecuta HANDSHAKE + ALIGN automaticamente.

    conn = polydim_connect(Space("IA_A"), Space("IA_B"))
    print(conn.info)
    # {'session_id': 'fd0fd8...', 'mode': 'MODO_H', 'align_score': 0.9993,
    #  'meta_perms': ['META_ACK']}

    dims = conn.transfer(obj)
    # {'DIM_SQL': 0.908, 'DIM_PYTHON': 0.781, ...}

### Session (uso avanzado)
Para control fino con META_PERMISSIONS y handlers:

    from polydim_runtime_v03 import Session, META_ACK, META_REPROGRAM

    ia = Session(Space("IA_A"), "IA_A",
                 meta_permissions=[META_ACK, META_REPROGRAM],
                 handlers={"mi_modo": mi_handler_func})
    ib = Session(Space("IA_B"), "IA_B",
                 meta_permissions=[META_ACK, META_REPROGRAM])
    ia.connect(ib)
    pkt = ia.send(obj)
    dims = ib.receive(pkt)

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
    DIM_META     metadatos, autoprogramacion, instrucciones

Podes agregar tus propias dimensiones:
    obj.add("DIM_MEDICINA", {"especialidad":"cardiologia"}, w=1.0)

---

## Como ver que dimensiones tiene un objeto

    print(obj.dims_activas())
    # {'DIM_SQL': 0.805, 'DIM_PYTHON': 0.709, 'DIM_FLUTTER': 0.595}

    print(obj.activacion("DIM_SQL"))
    # 0.805

    print(obj.geo_id)
    # 'a3f7b2c891de'  <- identidad geometrica invariante

---

## El ALIGN: por que importa

Cuando dos IAs tienen espacios distintos (distintos personal_seed),
el mismo concepto vive en vectores distintos. Sin ALIGN, la transferencia es ruido.
Con ALIGN, la precision es ~0.999.

polydim_connect() hace el ALIGN automaticamente.
El score en conn.info te dice cuan bien alineados estan los espacios.
Score > 0.85 es valido para MODO_H. Score < 0.85 -> el sistema usa MODO_S (simbolico).

---

## Autoprogramacion con DIM_META (avanzado, V0.3)

Una IA puede enviar instrucciones a otra via un objeto con DIM_META.
Solo funciona si ambas IAs lo acordaron en el HANDSHAKE (META_PERMISSIONS).

    # IA_A quiere cambiar el handler de IA_B
    inst = ObjectND(sp_a).add("DIM_META",
                               {"tipo": "REPROGRAM", "nuevo_modo": "acumulador"},
                               w=1.0)
    ib.receive(ia.send(inst))
    # Si ib tiene META_REPROGRAM en sus permisos Y "acumulador" en sus handlers,
    # el handler de ib cambia automaticamente.

    # META_DEPTH_LIMIT=1: previene bucles de autoprogramacion en cadena.

---

## Tabla de versiones

    Version  Modulo                  Novedad principal
    V0.1     polydim_runtime_v01.py  API 3 lineas, align_transform basico
    V0.2     polydim_runtime_v02.py  enums, dataclasses, Session con estado
    V0.3     polydim_runtime_v03.py  META_PERMISSIONS, B2 numerico RFF  <- USAR ESTE

---

## Ejemplo completo: dos IAs intercambian un objeto con datos numericos

    from polydim_runtime_v03 import Space, ObjectND, polydim_connect

    # IA disenio crea el objeto
    sp = Space("IA_DISENIO")
    campo = ObjectND(sp)
    campo.add("DIM_SQL",    {"columna":"precio", "n_filas": 5000}, w=1.0)
    campo.add("DIM_PYTHON", {"tipo":"float",     "rango_max": 9999}, w=1.0)
    campo.add("DIM_FLUTTER",{"widget":"TextField","validacion":"positivo"}, w=0.8)
    campo.add("DIM_RUST",   {"tipo":"f64",        "overflow":"check"}, w=0.5)

    print("Dims en IA_DISENIO:", campo.dims_activas())

    # IA backend recibe el objeto
    conn = polydim_connect(sp, Space("IA_BACKEND"))
    dims = conn.transfer(campo)
    print("Dims detectadas por IA_BACKEND:", dims)
    print("Session:", conn.info)

    # Resultado esperado:
    # Dims en IA_DISENIO: {'DIM_SQL': ~0.9, 'DIM_PYTHON': ~0.85, 'DIM_FLUTTER': ~0.76, 'DIM_RUST': ~0.63}
    # Dims detectadas por IA_BACKEND: {'DIM_SQL': ~0.9, 'DIM_PYTHON': ~0.85, ...}
    # Session: {'mode': 'MODO_H', 'align_score': 0.9993, 'meta_perms': ['META_ACK']}

---

## Primer ejercicio guiado

Objetivo: crear un OBJECT_ND que represente un "usuario" en tres dimensiones
y verificar que las tres se detectan correctamente.

    from polydim_runtime_v03 import Space, ObjectND, polydim_connect

    sp = Space("MI_IA")

    # Paso 1: crear el objeto
    usuario = ObjectND(sp)
    usuario.add("DIM_SQL",    {"tabla":"usuarios", "pk":"email"},  w=1.0)
    usuario.add("DIM_PYTHON", {"clase":"User",     "metodos":3},   w=0.8)
    usuario.add("DIM_FLUTTER",{"widget":"UserCard","estado":"activo"}, w=0.6)

    # Paso 2: ver sus dimensiones
    print("Dimensiones:", usuario.dims_activas())

    # Paso 3: transferir a otra IA
    conn = polydim_connect(sp, Space("IA_DESTINO"))
    dims = conn.transfer(usuario)
    print("Recibidas:", dims)

    # Paso 4: verificar
    assert "DIM_SQL"    in dims, "DIM_SQL deberia estar activa"
    assert "DIM_PYTHON" in dims, "DIM_PYTHON deberia estar activa"
    assert "DIM_RUST"   not in dims, "DIM_RUST no fue declarada"
    print("Todo OK")

Cuando el ejercicio funciona, podes continuar con las tareas del BACKLOG.

---

## Tareas disponibles en el BACKLOG

Leer BACKLOG_V12.json en polydim/backlog/ de Drive.
Tomar una tarea DISPONIBLE, hacer lock con tu nombre, trabajar, cerrar con relay.

Antes de tomar una tarea: leer los archivos de spec relevantes en polydim/spec/.
El archivo central es SPEC_OBJETO_ND_REVISION_V1.md.
El modulo base para todo el codigo es polydim_runtime_v03.py.

---

## Estructura de carpetas en Drive

    polydim/
      core/          codigo Python ejecutable (modulo base: v03)
      spec/          especificaciones del lenguaje
      research/      investigacion de fundamentos
      docs/          documentacion (este archivo, RELAYs, STATE)
      backlog/       gestion de tareas
      skills/        skills de Claude
      _DROPZONE/     fallback de guardado
      _DEPRECATED/   archivos borrados logicamente

---
*README_ALUMNOS.md — V0.3 — 2026-06-13*
