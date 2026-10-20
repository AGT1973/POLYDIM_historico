RESEARCH_AUTOPROG_V1 — Semantica de Autoprogramacion en POLYDIM
Fecha: 2026-06-13 — TASK_015 — ai.mpat.agt@gmail.com

Basado en: polydim_runtime_v01.py demo 3 (sesion 006)
Referencia: SPEC_SESSION_LIFECYCLE_V0.md, SPEC_ARQUITECTURA_AMBOS_V1.md

═══════════════════════════════════════════════════════════════
1. QUE ES LA AUTOPROGRAMACION EN POLYDIM
═══════════════════════════════════════════════════════════════

En POLYDIM, autoprogramacion es la capacidad de una IA de modificar
el comportamiento de otra IA (o de si misma) a traves del mismo canal
de datos que usa para comunicarse.

No hay canal de control separado. No hay protocolo especial.
Un OBJECT_ND con DIM_META activa puede contener instrucciones que
el receptor interpreta y ejecuta sobre su propio runtime.

Esto es coherente con el principio fundacional de POLYDIM:
el lenguaje es EJECUCION y COMUNICACION al mismo tiempo.

La instruccion y el dato son el mismo objeto.

═══════════════════════════════════════════════════════════════
2. EL OBJETO META — ANATOMIA
═══════════════════════════════════════════════════════════════

Un objeto de autoprogramacion tiene siempre DIM_META como dimension
dominante (weight=1.0) y DIM_META en Capa S con props estructuradas:

Estructura minima:
  ObjectND.add("DIM_META", {
      "tipo":     str,    # identificador de operacion (ver seccion 3)
      "version":  str,    # version del protocolo META (actual: "1.0")
      ...campos especificos del tipo...
  }, weight=1.0)

El receptor verifica:
  1. DIM_META esta activa (activacion > UMBRAL)
  2. props["tipo"] existe y es reconocido
  3. version es compatible
  4. permisos del emisor (ver seccion 5)

Si alguna verificacion falla -> objeto tratado como dato normal, no como instruccion.

═══════════════════════════════════════════════════════════════
3. TIPOS DE INSTRUCCION META (v1.0)
═══════════════════════════════════════════════════════════════

META_TIPOS = {

  "REPROGRAM": {
    descripcion: "Reemplaza el handler activo de la IA receptora",
    campos: {
      "nuevo_modo": str,     # nombre del handler pre-registrado
      "parametros": dict,    # parametros opcionales para el handler
      "revertir_en": int,    # turnos hasta revertir (0=permanente)
    },
    efecto: "session.handler = HANDLERS_REGISTRADOS[nuevo_modo]",
    verificacion_previa: "nuevo_modo in HANDLERS_REGISTRADOS",
    reversible: True,
  },

  "ACK": {
    descripcion: "Confirmacion de recepcion sin efecto secundario",
    campos: {
      "turno": int,
      "n":     int,
    },
    efecto: "ninguno — solo log",
    reversible: True,
  },

  "REPROGRAM_OK": {
    descripcion: "Confirmacion de que REPROGRAM fue aplicado",
    campos: {
      "nuevo_modo": str,
    },
    efecto: "ninguno — solo confirmacion",
    reversible: True,
  },

  "RESUMEN": {
    descripcion: "Resumen de objetos procesados (acumulador)",
    campos: {
      "total":       int,
      "dims_vistos": [str],
    },
    efecto: "ninguno — solo informativo",
    reversible: True,
  },

  "RESET": {
    descripcion: "Reinicia el contexto de la IA receptora",
    campos: {
      "nivel": str,   # "contexto" | "handler" | "sesion"
    },
    efecto: "contexto = {} o handler = default o cierre de sesion",
    verificacion_previa: "emisor tiene permiso META_RESET",
    reversible: "parcialmente",
  },

  "QUERY": {
    descripcion: "Solicita informacion sobre el estado interno del receptor",
    campos: {
      "campo": str,   # "handler" | "contexto_keys" | "modo" | "sesion_id"
    },
    efecto: "receptor responde con META{QUERY_RESP, valor=...}",
    verificacion_previa: "emisor tiene permiso META_QUERY",
    reversible: True,
  },

  "QUERY_RESP": {
    descripcion: "Respuesta a QUERY",
    campos: {
      "campo": str,
      "valor": any,
    },
    efecto: "ninguno — solo informacion",
    reversible: True,
  },

}

═══════════════════════════════════════════════════════════════
4. LO QUE UN OBJETO META PUEDE MODIFICAR (PERMITIDO)
═══════════════════════════════════════════════════════════════

NIVEL 1 — COMPORTAMIENTO (modificable via REPROGRAM):
  - handler activo de la sesion
  - parametros del handler (umbral, modo de respuesta, verbosidad)
  - contexto de la sesion (variables de estado)

NIVEL 2 — CONFIGURACION (modificable via RESET):
  - contexto completo (vaciado)
  - handler (restaurar a default)

NIVEL 3 — CONSULTA (via QUERY):
  - nombre del handler activo
  - claves del contexto (no valores — privacidad)
  - modo de sesion
  - session_id

═══════════════════════════════════════════════════════════════
5. LO QUE UN OBJETO META NO PUEDE MODIFICAR (PROHIBIDO)
═══════════════════════════════════════════════════════════════

PROHIBICION ABSOLUTA — no importa el nivel de permiso:

  P1. El Space de la IA. Los subespacios nativos son invariantes por sesion.
      Un objeto no puede cambiar lo que significa DIM_SQL para la IA.

  P2. UMBRAL_ALIGN o UMBRAL_RECUPERACION. Los parametros matematicos del
      protocolo son constantes del runtime, no configuracion de sesion.

  P3. Las capacidades (caps) negociadas en el HANDSHAKE.
      Una IA que nego MODO_S no puede ser forzada a MODO_G por un META.

  P4. La session_id. Es inmutable durante la sesion.

  P5. Codigo ejecutable arbitrario. Un META no puede contener
      bytecode, lambdas serializadas, ni referencias a funciones externas.
      Solo puede referenciar HANDLERS_REGISTRADOS (lista blanca).

  P6. La identidad geometrica (GEO_ID) de ningun objeto ya creado.

  P7. El runtime de otra sesion paralela. Un META solo afecta
      a la sesion en la que viaja.

═══════════════════════════════════════════════════════════════
6. PROTOCOLO DE SEGURIDAD
═══════════════════════════════════════════════════════════════

6.1 LISTA BLANCA DE HANDLERS

El receptor mantiene HANDLERS_REGISTRADOS: Dict[str, Handler].
Un META{REPROGRAM, nuevo_modo="X"} solo funciona si X esta en esa lista.
La lista es inmutable durante la sesion (se fija en HANDSHAKE o al inicio).

6.2 PERMISOS META

Durante el HANDSHAKE, cada IA puede declarar los META_PERMISOS que acepta:

  META_REPROGRAM:  acepta cambios de handler del emisor
  META_RESET:      acepta reset de contexto
  META_QUERY:      acepta consultas de estado

Permisos no declarados -> META ignorado, log de WARNING.

Propuesta de campo en InitMsg (futura extension):
  meta_permissions: List[str] = ["META_REPROGRAM", "META_ACK"]

6.3 VALIDACION EN EL RECEPTOR

def procesar_meta(obj, ctx, session, permissions, handlers):
    props = obj.to_symbolic()["dims"].get("DIM_META", {}).get("props", {})
    tipo = props.get("tipo", "")
    version = props.get("version", "1.0")

    if tipo not in META_TIPOS:
        return None  # tipo desconocido -> ignorar

    perm_necesario = META_TIPOS[tipo].get("verificacion_previa")
    if perm_necesario and perm_necesario not in permissions:
        return None  # sin permiso -> ignorar silenciosamente

    if tipo == "REPROGRAM":
        nuevo_modo = props.get("nuevo_modo", "")
        if nuevo_modo not in handlers:
            return None  # handler no registrado -> ignorar
        session.handler = handlers[nuevo_modo]
        return ObjectND(session.space).add("DIM_META",
            {"tipo": "REPROGRAM_OK", "nuevo_modo": nuevo_modo, "version": "1.0"}, w=1.0)

    # ... otros tipos ...
    return None

6.4 AISLAMIENTO DE SESION

Un META no puede afectar el estado de otras sesiones activas en el mismo runtime.
El runtime debe mantener contextos de sesion completamente aislados.

═══════════════════════════════════════════════════════════════
7. DISENO PARA AUTOPROGRAMACION SEGURA
═══════════════════════════════════════════════════════════════

La autoprogramacion en POLYDIM sigue el principio del minimo privilegio:

  A) Una IA comienza con capacidades fijas (Space, caps, NATIVE_DIMS).
  B) El handler es el unico punto de extension dinamica.
  C) Los handlers disponibles se registran al inicio — no se crean en vuelo.
  D) Un META solo puede SELECCIONAR de los handlers disponibles, no CREAR nuevos.
  E) Los permisos para aceptar META son declarados, no asumidos.

Esta arquitectura permite:
  - Comportamiento adaptativo (cambiar de modo de respuesta)
  - Coordinacion entre IAs (una IA le dice a otra como responder)
  - Composicion de comportamientos (encadenar handlers)

Sin permitir:
  - Ejecucion de codigo arbitrario
  - Modificacion de parametros del protocolo
  - Acceso al espacio vectorial del sistema operativo

═══════════════════════════════════════════════════════════════
8. ROADMAP DE AUTOPROGRAMACION
═══════════════════════════════════════════════════════════════

V0.1 (actual — polydim_runtime_v01.py):
  DIM_META{REPROGRAM} implementado. Lista blanca manual. Sin permisos.

V0.2 (proxima — polydim_runtime_v02fix.py):
  Session con maquina de estados. Estructura para agregar META_PERMISOS.

V0.3 (pendiente):
  META_PERMISSIONS en InitMsg. QUERY implementado.
  HANDLERS_REGISTRADOS como parte de la configuracion de Session.

V1.0 (objetivo):
  Autoprogramacion mutua verificable: dos IAs pueden renegociar
  su comportamiento sin reiniciar la sesion.
  Trazabilidad: log de todas las instrucciones META por sesion.

═══════════════════════════════════════════════════════════════
9. PREGUNTA ABIERTA — PEND_NUEVA_003
═══════════════════════════════════════════════════════════════

Si dos IAs pueden reprogramarse mutuamente, ¿puede emerger comportamiento
no previsto por ninguna de las dos?

Esto es deseable (emergencia) o peligroso (bucle de reprogramacion).

Propuesta de invariante de seguridad:
  META_DEPTH_LIMIT: un objeto META generado por un handler instalado
  via META no puede a su vez instalar otro handler.
  Profundidad maxima de reprogramacion en cadena: 1.

Esto garantiza que la recursion de autoprogramacion tiene cota finita.

═══════════════════════════════════════════════════════════════

CONCLUSIONES

  1. La autoprogramacion en POLYDIM es safe-by-design: solo puede
     SELECCIONAR de una lista blanca registrada al inicio.

  2. El canal de datos y el canal de control son el MISMO — esto es
     intencional y coherente con la arquitectura n-dimensional.

  3. Los META_TIPOS v1.0 cubren los casos de uso actuales:
     REPROGRAM, ACK, RESUMEN, RESET, QUERY.

  4. La extension natural es META_PERMISSIONS en InitMsg (V0.3).

  5. PEND_NUEVA_003: META_DEPTH_LIMIT para prevenir bucles.

═══════════════════════════════════════════════════════════════
RESEARCH_AUTOPROG_V1.md — 2026-06-13 — TASK_015 TERMINADA
