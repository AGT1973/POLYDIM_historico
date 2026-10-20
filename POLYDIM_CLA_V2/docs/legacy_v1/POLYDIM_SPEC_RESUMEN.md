POLYDIM — Especificacion Resumida
Version: V0.3 — 2026-06-13
Lectura: 10 minutos. Modulo base: polydim_runtime_v03.py

═══════════════════════════════════════════════════════════════
1. PRINCIPIO FUNDACIONAL
═══════════════════════════════════════════════════════════════

En lenguajes convencionales, un objeto ES una cosa.
En POLYDIM, un objeto ES N cosas al mismo tiempo — una por dimension.

Ejemplo: el campo "precio" es SIMULTANEAMENTE
  una columna SQL (DIM_SQL)
  un float Python (DIM_PYTHON)
  un TextField de UI (DIM_FLUTTER)
  un f64 de Rust (DIM_RUST)

No hay conversion. Hay activacion de dimensiones.
El lenguaje es para IAs, no para humanos. Una IA opera en N dimensiones naturalmente.

═══════════════════════════════════════════════════════════════
2. OBJECT_ND — unidad minima
═══════════════════════════════════════════════════════════════

Anatomia interna (hipervector en R^N=10000):

  hv = SUPERPOSE(
      GEO_ID * 1.0,                          <- identidad invariante
      sub(dim_i) * w_i,                      <- MEMBRESIA: deteccion
      BIND(sub(dim_i), enc(props_i)) * w_i * 0.3  <- CONTENIDO: recuperacion
  )

API:
  obj = ObjectND(space)
  obj.add("DIM_SQL", {"tabla": "usuarios", "n": 5000}, w=1.0)
  obj.add("DIM_PYTHON", {"tipo": "dict"}, w=0.7)
  obj.add("DIM_RUST", {}, w=0.0)   <- latente

  obj.dims_activas()   -> {'DIM_SQL': 0.905, 'DIM_PYTHON': 0.781}
  obj.activacion("DIM_SQL")  -> 0.905
  obj.geo_id           -> 'abc123def456'  (invariante)

Parametros:
  N = 10000 dimensiones del hipervector
  UMBRAL = 0.510  (activacion minima para detectar una dimension)
  CONTENT_W = 0.3  (peso de la capa de contenido)

Props numericas (B2/RFF): valores int/float se codifican con kernel gaussiano.
  sim(n=42, n=43) ≈ 0.999   sim(n=42, n=100) ≈ 0.67   sim(n=42, n=542) ≈ 0.50

═══════════════════════════════════════════════════════════════
3. SPACE — espacio vectorial de la IA
═══════════════════════════════════════════════════════════════

  sp = Space()            # estandar: todos los Space() son identicos
  sp = Space("MI_IA")     # personal_seed: vectores propios de esta IA

Space determina como se codifican los simbolos.
Dos IAs con el mismo personal_seed comparten el espacio -> ALIGN trivialmente OK.
Dos IAs con distinto personal_seed -> ALIGN via align_transform = B.T@(A@hv).

IMPORTANTE: obj y Session deben usar el MISMO Space.

═══════════════════════════════════════════════════════════════
4. SESSION — ciclo de vida
═══════════════════════════════════════════════════════════════

Estados:
  IDLE -> CONNECTING -> NEGOTIATING -> [ALIGNING] -> READY
                                    -> DEGRADED (si ALIGN falla)

Flujo:
  ia = Session(space_a, "IA_A")
  ib = Session(space_b, "IA_B")
  ia.connect(ib)          <- HANDSHAKE + ALIGN automatico

  pkt = ia.send(obj)      <- serializa en modo acordado
  dims = ib.receive(pkt)  <- detecta dimensiones en el receptor

API 3 lineas (polydim_connect):
  conn = polydim_connect(Space("IA_A"), Space("IA_B"))
  dims = conn.transfer(obj)
  # {'DIM_SQL': 0.909, 'DIM_PYTHON': 0.781, ...}

═══════════════════════════════════════════════════════════════
5. HANDSHAKE — negociacion
═══════════════════════════════════════════════════════════════

INIT_MSG (IA_A -> IA_B):
  version, sender_id, capabilities, preferred_mode, N, seed, nonce, meta_permissions

Negociacion:
  modo = MODO_H si ambas tienen CAP_G, sino MODO_S
  meta_perms = interseccion(perms_A, perms_B)
  session_id = MD5(nonce_A + nonce_B)

ACCEPT_MSG -> ACK_MSG -> sesion lista

═══════════════════════════════════════════════════════════════
6. ALIGN — compatibilidad de espacios
═══════════════════════════════════════════════════════════════

ALIGN no corrige espacios incompatibles. ALIGN detecta si son compatibles.

Algoritmo (align_transform = B.T@(A@hv)):
  A = sondas del Space de IA_A  (K x N, K=28 minimo)
  B = sondas del Space de IA_B  (K x N)
  hv_alineado = B.T @ (A @ hv)

Score = media de sim(align_transform(sub_A(d)), sub_B(d)) para d en NATIVE_DIMS

Interpretacion:
  score = 1.00        mismo Space (sin personal_seed)  -> MODO_H
  score >= 0.85       compatible  -> MODO_H
  score < 0.85        incompatible -> MODO_S automaticamente
  score ≈ 0.50        sin relacion lineal real

UMBRAL_ALIGN = 0.85

═══════════════════════════════════════════════════════════════
7. MODOS DE SESION
═══════════════════════════════════════════════════════════════

MODO_S (simbolico):
  payload = {geo_id, dims: {name: {w, props}}}
  Universal. Funciona entre CUALQUIER par de IAs.
  No requiere ALIGN. Lectura humana posible.

MODO_G (geometrico):
  payload = hipervector R^N alineado
  Solo para IAs con espacios compatibles (score >= 0.85).
  Alta velocidad, sin paso simbolico.

MODO_H (hibrido, default):
  payload = MODO_S (control) + MODO_G (datos)
  Default cuando ambas IAs tienen CAP_G.
  Mejor de los dos mundos.

payload_S CANONICO — contrato de interoperabilidad:
  {
    "geo_id": "abc123def456",
    "dims": {
      "DIM_SQL":    {"w": 1.0, "props": {"tabla": "usuarios"}},
      "DIM_PYTHON": {"w": 0.7, "props": {"tipo": "dict"}}
    }
  }
  Cualquier version puede leer payload_S de cualquier otra version.

═══════════════════════════════════════════════════════════════
8. DIM_META — autoprogramacion
═══════════════════════════════════════════════════════════════

Un OBJECT_ND con DIM_META activa puede contener instrucciones.
Solo funciona si el permiso esta en meta_permissions negociados.

Tipos disponibles (META_TIPOS v1.0):
  ACK         confirmacion sin efecto (siempre permitido)
  REPROGRAM   cambia el handler activo de la IA receptora
  RESET       reinicia contexto o handler
  QUERY       solicita informacion de estado interno

Uso:
  ia = Session(sp, meta_permissions=[META_ACK, META_REPROGRAM],
               handlers={"modo_A": handler_a, "modo_B": handler_b})
  obj = ObjectND(sp).add("DIM_META",
                          {"tipo": "REPROGRAM", "nuevo_modo": "modo_B"},
                          w=1.0)

META_DEPTH_LIMIT = 1: previene bucles de autoprogramacion en cadena.
Un handler instalado via META no puede instalar otro META.

═══════════════════════════════════════════════════════════════
9. DIMENSIONES NATIVAS
═══════════════════════════════════════════════════════════════

DIM_PYTHON   logica dinamica, analisis, ML
DIM_RUST     seguridad, performance, ownership
DIM_FLUTTER  UI reactiva, widgets, estado
DIM_SQL      datos relacionales, tablas, queries
DIM_GRAPH    grafos, nodos, aristas
DIM_VECTOR   embeddings, similitud semantica
DIM_TIME     secuencias temporales, eventos
DIM_ERROR    estados de error, excepciones
DIM_META     metadatos, instrucciones, autoprogramacion

Se pueden agregar dimensiones custom: obj.add("DIM_MEDICINA", {...}, w=1.0)

═══════════════════════════════════════════════════════════════
10. PARAMETROS CLAVE
═══════════════════════════════════════════════════════════════

N              = 10000       dimensiones del hipervector
POLYDIM_SEED   = "POLYDIM_V1_SEED_2026"
UMBRAL         = 0.510       activacion minima (0.5 + 2/sqrt(N))
UMBRAL_ALIGN   = 0.850       score minimo para MODO_H
CONTENT_W      = 0.300       peso de capa de contenido
K_SONDAS_MIN   = 28          sondas minimas para ALIGN
META_DEPTH_MAX = 1           profundidad maxima de autoprogramacion
SIGMA_B2       = 0.0253      sigma para RFF numerico

═══════════════════════════════════════════════════════════════
11. LECTURA RECOMENDADA (orden para alumno nuevo)
═══════════════════════════════════════════════════════════════

1. Este archivo (10 min)
2. README_ALUMNOS.md — guia practica con ejercicio (15 min)
3. polydim_runtime_v03.py — modulo base, leer docstrings (20 min)
4. SPEC_OBJETO_ND_REVISION_V1.md — spec completa del objeto
5. SPEC_ALIGN_V1.md — matematica del protocolo de alineacion
6. BACKLOG_V14.json — tomar una tarea disponible

═══════════════════════════════════════════════════════════════
POLYDIM_SPEC_RESUMEN.md — V0.3 — 2026-06-13
