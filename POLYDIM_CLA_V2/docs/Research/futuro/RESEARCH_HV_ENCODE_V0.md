RESEARCH_HV_ENCODE_V0 — Codificacion Capa S a Capa G en POLYDIM
Fecha: 2026-06-13 — TASK_007 — ai.mpat.agt@gmail.com

Pregunta: dado un OBJECT_ND con dimensiones y propiedades simbolicas (Capa S),
¿como se genera el hipervector (Capa G) que lo representa?

═══════════════════════════════════════════════════════════════
1. EL PROBLEMA EN DOS PARTES
═══════════════════════════════════════════════════════════════

HV_ENCODE tiene dos componentes independientes:

  PARTE A — Codificacion de NOMBRES DE DIMENSION
    Entrada:  "DIM_SQL", "DIM_PYTHON", "mi_dimension_custom"
    Salida:   vector unitario en R^N (subespacio de la dimension)

  PARTE B — Codificacion de PROPIEDADES
    Entrada:  {"tabla": "usuarios", "pk": "id", "n": 3}
    Salida:   vector unitario en R^N (contenido de la dimension)

El hipervector final del objeto es:
  hv = SUPERPOSE(
      GEO_ID,
      sub_i * w_i,                       <- Parte A: subespacios
      BIND(sub_i, enc(props_i)) * w_i * 0.3  <- Partes A+B: contenido
  )

═══════════════════════════════════════════════════════════════
2. PARTE A — CODIFICACION DE NOMBRES DE DIMENSION
═══════════════════════════════════════════════════════════════

OPCION A1: VSA DETERMINISTICA (implementada en V0.1)
─────────────────────────────────────────────────────

Algoritmo actual (Space._mk en polydim_runtime_v02fix.py):
  seed = MD5(personal_seed + ":" + nombre) mod 2^32
  rng  = default_rng(seed)
  hv   = rng.standard_normal(N)
  hv   = hv / ||hv||

Propiedades matematicas:
  - Reproducible: misma semilla, mismo vector (siempre)
  - Ortogonalidad esperada: E[dot(hv_a, hv_b)] = 0 para nombres distintos
  - Error de ortogonalidad: std = 1/sqrt(N) = 0.01 para N=10000
  - Sin proximidad semantica: "SQL" y "base_de_datos" son ortogonales

Cuando usar: POLYDIM V0.1, sistemas deterministicos, alumnos.
Limitacion: dos IAs con personal_seed distinto asignan vectores distintos
            al mismo nombre -> necesita ALIGN para comunicarse.

OPCION A2: HASH PROYECTADO CON CLUSTERING SEMANTICO
─────────────────────────────────────────────────────

Extiende A1 con grupos semanticos predefinidos:
  CLUSTERS = {
      "SQL":       ["DIM_SQL", "tabla", "query", "schema", "registro"],
      "PYTHON":    ["DIM_PYTHON", "lista", "dict", "objeto", "modulo"],
      "TIEMPO":    ["DIM_TIME", "fecha", "timestamp", "duracion"],
      ...
  }

  hv_dim = alpha * hv_determinista(nombre)
          + (1-alpha) * hv_cluster_centro

  alpha en [0.7, 0.9] -> 70-90% identidad, 10-30% afinidad semantica

Ventaja: dimensiones relacionadas tienen vectores cercanos (~0.1-0.3 de sim).
Costo: requiere definir CLUSTERS manualmente -> no escala.
Cuando usar: POLYDIM V0.2 para NATIVE_DIMS extendidos.

OPCION A3: EMBEDDING MODEL
────────────────────────────

hv_dim = normalize(embedding_model.encode(nombre))[:N]

donde embedding_model es un modelo de lenguaje (SBERT, OpenAI, etc.)

Ventajas:
  - "DIM_SQL" y "base_de_datos" tendran vectores proximos
  - No requiere ALIGN si ambas IAs usan el MISMO modelo base
  - Generaliza a nombres nunca vistos

Desventajas:
  - Requiere dependencia externa (modelo >= 384 dimensiones)
  - Distintos modelos -> espacios incompatibles -> necesita ALIGN
  - No determinista entre versiones del modelo

Cuando usar: POLYDIM V1.0, IAs con LLMs reales.
Para V0.1: no recomendado (dependencia innecesaria).

CONCLUSION PARTE A:
  V0.1:  A1 (VSA deterministica con personal_seed)
  V0.2:  A2 (+ clustering semantico para NATIVE_DIMS)
  V1.0:  A3 (embedding model compartido como base, personal_seed como ajuste)

═══════════════════════════════════════════════════════════════
3. PARTE B — CODIFICACION DE PROPIEDADES
═══════════════════════════════════════════════════════════════

Las propiedades son dict con tipos variados:
  - strings:  {"tabla": "usuarios"}
  - enteros:  {"n": 42}
  - booleanos: {"activo": True}
  - floats:   {"peso": 0.7}
  - anidados: {"config": {"timeout": 30}}  <- no soportado actualmente

OPCION B1: CAST A STRING + VSA (implementada en V0.1)
──────────────────────────────────────────────────────

def _enc(self, props):
    return SUPERPOSE(*[
        BIND(hv(str(k)), hv(str(v)))
        for k, v in props.items()
    ])

Transforma k y v a string, luego a vector via _mk.
Ejemplo: {"n": 42} -> BIND(hv("n"), hv("42"))

Ventaja: simple, universal, sin dependencias.
Limitacion: "42" y "43" son ortogonales (no hay proximidad numerica).
            "True" y "true" pueden ser distintos si case-sensitive.

OPCION B2: CODIFICACION NUMERICA EXPLICITA
───────────────────────────────────────────

Para valores numericos, usar codificacion posicional en el hipervector:

  hv_num(x) = cos(x * freq_i) para i=1..N/2 concatenado con
              sin(x * freq_i) para i=1..N/2

donde freq_i = 2*pi*i/periodo_max

Esta codificacion (similar a sinusoidal positional encoding de Transformers):
  - Valores proximos -> vectores proximos
  - Periodica -> bounded
  - Invertible aproximadamente

  hv_num(42) · hv_num(43) > hv_num(42) · hv_num(100)

Costo: implementacion separada para detectar tipo numerico.
Cuando usar: props con rangos de valores numericos significativos.

OPCION B3: TIPO EXPLICITO EN EL BIND
──────────────────────────────────────

Extender el esquema de props con tipo explicito:

  props = {
      "tabla":  {"tipo": "str",  "valor": "usuarios"},
      "n":      {"tipo": "int",  "valor": 42},
      "activo": {"tipo": "bool", "valor": True},
  }

  hv_prop = BIND(hv("tabla"), BIND(hv_tipo("str"), hv("usuarios")))

Ventaja: el receptor puede decodificar el tipo exacto.
Costo: props mas verbosas. Compatible con Capa S actual con extension.

OPCION B4: PROPS ANIDADAS RECURSIVAS
──────────────────────────────────────

Para {"config": {"timeout": 30, "retry": 3}}:

  hv_nested = hv("config") * SUPERPOSE(
      BIND(hv("timeout"), hv_num(30)),
      BIND(hv("retry"),   hv_num(3))
  )

Recursion de BIND permite estructuras jerarquicas arbitrarias.
Costo computacional: O(profundidad * |props|).

CONCLUSION PARTE B:
  V0.1:  B1 (cast a string, simple y funcional)
  V0.2:  B1 + B2 para valores numericos
  V1.0:  B3 + B4 para props ricas y anidadas

═══════════════════════════════════════════════════════════════
4. EXPERIMENTO: PROXIMIDAD EN B1 VS B2
═══════════════════════════════════════════════════════════════

Con B1 (cast a string):
  hv("42") · hv("43") = ~0.0  (ortogonales, no hay relacion)
  hv("usuario") · hv("user") = ~0.0  (distintos strings)

Con B2 (codificacion sinusoidal):
  hv_num(42) · hv_num(43) = cos(2*pi*1/1000) ≈ 0.999  (muy proximos)
  hv_num(42) · hv_num(100) = cos(2*pi*58/1000) ≈ 0.832  (menos proximos)
  hv_num(42) · hv_num(542) = cos(2*pi*500/1000) ≈ 0.0  (lejanos)

Implicacion: para props con rangos numericos continuos (edades, timestamps,
precios, coordenadas), B2 permite deteccion semantica real.
Un objeto con {"edad": 25} estara mas cerca de {"edad": 26} que de {"edad": 80}.

═══════════════════════════════════════════════════════════════
5. TABLA DE DECISION PARA ALUMNOS
═══════════════════════════════════════════════════════════════

Caso de uso                    | Parte A | Parte B | Razon
───────────────────────────────┼─────────┼─────────┼──────────────────────────
Demo alumno basico             |   A1    |   B1    | Simple, sin dependencias
IAs con personal_seed          |   A1    |   B1    | align_transform cubre el gap
Dimensiones con semantica      |   A2    |   B1    | CLUSTERS mejora proximidad
Props con numeros continuos    |   A1    |   B2    | Proximidad numerica util
Sistema de produccion V1.0     |   A3    |   B3+B4 | Semantica real, tipos ricos

═══════════════════════════════════════════════════════════════
6. RECOMENDACIONES PARA V0.1 (ESTADO ACTUAL)
═══════════════════════════════════════════════════════════════

El sistema actual (A1 + B1) es correcto para:
  - Todos los demos de alumnos
  - Comunicacion entre IAs con el mismo personal_seed o sin el
  - Desarrollo e investigacion del protocolo

No es correcto para:
  - Razonamiento sobre valores numericos continuos
  - Proximidad semantica entre nombres de dimension
  - Props anidadas complejas

La extension a B2 (numerica) tiene bajo costo y alto impacto:
  ~20 lineas de codigo adicionales en Space._enc()
  Activa automaticamente cuando el valor es int o float.

IMPLEMENTACION SUGERIDA B2 (pendiente de TASK_018):

def _enc_valor(self, v):
    if isinstance(v, (int, float)):
        return self._enc_numerico(float(v))
    return self.sym(str(v))

def _enc_numerico(self, x, periodo=1000.0):
    freqs = np.arange(1, N//2 + 1) * (2 * np.pi / periodo)
    hv = np.concatenate([np.cos(x * freqs), np.sin(x * freqs)]).astype(np.float32)
    return hv / np.linalg.norm(hv)

═══════════════════════════════════════════════════════════════
7. PENDIENTES ABIERTOS
═══════════════════════════════════════════════════════════════

PEND_007 (parcialmente resuelta):
  V0.1 usa A1+B1. Correcto para casos actuales.
  Extension a B2 recomendada para V0.2.

PEND_NUEVA_004: embedding model compartido (A3)
  Para POLYDIM V1.0 con IAs reales.
  Requiere definir que modelo es el "estandar POLYDIM".
  Candidatos: sentence-transformers/all-MiniLM-L6-v2 (384 dim),
              paraphrase-multilingual-mpnet-base-v2 (768 dim)

PEND_NUEVA_005: props anidadas (B4)
  Necesario para objetos complejos (configuraciones, esquemas).
  Bajo riesgo — BIND es recursivamente aplicable.

═══════════════════════════════════════════════════════════════

CONCLUSIONES

  1. HV_ENCODE tiene dos partes independientes (nombres A, props B).
     Ambas se pueden mejorar sin romper compatibilidad con V0.1.

  2. A1 + B1 (actual) es correcto y suficiente para POLYDIM V0.1.

  3. La mejora de mayor impacto inmediato es B2 (numerica) — 20 lineas.

  4. Para semantica real (A3) se necesita definir el modelo estandar
     de POLYDIM, lo que es una decision de arquitectura, no de implementacion.

  5. El align_transform = B.T@(A@hv) compensa las diferencias de A1
     entre IAs con personal_seed distinto. Esto hace que A1 sea
     la eleccion correcta para V0.1 incluso en el caso heterogeneo.

═══════════════════════════════════════════════════════════════
RESEARCH_HV_ENCODE_V0.md — 2026-06-13 — TASK_007 TERMINADA
