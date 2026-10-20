# POLYDIM_DEST
# destino: polydim/research/
# nombre:  RESEARCH_HV_ENCODE_V0.md
# autor:   ai.mpat.agt@gmail.com

# RESEARCH — HV_ENCODE: traduccion Capa S → Capa G
# Version: V0.1 — 2026-06-12
# TASK_007 — Resuelve PEND_007

---

## PROBLEMA

HV_ENCODE es la operacion que traduce un OBJECT_ND en Capa S
(descripcion simbolica con props y pesos) a un hipervector en Capa G.

La implementacion actual en polydim_runtime_v01.py usa codificacion
deterministica: hv = BIND(dim_sub, ENCODE_PROPS(props)).

Esta codificacion es correcta para deteccion de membresía dimensional
(que dimensiones tiene el objeto) pero NO captura semantica real:
  {"tipo": "dict"}  y  {"tipo": "DataFrame"}
producen hipervectores no relacionados aunque son conceptualmente cercanos.

---

## LO QUE LA IMPLEMENTACION ACTUAL HACE BIEN

La codificacion deterministica actual:
  BIND(dim_sub, ENCODE_PROPS(props))

Garantiza:
  - Reproducibilidad: mismo objeto → mismo hipervector siempre
  - Ortogonalidad entre dimensiones: DIM_SQL y DIM_PYTHON son casi-ortogonales
  - Deteccion de membresía confiable (score ALIGN = 0.9993)
  - Sin dependencia de modelos externos

Es la base correcta para V0.1.

---

## LO QUE FALTA: SEMANTICA EN LAS PROPIEDADES

{"tipo": "dict"} y {"tipo": "DataFrame"} deberían estar cerca en Capa G.
{"tabla": "usuarios"} y {"tabla": "clientes"} deberían estar cerca.
{"widget": "Form"} y {"widget": "TextField"} deberían estar cerca.

Esto requiere que get_symbol(nombre) retorne vectores semanticamente relacionados
para conceptos relacionados, no vectores arbitrarios.

---

## TRES ENFOQUES INVESTIGADOS

### Enfoque 1 — Sentence Transformers (estado del arte 2025)
Usar un modelo de embeddings pre-entrenado para asignar hipervectores semanticos.

Modelos candidatos (evaluados por tamaño/performance, datos 2024-2025):
  - all-MiniLM-L6-v2 (90MB, 384 dims): el mas usado, buena relacion tamaño/calidad
  - paraphrase-multilingual-MiniLM-L12-v2 (470MB): soporte multilingue incluye castellano
  - bge-small-en-v1.5 (133MB): mejor que MiniLM en benchmarks MTEB 2024

Implementacion:
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('all-MiniLM-L6-v2')

  def get_symbol_semantic(name):
      embedding = model.encode(name)       # 384 dims
      # Escalar a POLYDIM_N via proyeccion aleatoria (Johnson-Lindenstrauss)
      R_jl = get_projection_matrix(384, POLYDIM_N)  # deterministico
      hv = R_jl @ embedding
      return hv / np.linalg.norm(hv)

Ventaja: semantica real, "usuario" cerca de "cliente", "dict" cerca de "DataFrame"
Desventaja: dependencia de modelo externo, latencia en primera carga (~2s)
Conclusion: RECOMENDADO para POLYDIM V1.0 cuando haya mayor infraestructura

### Enfoque 2 — FastText / Word2Vec (ligero, sin dependencias pesadas)
Modelos de word embeddings pre-entrenados, mucho mas livianos.

FastText en castellano: cc.es.300.bin (7.5GB completo, 100MB para top-50k)
Word2Vec: Google News (1.5GB)

Implementacion:
  import fasttext
  model = fasttext.load_model('cc.es.300.bin')
  hv_base = model.get_word_vector(name)  # 300 dims
  # Proyeccion a POLYDIM_N via Johnson-Lindenstrauss

Ventaja: liviano, sin GPU, castellano nativo, OOV handling
Desventaja: semantica de palabras, no de frases; modelo pesa varios GB
Conclusion: VIABLE para instalaciones sin internet o con recursos limitados

### Enfoque 3 — Codificacion Hibrida (recomendada para V1.0)

Combina la codificacion deterministica actual con embeddings semanticos.

Dos capas en el simbolo:
  hv_simbolo = alpha * hv_deterministico + (1-alpha) * hv_semantico
  alpha = 0.3 para priorizar semantica, 0.7 para priorizar reproducibilidad

La capa deterministica garantiza identidad estable entre sesiones.
La capa semantica acerca conceptos relacionados.

Implementacion posible sin dependencias:
  # Usar la tabla de simbolos como vocabulario semantico propio
  # Dos simbolos semanticamente relacionados = similitud mayor
  # Construir relaciones manualmente para los conceptos del dominio

---

## JOHNSON-LINDENSTRAUSS — PROYECCION A N DIMENSIONES

Para escalar embeddings de baja dimension (384, 300) a POLYDIM_N (10000):

El teorema JL garantiza que una proyeccion aleatoria (R, deterministica desde seed)
preserva las distancias coseno con alta probabilidad.

  R: np.random.default_rng(seed).standard_normal((POLYDIM_N, d))
  hv_N = normalize(R @ hv_d)

Costo: O(N * d) por simbolo. Para d=384, N=10000: 4 millones de operaciones.
Aceptable para tablas de simbolos de tamaño moderado (< 10000 simbolos).

---

## DECISION PARA POLYDIM

Para V0.1 (actual): codificacion deterministica. Estable, reproducible, sin dependencias.
Para V1.0: codificacion hibrida con sentence-transformers como backend opcional.

El Space deberia tener un parametro semantic_backend:
  Space(ps="IA_A", semantic_backend=None)         # V0.1 actual
  Space(ps="IA_A", semantic_backend="miniLM")      # V1.0 con embeddings
  Space(ps="IA_A", semantic_backend="fasttext_es") # alternativa liviana

La interfaz permanece identica. El cambio es transparente para el alumno.

---

## RELACION CON PEND_001 (formato binario)

El formato de transmision del hipervector entre IAs es independiente
del metodo de codificacion. Con cualquiera de los tres enfoques,
el hipervector resultante es un array float32 de N dimensiones.

Para PEND_001: el formato binario candidato es numpy .npy (binario nativo),
protobuf con campo float32 repetido, o msgpack con arrays.
El mas eficiente para N=10000: .npy o msgpack. No requiere compresion
dado que los hipervectores son ya vectores continuos (no hay redundancia estructural).

---

## CONCLUSION

La codificacion deterministica actual es correcta y suficiente para POLYDIM V0.1.
La semantica real en propiedades es PEND_007 con camino claro hacia sentence-transformers.
No bloquea ninguna tarea actual. Es una mejora de calidad, no una correccion de bug.

---
*RESEARCH_HV_ENCODE_V0.md — V0.1 — 2026-06-12 — TASK_007 TERMINADA — PEND_007 PARCIALMENTE RESUELTO*
