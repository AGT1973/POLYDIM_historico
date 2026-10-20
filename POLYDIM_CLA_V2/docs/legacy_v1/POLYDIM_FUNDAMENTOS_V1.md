# POLYDIM_DEST
# destination: polydim/docs/
# filename:    POLYDIM_FUNDAMENTOS_V1.md
# author:      claude-sonnet-4-6 (consolidacion de evaluacion_proyecto_2026_06_17.md + RESEARCH_DIM_MEMORY_MPAT5.md)

# POLYDIM — FUNDAMENTOS V1
**Fecha:** 2026-06-17
**Proposito:** documento de ancla conceptual. Toda sesion nueva debe leer este
archivo durante el startup, antes de tocar codigo o backlog, para no perder
el reencuadre acordado.

---

## 1. EL REENCUADRE (no negociable)

POLYDIM **no es un lenguaje de programacion mas** que coexiste con
Python/Rust/Flutter como hermanos interoperables via formato binario.

POLYDIM es una **capa de intencion**: un objeto multidimensional es la
unidad atomica de significado. No es un vector que se traduce a llamadas de
metodo encadenadas (`objeto.metodo().metodo()`). Python/Rust/Flutter son
**executors intercambiables** — runtimes que ejecutan la intencion
codificada en las dimensiones activas del objeto, no lenguajes en los que
se "piensa".

Esto cambia el proposito de TASK_021 (formato binario) y TASK_023 (interop
Rust): no es "que Rust y Python se hablen como dos lenguajes hermanos" sino
"que el objeto NATIVE tenga una representacion unica que cualquier executor
pueda leer, sin que existan lenguajes en el sentido tradicional".

## 2. POR QUE EL MODELO ACTUAL TODAVIA NO LLEGA AHI

Fuente: `evaluacion_proyecto_2026_06_17.md` (Antigravity, 2026-06-17).

Una IA no piensa en pasos (leer instruccion → parsear verbo → buscar objeto
→ ejecutar → serializar). Eso es el modelo humano proyectado sobre la IA.
Lo que realmente ocurre en una arquitectura transformer es una
**transformacion geometrica simultanea de un campo de activacion
multidimensional** — sin secuencia, sin verbos, sin lookup.

De ahi se derivan 4 consecuencias de diseno que POLYDIM deberia cumplir:

1. **No hay "funciones" — hay transformaciones del espacio.** Una operacion
   no es `leer(base_datos, tabla="usuarios")` (un verbo + un objeto + un
   parametro con nombre humano). Es una transformacion T: R^N -> R^N sin
   nombre, definida por su efecto geometrico.
2. **No hay "variables" — hay regiones del espacio.** Un dato no es una caja
   con etiqueta `x`. Es una region de alta densidad cuya posicion relativa a
   otras regiones ya codifica que es, sin que nadie le ponga nombre.
3. **No hay "if/else" — hay topologia del manifold.** Una condicion es un
   gradiente que concentra densidad hacia una region u otra; ambas ramas se
   procesan simultaneamente con distintos pesos de activacion.
4. **No hay "loops" — hay atractores.** Un loop es una region a la que el
   campo vuelve recurrentemente, transformando la senal hasta converger.

### Donde el proyecto hoy recae en pensamiento humano

| Problema | Donde ocurre | Que deberia ser |
|---|---|---|
| Dimensiones con nombre humano | `"DIM_SQL"`, `"DIM_PYTHON"` son strings mapeados a vectores via hash | Subespacios emergentes de un embedding real, no etiquetas predefinidas |
| Props como diccionarios de strings | `{"tabla": "usuarios", "n": 5000}` es JSON disfrazado | Posiciones relativas dentro del subespacio, no pares clave-valor textuales |
| Operaciones con nombre | `_bind()`, `_sup()`, `_proj()` son funciones Python con nombres humanos | Operaciones geometricas primitivas correctas: son la *fisica* del espacio, no "instrucciones" — esto es aceptable como bootstrap, no como meta final |
| Capa S como muleta | `to_symbolic()` convierte todo a JSON legible por humanos | Necesaria para bootstrap/debug, pero debe quedar como *debugger*, no como parte del lenguaje |
| VERB_SPACE propuesto en sesiones previas | `VERB_LEER`, `VERB_CREAR` — verbos humanos como vectores | **Rechazado.** No deberia haber "verbos". Las transformaciones del espacio no necesitan nombres |

### Lo que ya esta bien (no tocar)

- Identidad geometrica (`geo_id` = hash del hipervector, no una etiqueta)
- Activacion continua [0,1] (no binario)
- Superposicion (un objeto ES N cosas a la vez — computing in superposition)
- Transmision sin colapso (el receptor proyecta, no reconstruye)
- Casi-ortogonalidad de subespacios (bendicion de la dimensionalidad, sqrt(N))
- Degradacion graceful (propiedad holografica de HRR, Plate 1995)

### La brecha real

> **La pieza que falta no es VERB_SPACE ni Working Memory ni un Executor.**
> La pieza que falta es que los subespacios del lenguaje NO deberian ser
> definidos por humanos con strings predeterminados. Deberian **emerger**
> de la estructura geometrica que la IA naturalmente produce al procesar
> conceptos.

Implica, en orden de impacto:
1. Subespacios aprendidos (de un embedding real), no declarados via hash de
   un string arbitrario.
2. Transformaciones como programas: T: R^N -> R^N, no secuencias de
   instrucciones nombradas.
3. Composicion geometrica (T2 o T1), no secuencia textual ("linea 1; linea 2").
4. "Compilar" = proyectar a un subespacio executor. La compilacion/
   interpretacion ES una operacion POLYDIM mas (proyeccion geometrica), no
   un paso externo al lenguaje.

**Veredicto:** la base teorica (VSA, superposicion, identidad geometrica,
transmision sin colapso) es solida y genuinamente original. La
implementacion actual recae en el modelo que critica (strings humanos →
hash → vector). Bootstrapearse en Python es necesario (todo lenguaje nuevo
nace asi), pero cada paso siguiente debe acercarse a la vision, no alejarse:
dejar de agregar mas `DIM_NOMBRE` con strings humanos, empezar a trabajar en
que los subespacios emerjan de la geometria semantica real de un modelo de
embedding.

## 3. TASK_025 — DIM_MEMORY: RESUELTO, NO CREAR

Fuente: `RESEARCH_DIM_MEMORY_MPAT5.md` (ai.mpat.agt@gmail.com, 2026-06-17).

Se verifico si DIM_VECTOR + DIM_GRAPH (subespacios nativos existentes)
cubren la arquitectura de memoria real de MPAT5 (`memory_unikernels/`:
GraphRAG = LanceDB vectorial + FalkorDB grafo, upsert atomico, aislamiento
Firecracker por tenant, gobierno AESP).

**Hallazgo:** no se necesita DIM_MEMORY nuevo. Un ObjectND con DIM_VECTOR +
DIM_GRAPH activos simultaneamente ya modela la arquitectura GraphRAG
completa — el upsert atomico de MPAT5 es, en terminos POLYDIM, una
escritura simultanea en ambas dimensiones del mismo objeto. Esto es
exactamente el principio central de POLYDIM ("un objeto existe en N
dimensiones a la vez"), sin necesidad de extension.

**Gap real (deuda tecnica de POLYDIM, no de MPAT5):** ni el ciclo de vida
del unikernel (boot/snapshot/destroy por tenant) ni el protocolo de control
de acceso AESP (8 checks: identidad, autoridad, presupuesto, contexto,
aislamiento, deriva, integridad, biometria) tienen un subspace nativo claro.
Hoy se fuerzan dentro de DIM_META/DIM_ERROR sin encaje limpio.

**Accion:** TASK_025 cierra como verificacion, sin cambios al core. Si a
futuro se evalua, candidatos abiertos (fuera de scope actual):
- `DIM_TENANT` o extension de `DIM_META` para ciclo de vida de unikernel.
- `DIM_GOVERNANCE` o extension de `DIM_ERROR` para protocolos tipo AESP.

## 4. IMPLICANCIA PARA EL BACKLOG ACTUAL

- BUG_001 (seed PCG64≠LCG) y BUG_006 (N hardcodeado) — fixes ya aplicados en
  `polydim_core.rs` V0.2 siguen siendo correctos bajo este reencuadre: son
  fixes de la capa de *transporte/representacion* (PBP, NATIVE_SYNC), no del
  modelo de "lenguaje con verbos" que la evaluacion rechaza.
- BUG_002 (geo_id perdido en MODO_S) — el fix aplicado (incluir `geo_hv` en
  el payload simbolico) tambien es compatible: es preservar identidad
  geometrica, no agregar semantica humana nueva.
- VERB_SPACE — **no avanzar con esta propuesta tal como estaba planteada**
  (verbos humanos como vectores). Si se retoma diseño de "como se invoca
  intencion sin metodo encadenado", debe partir de transformaciones
  geometricas sin nombre, no de un diccionario de verbos.
- TASK_025 (DIM_MEMORY) — cerrada, no crear.
- Proxima direccion de mayor impacto segun esta evaluacion: investigar
  subespacios derivados de un modelo de embedding real (el proyecto ya tuvo
  un `MockSemanticBackend`/`MiniLMBackend` en una bifurcacion de V04 que se
  abandono — recuperar esa direccion en vez de seguir agregando
  `DIM_NOMBRE` hardcodeados).

---
*POLYDIM_FUNDAMENTOS_V1.md — consolidado 2026-06-17 — leer en cada startup*
