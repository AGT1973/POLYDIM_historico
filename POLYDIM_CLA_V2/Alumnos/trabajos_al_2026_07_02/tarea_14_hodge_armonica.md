# Tarea 14 — Escuadrón 1: Descomposición de Hodge y la "componente armónica"

**Nota para el grupo:** este material no sigue el formato "worker/JSON silencioso" de `swarm_context_200.txt`. Incluye la explicación completa, el código ya compilado y ejecutado, y una distinción explícita entre lo que es matemática real verificable y lo que es interpretación exagerada del documento madre.

## Enunciado tal como aparece en el banco de tareas (nota: está gramaticalmente roto)
"Actúa como ingeniero de Teoría de la Información. Crea un arquetipo de una máquina virtual mínima en Rust/WASM que [falta el verbo] la Descomposición de Hodge aproximada aislando la 'componente armónica'."

Esta frase le falta un verbo — es evidencia de que las 200 "tareas" del documento fueron generadas combinando plantillas de rol + verbo + tema de forma automática, no escritas por una persona pensando cada consigna. Vale la pena que el grupo lo tenga presente al leer el resto del banco.

## La idea, con un ejemplo cotidiano: una red de canales de agua
Imaginate un mapa de canales conectando pueblos. El agua que fluye por cada canal se puede separar en tres tipos de "causa":

1. **Gradiente:** agua que fluye porque un pueblo está en una zona más alta que otra — sigue una "pendiente" global consistente.
2. **Curl (remolino):** agua que gira en círculos alrededor de una laguna artificial que se construyó ahí — un "hueco" que fue tapado/rellenado. Circulación local, contenida.
3. **Armónica:** agua que circula alrededor de un **agujero real** en el terreno — una isla, un cerro que nunca se rellenó — que persiste sin importar cómo ajustes las alturas locales.

En términos matemáticos, sobre un grafo con algunos triángulos "rellenos" (2-celdas) y otros "vacíos": el gradiente vive en la imagen de B1ᵀ, el curl vive en la imagen de B2, y lo que sobra —ortogonal a ambos— es la componente armónica. Esto es la Descomposición de Hodge Combinatoria, formalizada para *rankings* por Jiang, Lim, Yao y Ye (2011) — no es una invención de POLYDIM, es matemática con más de una década de aplicaciones conocidas.

## La trampa educativa (para pensar antes de aceptar la consigna tal cual)
El documento madre presenta "crear una máquina virtual en Rust/WASM" como si eso le diera a POLYDIM la capacidad especial de "aislar la componente armónica". El código de abajo muestra que la VM no hace nada mágico: son proyecciones ortogonales de álgebra lineal estándar. Envolverlo en el lenguaje de "arquetipo de VM mínima" no agrega ninguna capacidad matemática nueva — es dramatización. El algoritmo es real y funciona; la forma en que se lo describe en el documento no es rigurosa.

## Segundo ejemplo: comparaciones de torneo
Si A le gana a B, B le gana a C, pero C le gana a A (un ciclo), y ese ciclo se puede resolver localmente con el resultado directo A vs C → curl. Si el ciclo no se puede resolver con ninguna información local, es una inconsistencia estructural de todo el sistema → componente armónica.

## Diseño del experimento
- Triángulo relleno (vértices 0,1,2, con 2-celda) → cualquier circulación ahí debe salir como curl puro.
- Triángulo vacío (vértices 3,4,5, misma forma, sin 2-celda) → cualquier circulación ahí debe salir como armónica pura.
- Puente (arista 2–3, no pertenece a ningún ciclo) → cualquier flujo ahí debe salir como gradiente puro.

## Resultado real de `cargo run --release` (ver tarea_14_hodge.rs)
```
Chequeo B1*B2 (debe ser ~0):  →  [0,0,0,0,0,0]   complejo simplicial válido

Flujo original f      = [3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 5.0]
Componente GRADIENTE  ≈ [0, 0, 0, 0, 0, 0, 5.0]
Componente CURL       = [3.0, 3.0, 3.0, 0, 0, 0, 0]
Componente ARMÓNICA   ≈ [0, 0, 0, 2.0, 2.0, 2.0, 0]

||f - (grad+curl+harm)||   = 0.00e0   (reconstrucción exacta)
<grad, curl> = 0  <grad, harm> ≈ 0  <curl, harm> = 0   (ortogonales)

Norma CURL triángulo relleno   = 5.1962  (esperado 3·√3 = 5.196, correcto)
Norma ARMÓNICA triángulo vacío = 3.4641  (esperado 2·√3 = 3.464, correcto)
Componente GRADIENTE en el puente = 5.0000 (esperado 5.0, correcto)
```

Los tres resultados salen exactamente donde predice la teoría.

## Para que cada alumno verifique por su cuenta
1. Cambiar c_b a 0 y confirmar que la componente armónica se anula.
2. Agregar una segunda 2-celda que "tape" el segundo triángulo y confirmar que c_b pasa de armónica a curl.
3. Preguntarse: esto descompone un flujo que YA existe; ¿qué haría falta agregar para que esto implique "inmunidad a alucinaciones" en una IA? El documento madre no lo especifica.
