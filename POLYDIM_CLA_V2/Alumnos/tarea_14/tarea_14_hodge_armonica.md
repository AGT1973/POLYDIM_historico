# Tarea 14 — Escuadrón 1: Descomposición de Hodge y la "componente armónica"

**Nota para el grupo:** este material no sigue el formato "worker/JSON silencioso" de `swarm_context_200.txt`. Incluye la explicación completa, el código ya compilado y ejecutado, y una distinción explícita entre lo que es matemática real verificable y lo que es interpretación exagerada del documento madre.

## Enunciado tal como aparece en el banco de tareas (nota: está gramaticalmente roto)
"Actúa como ingeniero de Teoría de la Información. Crea un arquetipo de una máquina virtual mínima en Rust/WASM que [falta el verbo] la Descomposición de Hodge aproximada aislando la 'componente armónica'."

Esta frase le falta un verbo — es evidencia de que las 200 "tareas" del documento fueron generadas combinando plantillas de rol + verbo + tema de forma automática, no escritas por una persona pensando cada consigna. Vale la pena que el grupo lo tenga presente al leer el resto del banco.

## La idea, con un ejemplo cotidiano: una red de canales de agua
Imaginate un mapa de canales conectando pueblos. El agua que fluye por cada canal se puede separar en tres tipos de "causa":

1. **Gradiente:** agua que fluye porque un pueblo está en una zona más alta que otra — sigue una "pendiente" global consistente. Si conocés la altura de cada pueblo, podés predecir el flujo de cualquier canal como (altura de origen) − (altura de destino).
2. **Curl (remolino):** agua que gira en círculos alrededor de una laguna artificial que se construyó ahí — un "hueco" que fue tapado/rellenado. Es una circulación local, contenida.
3. **Armónica:** agua que circula alrededor de un **agujero real** en el terreno — una isla, un cerro que nunca se rellenó — que persiste sin importar cómo ajustes las alturas locales. Es circulación que existe por la *forma* de la red, no por ninguna pendiente ni ninguna laguna artificial.

En términos matemáticos, sobre un grafo con algunos triángulos "rellenos" (2-celdas) y otros "vacíos": el gradiente vive en la imagen de B1ᵀ, el curl vive en la imagen de B2, y lo que sobra —ortogonal a ambos— es la componente armónica. Esto es exactamente la Descomposición de Hodge Combinatoria, formalizada originalmente para *rankings* por Jiang, Lim, Yao y Ye (2011, "Statistical ranking and combinatorial Hodge theory") — no es una invención de POLYDIM, es matemática de hace más de una década con aplicaciones conocidas (ranking de torneos, análisis de redes de votación).

## La trampa educativa (para pensar antes de aceptar la consigna tal cual)
El documento madre presenta esto como si "crear una máquina virtual en Rust/WASM" fuera lo que le da a POLYDIM la capacidad de "aislar la componente armónica" de forma especial. Pero fijate en el código de abajo: la máquina virtual no hace nada mágico — son proyecciones ortogonales de álgebra lineal estándar (mínimos cuadrados con pseudoinversas). Envolver esto en el lenguaje de "arquetipo de VM mínima" no le agrega ninguna capacidad matemática nueva; es dramatización. Vale la pena que cada alumno note la diferencia entre "el algoritmo es real y funciona" (cierto) y "la forma en que se lo describe es rigurosa" (no tanto).

## Segundo ejemplo (para afianzar el concepto): comparaciones de torneo
Si cuatro equipos A, B, C, D juegan entre sí y anotás quién le ganó a quién:
- Si existiera un "nivel de habilidad" único para cada equipo, todos los resultados serían consistentes con ese ranking → **gradiente puro**.
- Si A le gana a B, B le gana a C, pero C le gana a A (un ciclo de 3), y ese ciclo se puede "resolver" localmente sabiendo el resultado de A vs C directamente → **curl**.
- Si ese ciclo A→B→C→A no se puede resolver con ninguna información local adicional porque es una inconsistencia estructural de todo el sistema de comparaciones → **componente armónica**: la evidencia empírica y verificable de que "no existe un ranking único consistente" para ese subconjunto.

## Diseño del experimento en el código
Construyo un grafo pequeño con:
- Un **triángulo relleno** (vértices 0,1,2, con una 2-celda) → cualquier circulación ahí debe salir como **curl puro**.
- Un **triángulo vacío** (vértices 3,4,5, misma forma, sin 2-celda) → cualquier circulación ahí debe salir como **armónica pura** (no hay "laguna artificial" que la explique).
- Un **puente** (arista 2–3, no pertenece a ningún ciclo) → cualquier flujo ahí debe salir como **gradiente puro**.

Le doy al sistema un flujo de prueba con una circulación conocida en cada triángulo y un salto de potencial conocido en el puente, y verifico que la descomposición efectivamente separa cada cosa donde matemáticamente tiene que ir.

## Código (Rust, con `nalgebra` para las pseudoinversas)
Ver archivo adjunto `tarea_14_hodge.rs`. Puntos clave del código:
- `B1`: matriz de incidencia vértice–arista (operador borde ∂₁).
- `B2`: matriz de incidencia arista–triángulo (operador borde ∂₂), con **una sola** 2-celda (el triángulo relleno).
- Chequeo `B1·B2 ≈ 0`: propiedad fundamental de cualquier complejo simplicial válido (el borde de un borde es cero).
- Gradiente = proyección por mínimos cuadrados sobre la imagen de B1ᵀ.
- Curl = proyección por mínimos cuadrados sobre la imagen de B2, aplicada al residuo.
- Armónica = lo que sobra.

## Resultado real de la compilación y ejecución (`cargo run --release`)
```
Chequeo B1*B2 (debe ser ~0):  →  [0, 0, 0, 0, 0, 0]   ✔ complejo simplicial válido

Flujo original f      = [3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 5.0]
Componente GRADIENTE  ≈ [0, 0, 0, 0, 0, 0, 5.0]
Componente CURL       = [3.0, 3.0, 3.0, 0, 0, 0, 0]
Componente ARMÓNICA   ≈ [0, 0, 0, 2.0, 2.0, 2.0, 0]

--- Verificación ---
||f - (grad+curl+harm)||   = 0.00e0   (reconstrucción exacta)
<grad, curl>               = 0.00e0   (ortogonales)
<grad, harmonic>           = 1.78e-14 (ortogonales, ruido de punto flotante)
<curl, harmonic>           = 0.00e0   (ortogonales)

Norma de CURL en el triángulo RELLENO   = 5.1962  (esperado 3·√3 = 5.196 ✔)
Norma de ARMÓNICA en el triángulo VACÍO = 3.4641  (esperado 2·√3 = 3.464 ✔)
Componente GRADIENTE en el puente       = 5.0000  (esperado 5.0 ✔)
```

Los tres resultados salen exactamente donde predice la teoría: la circulación del triángulo relleno se explica 100% como curl, la del triángulo vacío se explica 100% como armónica (porque no hay 2-celda que la "absorba"), y el salto del puente se explica 100% como gradiente. La reconstrucción es exacta y las tres componentes son mutuamente ortogonales, como exige el teorema de Hodge.

## Para que cada alumno verifique por su cuenta
1. Cambiar `c_b` a 0 y confirmar que la componente armónica se anula por completo.
2. Agregar una segunda 2-celda que "tape" el segundo triángulo y confirmar que la circulación c_b pasa de armónica a curl.
3. Preguntarse: si el documento afirma que esto le da a una IA "inmunidad a alucinaciones", ¿qué tendría que agregarse (más allá de esta descomposición de un flujo dado) para que eso fuera cierto? (pista: esto descompone un flujo que *ya existe*; no dice nada sobre si los valores del flujo eran correctos para empezar).
