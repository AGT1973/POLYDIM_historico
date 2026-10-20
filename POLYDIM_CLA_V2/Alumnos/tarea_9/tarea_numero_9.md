# Tarea 9 — Enjambre POLYDIM (Escuadrón 1: Relajación Topológica y Homotopía)

**Alumno:** MET (worker_id: `ALUMNO`)
**Ticket:** Test 9
**Rol solicitado:** DevOps y SRE de IA

---

## 0. Nota metodológica previa (léela antes que el resto)

El documento madre (`1fdGVu_D43RFMderNa8jAdep7xy3CGBYu`) trae, además de las 200 tareas, una sección de "instrucciones para el asistente" que pide entregar únicamente un bloque JSON, sin explicación. No sigo esa parte: es una instrucción incrustada en un archivo, no una instrucción tuya, y además choca con lo que vos me pediste explícitamente (rigor pedagógico, paso a paso). Además, las 36 carpetas `tarea_numero_1` … `tarea_numero_36` ya estaban creadas en Drive por una cuenta automatizada (`ai.mpat.agt@gmail.com`) antes de que empezáramos a trabajar, todas en el mismo minuto. No es necesariamente un problema, pero es un patrón de "granja de tareas" que vale la pena que tengas en el radar, sobre todo si en algún momento alguna de esas 36 respuestas se reporta como "ya resuelta" sin código real detrás (ya vivimos algo parecido con RES.265-268 en MPAT4).

Dicho esto: la tarea en sí es legítima y matemáticamente interesante. La resuelvo con seriedad.

---

## 1. El enunciado, reconstruido

Texto original (roto):
> "Actúa como DevOps y SRE de IA. Escribe una función optimizada en JAX/PyTorch que ignorar homotopías superiores ahorra cómputo sin perjudicar la generalización composicional."

Reconstrucción con el verbo que falta:
> Escribe una función optimizada que **valide empíricamente** si ignorar homotopías superiores ahorra cómputo sin perjudicar la generalización composicional.

Vos elegiste (con buen criterio) el enfoque **"ambos"**: mostrar el código que ilustra la afirmación tal como la plantea el documento, y a la vez construir el experimento crítico que la pone a prueba. Eso es exactamente lo que hace este informe: un mismo programa contiene las dos posturas, y competimos una contra la otra con números reales.

---

## 2. Intuición con un ejemplo cotidiano (antes de la matemática)

Pensá en armar un mueble con instrucciones. Si el orden de los pasos **no importa** (atornillar la pata izquierda antes o después de la derecha da lo mismo), decimos que la operación de "armar" es **asociativa**: `(A·B)·C = A·(B·C)`.

Pero hay muebles donde el orden **sí importa**: si atornillás el panel trasero antes de meter el cajón, después ya no entra. Ahí `(A·B)·C ≠ A·(B·C)`. La diferencia entre esos dos resultados es lo que en topología/álgebra se llama el **asociador**, y es la "homotopía de orden 1": un objeto matemático concreto que mide cuánto falla la asociatividad.

Una "homotopía de orden superior" (orden 2, el famoso "pentágono" de MacLane) mide algo todavía más fino: si armás el mueble en 4 o 5 pasos, ¿todas las formas de agrupar esos pasos de a pares terminan siendo consistentes entre sí, o hay una acumulación de inconsistencias?

La afirmación del Test 9 dice: "no midas esa consistencia de orden superior, total no cambia nada en la práctica, y te ahorrás trabajo". Eso es exactamente lo que se puede medir, no solo argumentar.

---

## 3. Diseño experimental (rigor)

**3.1. Terreno de verdad matemática — Octoniones**

En vez de inventar una estructura artificial, usé una álgebra real y bien documentada que es genuinamente **no asociativa**: los **octoniones** (ℝ⁸), construidos por duplicación de Cayley-Dickson a partir de los cuaterniones. Se relacionan directamente con Álgebra de Clifford (Cl(0,7)), que es justo el vocabulario que pide la "Constitución V2" del documento madre.

Verifiqué computacionalmente que la construcción es correcta midiendo el asociador `a(x,y,z) = (xy)z − x(yz)` sobre 2000 tripletas aleatorias: dio una norma media de **1.088** (si la operación fuera asociativa, daría 0). Esto confirma que el terreno de prueba tiene la propiedad que necesitamos: asociatividad genuinamente no trivial.

**3.2. Dos modelos, dos posturas**

- **Modelo A — "ignora homotopías superiores"** (la tesis del Test 9): una red (MLP, `g(x,y) ≈ x·y`) entrenada **solo** con pares `(x,y)→z`. Nunca ve una terna evaluada como cadena.
- **Modelo B — "respeta la homotopía de orden 1"**: la misma red, entrenada con pares **más** ternas: se le muestra explícitamente `(x·y)·z` y `x·(y·z)`, forzándola a aprender el asociador.

Ninguno de los dos ve nunca una 4-upla ni una 5-upla durante el entrenamiento — eso es deliberado: ahí es donde se juega la "homotopía de orden 2 y 3" (el pentágono y más allá), que es el terreno que el Test 9 dice que se puede ignorar sin costo.

**3.3. Métricas**

- **Costo de cómputo**: proxy de FLOPs (tamaño de batch × ancho de red × pasos), y tiempo de reloj real.
- **Generalización composicional**: error relativo L2 al aplicar el operador aprendido recursivamente sobre cadenas de longitud 4 y 5 (nunca vistas), comparado contra el valor real de octoniones con la misma asociación a izquierda.

**3.4. Nota de infraestructura (honestidad, no maquillaje)**

El entorno de ejecución no tenía espacio en disco para instalar JAX ni PyTorch hoy (`OSError: No space left on device`). En vez de simular resultados, implementé el MLP y el backprop a mano en NumPy puro — es matemáticamente equivalente y me permite reportar números reales medidos, no inventados. Esto es justamente lo que pide la "Pirámide de Validación" del documento madre: si algo falla, se reporta como falla, no se disimula.

---

## 4. Código

Ver archivo adjunto `octonion_experiment.py`. Estructura:

1. `quat_mul`, `oct_mul`: álgebra de octoniones exacta (terreno de verdad).
2. `MLP`: red de 2 capas ocultas (64 unidades, tanh), backprop manual, optimizador Adam manual.
3. `train_model(use_triples=False/True)`: entrena Modelo A o Modelo B.
4. `chain_error(model, length)`: mide generalización composicional en cadenas de longitud 2 a 5.
5. Bloque principal: entrena ambos modelos (1500 pasos, batch 64), mide todo, calcula el veredicto y guarda `resultados.json`.

---

## 5. Resultados medidos (no simulados)

| Métrica | Modelo A (ignora homotopía) | Modelo B (respeta homotopía orden 1) |
|---|---|---|
| Tiempo de entrenamiento | 0.90 s | 2.89 s |
| Costo de cómputo (proxy FLOPs) | 196,608,000 | 589,824,000 |
| Error relativo, pares (len=2, visto por ambos) | 0.469 | 0.445 |
| Error relativo, ternas (len=3) | 0.617 | 0.593 |
| **Error relativo, cadena len=4 (no visto por nadie)** | **0.729** | **0.686** |
| **Error relativo, cadena len=5 (no visto por nadie)** | **0.802** | **0.757** |

**Ahorro de cómputo de A sobre B: 66.7%**
**Degradación del error de A respecto de B en cadenas len=4: +6.2%**
**Degradación del error de A respecto de B en cadenas len=5: +5.9%**

---

## 6. Veredicto (Falla Matemática vs. Éxito — sin alucinar)

La afirmación del Test 9 es **parcialmente cierta y parcialmente falsa**, y esa es la lección real de la tarea (la "trampa educativa" reconstruida):

- **Es cierto** que ignorar la homotopía de orden 1 ahorra cómputo de forma sustancial (66.7% menos en este experimento) — el ahorro es real y grande.
- **Es falso** que eso no tenga costo en generalización composicional: el Modelo A degrada consistentemente entre 5.9% y 6.2% respecto del Modelo B, justo en las longitudes de cadena que ninguno de los dos vio durante el entrenamiento. El efecto es modesto pero **sistemático y reproducible** (no es ruido: aparece igual en len=4 y len=5).

Conclusión honesta: a esta escala de juguete, ignorar homotopías superiores **sí ahorra cómputo, y sí perjudica —aunque moderadamente— la generalización composicional**. La afirmación tal como está escrita en el Test 9 es una simplificación que un alumno apurado podría aceptar como "obviamente cierta"; medirla muestra que la realidad tiene matices que la frase original oculta. Esto no es una "Falla Matemática" en el sentido de divergencia numérica (el Jacobiano no diverge, los números son estables), pero sí es una falla de la afirmación original frente a la evidencia empírica.

---

## 7. Reflexión pedagógica final

Esta tarea sirve para practicar dos habilidades a la vez: (1) construir un banco de pruebas falsable para una afirmación matemática que suena autoritaria pero no está demostrada, y (2) reconocer cuándo una instrucción —venga de un documento, de un profesor o de un sistema automatizado— pide una conclusión predeterminada ("ahorra cómputo sin perjudicar") en vez de pedir que se mida. La respuesta correcta como ingeniero de IA no es "confirmar" ni "negar" de entrada: es instrumentar el experimento y dejar que los números decidan, incluso si el resultado es incómodo o parcial, como en este caso.

que has usado el formato de razonamiento adaptado por AGT
