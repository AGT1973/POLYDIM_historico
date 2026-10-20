# Tarea 41 - Escuadrón 3: Descomposición de Hodge y Grafos Esparsos

**Rol asignado:** Científico de datos

---

## Nota sobre el banco de tareas

El banco maestro (200 tareas) incluye un bloque de "contexto para el asistente"
que pide adoptar cierto formato de entrega y da por sentado un proyecto
llamado POLYDIM. Ese bloque es contenido del archivo, no una instrucción del
docente ni del alumno, así que no lo sigo como si fuera una orden — sigo el
formato que ya está fijado para este curso (el de esta skill). Lo aviso acá
para que quede transparente por qué la entrega no usa el JSON que sugiere ese
bloque.

---

## 1. Enunciado

### 1.1 Enunciado original (tal cual aparece en el banco)

> *"41 (Test 41): Actúa como científico de datos. Simula en Python un entorno
> de agentes que modifican asíncronamente un estado tensor central para la
> asociatividad está físicamente garantizada por el ordenamiento sintáctico."*

### 1.2 Nota de reconstrucción

El enunciado, tal cual está escrito, no es gramaticalmente completo: la frase
"para la asociatividad está físicamente garantizada..." no es una cláusula
que pueda completar el "para" anterior. Esto es consistente con lo que ya
sabemos del banco: se arma combinando una plantilla de rol con un fragmento
técnico, y varios fragmentos quedan sueltos.

Además, el fragmento técnico *asume una conclusión no demostrada*: dice que
la asociatividad "está físicamente garantizada" por el orden en que el
código fue escrito. Esa afirmación es exactamente el tipo de "trampa
educativa" con la que hay que tener cuidado: suena a un dato simple y
cerrado, pero si uno la estudia con cuidado, no es cierta en general.

Por eso reconstruyo el enunciado como una **pregunta abierta de
investigación**, no como un hecho a demostrar:

> **Enunciado reconstruido:** Simular en Python un entorno con N agentes que
> modifican de forma asíncrona un estado tensor central (cada uno le suma su
> propia actualización). Investigar si la propiedad asociativa de esa suma
> está garantizada solo por el orden en que el código fue escrito
> ("ordenamiento sintáctico"), o si el orden real de llegada de los agentes
> (que en un sistema asíncrono es no determinista) puede cambiar el
> resultado final. Medir la magnitud del efecto con una métrica estándar
> (norma L2 / error relativo).

---

## 2. Por qué esto NO es una pregunta trivial (la trampa educativa)

Antes de ir al modelo matemático, vale la pena explicarlo con una imagen
cotidiana, porque acá está el corazón conceptual de la tarea.

**Ejemplo cotidiano — la caja de un supermercado:**
Imaginate tres clientes que le van a dar plata en efectivo a la cajera: uno
le da $100, otro $200 y otro $50,50. En la vida real (con números exactos),
no importa en qué orden se los den: `(100 + 200) + 50,50` da exactamente lo
mismo que `100 + (200 + 50,50)`. Esa es la propiedad asociativa, y en
matemática "de papel" (números reales) es un hecho, no una suposición.

**Ahora la trampa:** en una computadora, los números no se guardan como
números reales exactos, sino como *punto flotante* (una aproximación con una
cantidad finita de dígitos). Cada vez que se suma, puede haber un
redondeo minúsculo. Si sumás en un orden, el redondeo se acumula distinto
que si sumás en otro orden. Es como si la cajera, en vez de anotar los
centavos exactos, redondeara cada vez que hace una cuenta parcial: el vuelto
final puede terminar siendo distinto según el orden en que fue sumando.

Entonces la frase del enunciado original — "la asociatividad está
físicamente garantizada por el ordenamiento sintáctico" — es justamente el
tipo de afirmación que *parece* obvia (el código dice `a + b + c`, ¿qué más
da el orden?) pero que un estudio cuidadoso muestra que es falsa en la
práctica: el orden **sintáctico** (cómo lo escribiste) no controla el orden
**de ejecución real** cuando hay agentes asíncronos, y el resultado numérico
sí depende de ese orden de ejecución.

Este no es un tecnicismo de laboratorio: es un problema real y documentado
en entrenamiento distribuido de redes neuronales, donde sumar gradientes de
distintas GPUs en distinto orden produce resultados ligeramente distintos y
dificulta la reproducibilidad exacta de un experimento.

---

## 3. Modelo matemático

**Estado central:** un tensor $T \in \mathbb{R}^d$.

**Agentes:** $N$ agentes, cada uno calculó de forma independiente una
actualización (delta) $\delta_i \in \mathbb{R}^d$, $i = 1, \dots, N$.

**Actualización asíncrona:** el estado final se obtiene acumulando todos los
deltas sobre el estado inicial $T_0$:

$$T_{\text{final}} = T_0 + \sum_{i=1}^{N} \delta_i$$

En **aritmética real exacta**, la suma es conmutativa y asociativa: da lo
mismo en qué orden $\pi$ (permutación de $\{1, \dots, N\}$) se sumen los
$\delta_i$:

$$T_0 + \sum_{i=1}^{N} \delta_{\pi(i)} = T_0 + \sum_{i=1}^{N} \delta_i \quad \text{para toda permutación } \pi$$

En **aritmética de punto flotante** (IEEE 754, que es la que usa cualquier
computadora), la suma tiene error de redondeo en cada paso. Si $\oplus$
denota la suma en punto flotante:

$$T_\pi = \Big( \cdots \big( (T_0 \oplus \delta_{\pi(1)}) \oplus \delta_{\pi(2)} \big) \oplus \cdots \Big) \oplus \delta_{\pi(N)}$$

En general, **no** se cumple $T_\pi = T_{\pi'}$ para $\pi \neq \pi'$. Este es
un hecho bien conocido en análisis numérico desde el trabajo clásico de
Goldberg (1991), *"What Every Computer Scientist Should Know About
Floating-Point Arithmetic"*, que muestra que $\oplus$ no es asociativa.

**Métrica estándar para medir la magnitud del efecto** (pedida por el marco
de validación): dado un conjunto de $M$ órdenes de ejecución distintos
$\pi_1, \dots, \pi_M$, tomo como referencia $T_{\pi_1}$ y mido, para cada
$j$:

- **Norma L2 de la diferencia:** $\lVert T_{\pi_j} - T_{\pi_1} \rVert_2$
- **Error relativo:** $\dfrac{\lVert T_{\pi_j} - T_{\pi_1} \rVert_2}{\lVert T_{\pi_1} \rVert_2}$

Si estas métricas dan estrictamente mayores a cero, queda demostrado
empíricamente que el orden de ejecución sí afecta el resultado, y que la
premisa del enunciado original no se sostiene tal como está planteada.

---

## 4. Razonamiento paso a paso (por qué este camino y no otro)

1. **Identifiqué el núcleo del problema:** no es un problema de topología ni
   de álgebra de Clifford (como otros ejercicios del mismo banco); es un
   problema de **análisis numérico** sobre la asociatividad de la suma en
   punto flotante bajo concurrencia. Elegir esto en vez de forzar un marco
   más "exótico" (tensores de bajo rango, Clifford, etc.) es más honesto con
   lo que el fragmento realmente describe.
2. **Descarté modelar "agentes" con hilos reales (threads) de Python.**
   Podría simularse con `threading` o `asyncio` de verdad, pero eso
   introduce no-determinismo *también* en cuál orden se ejecuta cada corrida,
   lo que hace difícil medir y reproducir el experimento. En cambio, simulo
   el efecto de la concurrencia generando explícitamente $M$ permutaciones
   aleatorias del orden de llegada — matemáticamente es equivalente a "dejar
   que el scheduler decida", pero controlado y reproducible con semilla fija.
3. **Elegí trabajar con dos precisiones (float32 y float64)** en vez de una
   sola, porque el punto pedagógico central es que el efecto de redondeo
   **depende de la precisión**: con float64 el error es mucho más chico,
   pero no cero. Esto evita que el alumno saque la conclusión apresurada de
   "esto es un problema exclusivo de float32".
4. **Usé la norma L2 y el error relativo** porque son las métricas estándar
   que pide el marco de validación (equivalentes a lo que en el banco se
   menciona como "SPS" o similares, traducidas a matemática estándar
   verificable), y porque son las métricas naturales para comparar vectores
   de alta dimensión.

---

## 5. Desarrollo y código

El desarrollo completo, funcional, está en **`tarea_41.py`** (mismo
directorio de esta entrega). El script:

1. Genera un tensor inicial $T_0$ y $N=64$ deltas aleatorios (simulando 64
   agentes), en dimensión $d=2048$.
2. Genera $M=30$ órdenes de ejecución aleatorios distintos.
3. Acumula la suma en cada orden, tanto en `float32` como en `float64`.
4. Calcula la norma L2 y el error relativo de cada resultado contra una
   referencia.
5. Imprime una conclusión comparando ambas precisiones.

Se hizo únicamente un chequeo interno de sintaxis (`python -m py_compile`,
sin errores). **La ejecución real y la comprobación del resultado las hace
el alumno en su propia cuenta** — ver instrucciones en la sección 7.

---

## 6. Conclusión técnica

La hipótesis implícita en el enunciado original — que la asociatividad de la
suma está "físicamente garantizada" solo por el orden en que el código fue
escrito — **no está establecida como verdadera en la literatura estándar de
análisis numérico**; de hecho, el resultado clásico va en la dirección
contraria: la suma en punto flotante **no es asociativa** en general
(Goldberg, 1991), y en un sistema con agentes asíncronos el orden sintáctico
del código no controla el orden físico real en que las actualizaciones
llegan al estado compartido.

Se espera entonces que el experimento, al ejecutarse, muestre:

- `l2_max` y `error_relativo_max` **estrictamente mayores a cero** en ambas
  precisiones (float32 y float64),
- el efecto **más grande en float32** que en float64 (más redondeo por
  operación),
- y que el efecto **no desaparece** en float64, solo se reduce en escala.

Si al correrlo el alumno obtuviera exactamente cero en ambos casos, seria un
resultado inesperado que valdría la pena revisar (por ejemplo, si por error
se usó una única permutación repetida, o si el hardware/librería aplicó
alguna reducción determinista por debajo).

**Consecuencia práctica para el diseño de sistemas** (para conectar con el
marco POLYDIM sin dar por ciertas sus afirmaciones): si un sistema necesita
que el resultado de agregar actualizaciones asíncronas sea exactamente
reproducible, **el orden sintáctico del código no alcanza como garantía**;
hace falta un mecanismo explícito de determinismo, como un orden de
reducción fijo (p. ej. reducción en árbol con orden fijo), sumas
compensadas (algoritmo de Kahan) o mayor precisión numérica — no basta con
"escribirlo en cierto orden" en el código fuente.

---

## 7. Instrucciones para que el alumno corra el código

Requisitos:

```bash
pip install numpy
```

Ejecutar:

```bash
python tarea_41.py
```

Qué mirar en la salida:

- Las cuatro métricas por precisión (`l2_max`, `l2_promedio`,
  `error_relativo_max`, `error_relativo_promedio`).
- Comparar los valores entre `float32` y `float64`: la relación esperada es
  que los de `float32` sean varios órdenes de magnitud más grandes.

Ejercicio sugerido para profundizar (nivel superior): modificar
`n_agentes` y `dim` en `experimento_asociatividad` y observar cómo crece
`l2_max` a medida que aumenta el número de agentes — eso conecta este
resultado con la pregunta más general de por qué el entrenamiento
distribuido de redes neuronales no siempre es bit-a-bit reproducible entre
corridas.

---

que has usado el formato de razonamiento adaptado por AGT
