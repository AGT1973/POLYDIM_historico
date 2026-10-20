# Tarea 39 — Enjambre POLYDIM

## 1. Enunciado original (banco maestro, Escuadrón 6)

> "39. (Test 39): Actúa como experto en Graph Neural Networks. Diseña un
> kernel en Triton ultra-optimizado para realizar el mayor valor singular
> para asegurar convergencia topológica en FIXPOINT."

**Nota de reconstrucción (paso 4 de la metodología):** el fragmento está
gramaticalmente roto — "diseñar un kernel *para realizar* el mayor valor
singular" no es una acción bien formada (un kernel no "realiza" un valor
singular, lo *calcula* o lo *estima*). Esto es típico del banco: se generó
combinando una plantilla de rol ("Actúa como experto en Graph Neural
Networks") con un fragmento técnico de otra plantilla (sobre el bucle
`FIXPOINT` y el "mayor valor singular", ver Apéndice del paper POLYDIM,
sección 9.4).

Versión reconstruida y matemáticamente investigable:

> Diseñar (con un kernel en Triton como referencia de implementación en
> GPU) un procedimiento que **estime el mayor valor singular** del
> Jacobiano del operador de actualización de una capa tipo GNN implícita,
> y usar ese número como **certificado de contracción** que garantiza —
> vía el Teorema del Punto Fijo de Banach — que el bucle `FIXPOINT`
> converge a un único punto fijo.

Esta reconstrucción no inventa el problema: se apoya directamente en la
sección 9.4 del paper del proyecto ("Differentiable Banach Theorem"), que
dice explícitamente que se fuerza `||J_T(x)||_2 <= lambda < 1` para
garantizar convergencia global sin oscilaciones. El rol de "experto en
GNN" tampoco es arbitrario: existe una línea de trabajo real y publicada
— las **Implicit Graph Neural Networks** (Gu, Zhu, Cotta et al., NeurIPS
2020) — que usa exactamente esta condición espectral para garantizar que
la ecuación de punto fijo de la red esté "bien planteada" (well-posed).

## 2. La trampa educativa de este enunciado

A primera vista, "calcular el mayor valor singular de una matriz" suena a
ejercicio de álgebra lineal de manual: se llama a `np.linalg.svd` y listo.
**Esa lectura es la trampa.** Si uno estudia el enunciado con más cuidado
aparecen tres capas que la lectura superficial se salta:

1. **No hace falta la SVD completa.** Calcular la SVD de una matriz de
   `n x n` cuesta `O(n^3)`. Si lo único que se necesita es el valor
   singular más grande (no toda la descomposición), existe un método
   mucho más barato — la **iteración de potencias** — que cuesta
   `O(n^2)` por iteración y es la razón real por la que el enunciado pide
   un "kernel ultra-optimizado": es una operación que conviene fusionar
   en GPU precisamente porque se repite muchas veces y es más liviana que
   una SVD.
2. **El número en sí no es el objetivo, es un certificado.** El punto no
   es "saber cuánto vale sigma_max", sino que ese valor decide si el
   Teorema de Punto Fijo de Banach **aplica o no** al bucle `FIXPOINT`.
   Si `sigma_max < 1`, hay garantía matemática de convergencia a un único
   punto fijo. Si `sigma_max >= 1`, no hay ninguna garantía — puede
   converger igual por suerte, oscilar, o divergir.
3. **La cota tiene que ser global, no puntual.** Acotar el Jacobiano en
   un solo punto `x` no alcanza: Banach exige que la contracción valga
   para *todo* el dominio. Por eso en este trabajo se usa la propiedad de
   que `|tanh'(z)| <= 1` para todo `z`, lo que permite subir la cota
   puntual del Jacobiano a una cota *global* válida en todo el espacio
   usando solamente `sigma_max(W)` — sin tener que evaluar el Jacobiano
   en cada punto posible.

Quien se quede en "es solo calcular un número con álgebra lineal" no va a
entender por qué la industria (spectral normalization en GANs, IGNN en
GNNs, control de Lipschitz en redes de equilibrio profundo) invirtió tanto
esfuerzo en algo que "ya se sabía calcular". El desarrollo completo de
abajo muestra por qué ese número es el punto de apoyo de toda la garantía.

## 3. Modelo matemático estándar

**Marco teórico:** Teorema del Punto Fijo de Banach (contracción).

Sea $(X, d)$ un espacio métrico completo (aquí $X = \mathbb{R}^n$ con la
norma euclídea) y $T: X \to X$. Si existe $\lambda \in [0, 1)$ tal que

$$d(T(x), T(y)) \le \lambda \, d(x, y) \quad \forall x, y \in X$$

entonces $T$ tiene un **único** punto fijo $x^*$, y la iteración
$x_{k+1} = T(x_k)$ converge a $x^*$ desde **cualquier** punto inicial, a
tasa geométrica $d(x_k, x^*) \le \lambda^k \, d(x_0, x^*)$.

**Operador concreto (capa tipo GNN implícita / equilibrium layer):**

$$T(x) = \tanh(W x + b), \qquad W \in \mathbb{R}^{n \times n}, \; b \in \mathbb{R}^n$$

**Condición de contracción vía valor singular máximo:**

$$
\|J_T(x)\|_2 = \big\| \mathrm{diag}\big(1 - \tanh^2(Wx+b)\big) \, W \big\|_2
\le \underbrace{\|\mathrm{diag}(1-\tanh^2(\cdot))\|_2}_{\le\, 1} \cdot \|W\|_2
= \sigma_{\max}(W)
$$

La desigualdad usa que $|\tanh'(z)| = 1 - \tanh^2(z) \le 1$ para todo
$z \in \mathbb{R}$. Como la cota vale para **cualquier** $x$, si se impone

$$\sigma_{\max}(W) = \lambda < 1$$

entonces $T$ es una contracción global y Banach garantiza convergencia
única. Esta es exactamente la condición de "well-posedness" usada en
Implicit Graph Neural Networks (Gu et al., NeurIPS 2020) y es la misma
idea detrás de Spectral Normalization (Miyato et al., ICLR 2018), aplicada
ahí a discriminadores de GANs en lugar de a bucles de punto fijo.

**Cómo obtener $\sigma_{\max}(W)$ sin SVD completa — iteración de
potencias sobre $W^\top W$:**

$$v_{k+1} = \frac{W^\top W v_k}{\|W^\top W v_k\|}, \qquad
\sigma_{\max}(W) \approx \|W v_k\| \; \text{cuando converge}$$

Esto converge geométricamente a la dirección singular dominante siempre
que el segundo valor singular sea estrictamente menor que el primero, y
es el algoritmo que efectivamente se implementa (y se corrió) en
`spectral_fixpoint.py`.

## 4. Razonamiento paso a paso (por qué este camino y no otro)

1. **Por qué no SVD completa como "solución final":** funciona, pero es
   `O(n^3)` y descarta toda la información salvo un número. En una capa
   que se re-normaliza en cada paso de entrenamiento (como en spectral
   normalization), ese costo es prohibitivo a escala. La iteración de
   potencias es la alternativa estándar en la literatura precisamente por
   esto.
2. **Por qué no calcular el Jacobiano completo en cada `x`:** el
   Jacobiano de una capa de `n` unidades es una matriz `n x n` que cambia
   en cada punto. Evaluarlo y acotarlo puntualmente en cada iteración del
   bucle `FIXPOINT` sería, de nuevo, mucho más caro que acotarlo una sola
   vez de forma global usando la propiedad `|tanh'| <= 1`.
3. **Por qué elegir $\tanh$ y no otra no linealidad:** cualquier función
   con derivada acotada por 1 en todo su dominio (tanh, sigmoide escalada,
   ReLU con clipping) permite el mismo argumento. Se descartó usar una
   no linealidad sin cota de derivada (por ejemplo, una exponencial pura)
   porque ahí la cota global dejaría de existir y habría que volver a
   acotar puntualmente, perdiendo la ventaja práctica del método.
4. **Por qué verificar con múltiples condiciones iniciales:** Banach no
   solo garantiza convergencia, garantiza **unicidad** del punto fijo. La
   forma correcta de comprobar esto empíricamente no es correr la
   iteración una sola vez, sino lanzarla desde varios puntos iniciales
   bien separados y verificar que **todas** las trayectorias terminan en
   el mismo punto. Por eso `run_fixpoint_experiment` en el script usa 4
   inicializaciones aleatorias y mide la distancia final entre ellas.
5. **Por qué agregar también un caso "expansivo" (sigma_max = 2.5):** para
   que la comprobación empírica no se quede solo en "el caso bueno
   funciona", sino que muestre el contraste: sin la cota espectral, la
   garantía de unicidad se rompe (las trayectorias no convergen al mismo
   punto).

## 5. Desarrollo, código y resultado de la corrida interna

El código completo y funcional está en **`spectral_fixpoint.py`** (NumPy
puro, sin dependencias exóticas). Se hizo una corrida interna de
sanity-check (no es la comprobación final del alumno) y estos fueron los
resultados obtenidos:

```
=== Paso A: validacion del estimador (iteracion de potencias vs SVD) ===
sigma_max estimado (power iteration): 5.105360
sigma_max exacto (SVD, LAPACK)       : 5.105360
error relativo                       : 1.57e-10
OK: el estimador reproduce el valor exacto dentro de tolerancia.

--- Experimento: W SIN normalizar (sin garantia de Banach) ---
sigma_max(W) real (SVD)      : 5.1054
||x_n - x_(n-1)|| (ultimo)   : 3.805530e+00   (no decrece a 0)
distancia final entre las 4 trayectorias: 4.646529e+00   (NO convergen al mismo punto)

--- Experimento: W normalizada espectralmente (lambda_target = 0.9) ---
sigma_max(W) real (SVD)      : 0.9000
||x_n - x_(n-1)|| (ultimo)   : 2.301371e-17   (decrece a ~0, geometricamente)
distancia final entre las 4 trayectorias: 2.775558e-17   (SI convergen al mismo punto)

--- Experimento: W expansiva (sigma_max = 2.5, viola Banach) ---
sigma_max(W) real (SVD)      : 2.5000
||x_n - x_(n-1)|| (ultimo)   : 1.451870e+00   (no decrece a 0)
distancia final entre las 4 trayectorias: 2.252749e+00   (NO convergen al mismo punto)
```

**Lectura de estos números:**

- El estimador de iteración de potencias reproduce el valor exacto de la
  SVD con error relativo de `1.57e-10`: confirma que el método barato es
  numéricamente confiable como sustituto de la SVD completa.
- Con $\sigma_{\max}(W) = 0.9 < 1$: las cuatro trayectorias, arrancando
  desde puntos separados por normas de hasta ~11 unidades, terminan
  indistinguibles entre sí (distancia final `~10^{-17}`, es decir, cero a
  precisión de punto flotante). Esto **es** la unicidad de Banach
  observada empíricamente.
- Con $\sigma_{\max}(W) = 5.1$ o $2.5$ (ambos $\ge 1$, violando la
  hipótesis): las trayectorias no colapsan a un punto común (distancias
  finales de orden 1, no de orden $10^{-17}$), y el paso final no decrece
  hacia cero. Esto no es una falla del código: es exactamente lo que
  predice la teoría cuando la hipótesis de contracción no se cumple — no
  hay garantía, y en este caso concreto efectivamente no converge.

No hubo ningún caso de no-convergencia inesperada ni de resultado
"inventado": todos los números de arriba salen de esta corrida real.

## 6. Sobre el kernel Triton pedido en el enunciado original

El enunciado original pide explícitamente un "kernel en Triton". Esto
**no se ejecutó**: este sandbox no tiene GPU NVIDIA ni el paquete
`triton` instalado (requiere CUDA). Se entrega en
**`triton_kernel_reference.py`** un esqueleto de referencia que muestra
cómo se llevaría el paso de iteración de potencias (mat-vec + mat-vec +
normalización) a un kernel fusionado, y se explica ahí mismo por qué
fusionar esos pasos importa (evitar materializar resultados intermedios
en memoria global — el cuello de botella real en operaciones de este
tamaño es de ancho de banda de memoria, no de cómputo). Ese archivo está
marcado como no ejecutado de forma explícita en su propio docstring; la
comprobación matemática real de la tarea está enteramente en
`spectral_fixpoint.py`, que sí es funcional y sí corrió.

## 7. Conclusión técnica

Queda demostrado empíricamente (no solo argumentado en el papel) que:

1. El mayor valor singular de $W$ se puede estimar de forma barata y
   precisa mediante iteración de potencias, sin necesidad de SVD
   completa.
2. Esa cantidad es exactamente la que decide, vía el Teorema de Punto
   Fijo de Banach, si el bucle `FIXPOINT` de un operador tipo GNN
   implícita tiene garantía de convergencia a un único punto fijo.
3. La garantía se rompe de forma observable (no solo teórica) cuando no
   se respeta la cota, lo cual valida que la condición
   $\sigma_{\max}(W) < 1$ no es un capricho técnico sino una hipótesis
   necesaria del teorema que se está invocando.

## 8. Instrucciones para que el alumno corra el código él mismo

```bash
# Requisito: Python 3.9+ y numpy
pip install numpy

# Ejecutar la comprobación completa (paso A a D del script)
python spectral_fixpoint.py
```

No hace falta ninguna GPU ni instalar `triton` para reproducir la
comprobación matemática central de esta tarea. El archivo
`triton_kernel_reference.py` es material adicional/opcional: solo se
puede correr si se dispone de una GPU NVIDIA con el paquete `triton`
instalado; si no se tiene esa GPU, el propio script lo indica con un
mensaje claro en lugar de fallar de forma confusa.

Se recomienda, como ejercicio adicional, cambiar `lambda_target` en el
script (por ejemplo a `0.99` o a `1.01`) y observar en qué punto exacto
se pierde la convergencia a un único punto fijo — eso conecta
directamente con la discusión de la sección 2 sobre por qué la cota debe
ser estrictamente menor a 1.

---

```json
{
  "alumno": "ai.mpat.mimi02@gmail.com",
  "tarea": 39,
  "tipo_problema": "Teoria de punto fijo (Banach) aplicada a convergencia de operadores tipo GNN implicita via cota espectral del Jacobiano",
  "estado": "resuelto",
  "resultado_resumen": "Se valido empiricamente que sigma_max(W) < 1 (estimado por iteracion de potencias) garantiza convergencia unica del bucle FIXPOINT; sin esa cota la unicidad se rompe.",
  "metrica_estandar_usada": "Valor singular maximo (norma espectral) del Jacobiano; distancia euclidiana entre trayectorias e iteraciones sucesivas"
}
```

que has usado el formato de razonamiento adaptado por AGT
