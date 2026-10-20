---
worker_id: ALUMNO (ai.mpat.mimi02@gmail.com)
ticket_id: 6
escuadron: "Relajación Topológica y Homotopía (TDA)"
fecha: 2026-07-03
status: SUCCESS
---

# Tarea 6 — Cotas de intervalo para activaciones (dominio poliédrico)

## 0. Nota honesta antes de empezar

El documento maestro pide actuar como "matemático numérico" y diseñar "un kernel en
Triton ultra-optimizado para realizar los intervalos (bounds) numéricos de las
activaciones emulando dominios poliédricos", envuelto en un marco ("POLYDIM") que
mezcla topos de Grothendieck, álgebra de Clifford, etc. Esos términos son reales,
pero la forma en que el documento los combina no corresponde a ninguna arquitectura
de IA validada en la literatura — es lenguaje que *suena* riguroso sin serlo.

Lo que sí es una pregunta real, con más de 10 años de literatura seria detrás
(verificación formal de redes neuronales), es esto:

> **Dada una red y un rango de valores posibles de entrada, ¿qué rango de valores
> puede tomar la salida — sin tener que probar todas las entradas posibles?**

Eso es exactamente lo que vamos a resolver: **Interval Bound Propagation (IBP)**,
la versión más simple y robusta de un "dominio poliédrico" (el dominio de caja/box).

## 1. La idea con un ejemplo cotidiano

Imaginá que sos dueño de un local y tu ganancia del día es:

$$ y = 3x_1 - 2x_2 + 10 $$

donde $x_1$ = clientes que entran (entre 50 y 80) y $x_2$ = devoluciones (entre 5 y 15).

No sabés los valores exactos de $x_1$ y $x_2$ hoy, solo el **rango** en que van a
caer. Pero igual querés saber: *¿cuál es la ganancia mínima y máxima posible hoy?*

Razonás así, sin necesitar probar las 30×10 combinaciones:
- Para maximizar $y$: querés el $x_1$ más alto posible (coeficiente +3, así que
  más $x_1$ = más ganancia) y el $x_2$ más **bajo** posible (coeficiente -2, así
  que menos $x_2$ = menos resta).
  $y_{max} = 3(80) - 2(5) + 10 = 240 - 10 + 10 = 240$
- Para minimizar $y$: al revés — $x_1$ mínimo, $x_2$ máximo.
  $y_{min} = 3(50) - 2(15) + 10 = 150 - 30 + 10 = 130$

Esa regla — "si el coeficiente es positivo usá el extremo superior de la variable
para el máximo y el inferior para el mínimo; si es negativo, al revés" — es
**exactamente** el algoritmo que implementa una capa lineal de una red neuronal.
No hace falta álgebra de Clifford para esto: es álgebra de secundario aplicada
con disciplina. Lo único "nuevo" es hacerlo para miles de neuronas en paralelo
en una GPU, ahí es donde entra Triton.

## 2. Formalización (esto sí generaliza al caso de la red)

Sea una capa afín $y = Wx + b$, con $x \in [l, u]$ (una caja: cada coordenada de
$x$ tiene su propio rango).

Separamos la matriz de pesos en su parte positiva y negativa:

$$ W^+ = \max(W, 0), \qquad W^- = \min(W, 0) $$

y la cota es **exacta** (no aproximada):

$$ y_l = W^+ l + W^- u + b, \qquad y_u = W^+ u + W^- l + b $$

**Por qué es exacta y no una aproximación optimista:** una combinación lineal de
variables independientes, cada una acotada en una caja, alcanza su óptimo siempre
en un vértice de la caja. La fórmula de arriba selecciona el vértice correcto
coordenada por coordenada. Esto es lo que en el paper adjunto (`POLYDIM_PAPER_3_EN.md`)
se llama, de forma florida, "pullback al objeto inicial" — en realidad es
programación lineal de primer año.

Para la no linealidad (ReLU), como es monótona no decreciente:

$$ \text{relu}(l) \le \text{relu}(x) \le \text{relu}(u) $$

también exacta. (La pérdida de precisión aparece recién cuando se **encadenan**
varias capas — ahí el rango se va ensanchando artificialmente porque se pierde
la correlación entre neuronas. Los dominios poliédricos más finos, como zonotopos
o DeepPoly, existen justamente para recuperar parte de esa correlación perdida.
Eso queda fuera del alcance de esta tarea puntual, pero es la dirección natural
de la Tarea 6 si se quisiera "subir de nivel".)

## 3. El código (verificado, no solo escrito)

Ver `ibp.py` adjunto. Contiene:

1. `affine_bounds(W, b, l, u)` — la fórmula de la sección 2, en NumPy.
2. `relu_bounds(l, u)` — propagación por ReLU.
3. Un **kernel Triton** (`TRITON_KERNEL_SKETCH`) que implementa la misma fórmula
   en GPU, un programa por neurona de salida, con reducción vectorizada por
   bloque (`BLOCK_N`). **Nota honesta:** este sandbox no tiene GPU CUDA, así que
   el kernel Triton no se pudo compilar ni ejecutar acá — su lógica es la misma
   fórmula ya verificada en NumPy, pero no debe reportarse como "ejecutado
   empíricamente" porque no lo fue.
4. `verify()` — la parte que sí se ejecutó: genera una red afín aleatoria,
   calcula la cota analítica, y la compara contra 200.000 muestras aleatorias
   dentro de la caja de entrada. **Resultado real de la corrida:**

   ```
   Capa afín -> cota analítica contiene a la empírica: True
   ReLU      -> cota analítica contiene a la empírica: True
   Holgura promedio de la cota: 0.4016
   Status final: SUCCESS
   ```

   Es decir: la cota nunca fue violada por ninguna de las 200.000 muestras (como
   matemáticamente debe ser), y se cuantificó cuánto más ancha es la caja
   analítica respecto del rango real muestreado (la "holgura" — esto es lo
   único honesto que se puede reportar como métrica de ajuste, en vez de
   inventar un "SPS" o "GeoID Stability" sin definición operacional).

## 4. Para seguir avanzando (nivel superior)

Si quisieras de verdad acercarte a un "dominio poliédrico" en el sentido fuerte:
- **Zonotopos**: representar cada neurona como una combinación lineal de "términos
  de error" compartidos entre capas, en vez de un intervalo aislado — reduce la
  holgura que viste arriba (0.40) porque preserva correlaciones.
- **DeepPoly (Singh et al., 2019)**: usa restricciones lineales superior/inferior
  en vez de solo constantes, y hace *back-substitution* a través de las capas.

Estas dos sí son bibliografía real de verificación de redes neuronales, a
diferencia del marco "POLYDIM".

---
*que has usado el formato de razonamiento adaptado por AGT*
