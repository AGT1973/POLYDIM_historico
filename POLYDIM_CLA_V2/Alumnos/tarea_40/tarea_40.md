# Tarea 40 — POLYDIM V2 (Escuadrón 2: Álgebra de Clifford y Rotores)

## 1. Enunciado original (banco maestro)

> "40 (Test 40): Actúa como arquitecto de Sistemas Distribuidos. Diseña una
> operación encadenada que demuestre a nivel de volcado intermedio de grafo
> que un pooling promedio en tiempo seguido de una reconstrucción óptima
> lineal."

## 2. Nota de reconstrucción

Igual que en otras tareas del banco, el enunciado combina dos fragmentos de
plantilla sin verbo que los conecte: "...que demuestre... que [fragmento
técnico]" y el fragmento técnico ("un pooling promedio en tiempo seguido de
una reconstrucción óptima lineal") **nombra una operación pero no dice qué
propiedad hay que demostrar sobre ella**. Tal cual está escrito, no es una
tarea investigable — falta la afirmación a probar.

Reconstrucción investigable, conservando el rol (arquitecto de Sistemas
Distribuidos) y ambos fragmentos técnicos reconocibles ("operación
encadenada", "a nivel de volcado intermedio de grafo", "pooling promedio en
tiempo", "reconstrucción óptima lineal"):

> **Diseña la cadena de operaciones "pooling promedio en tiempo →
> reconstrucción óptima lineal" como un pequeño grafo de cómputo, y demuestra
> — inspeccionando la representación intermedia del grafo (las matrices
> compuestas), sin ejecutar la cadena sobre ningún dato concreto — que la
> composición es una operación estructuralmente coherente (idempotente).
> Después, mide empíricamente cuánta información se pierde realmente en la
> reconstrucción, y cómo depende de las características de la señal de
> entrada.**

Se separan deliberadamente dos preguntas que el enunciado original mezclaba
sin distinguir:
- **(a) Coherencia estructural** de la cadena (verificable en el grafo, sin
  datos) — esto sí puede demostrarse con certeza absoluta.
- **(b) Fidelidad de la reconstrucción** (cuánta información sobrevive) —
  esto depende de los datos y **no puede demostrarse en abstracto**; solo
  medirse empíricamente. Tratar ambas preguntas como si tuvieran la misma
  respuesta ("funciona" a secas) sería la premisa no fundamentada que había
  que evitar dar por sentada.

## 3. Modelo matemático estándar

Sea x ∈ R^n una señal temporal dividida en m = n/k bloques
contiguos de tamaño k (el "pooling" agrupa k muestras consecutivas).

**Pooling promedio (operador lineal P):** P ∈ R^(m×n),
con cada fila i teniendo 1/k en las posiciones del bloque i y 0 en el
resto. Entonces y = Px ∈ R^m son los promedios por bloque.

**Reconstrucción óptima lineal:** entre todas las matrices R que intentan
recuperar x a partir de y=Px, la solución de mínimos cuadrados / norma
mínima es la **pseudoinversa de Moore–Penrose**:
R = P^+ = P^T (P P^T)^(-1)
(aplicable porque P tiene rango fila completo m). Se puede demostrar que,
para esta P en particular, P^+ = k P^T.

**Composición ("operación encadenada"):** M = R P = P^+ P ∈ R^(n×n).

**Propiedad estructural a demostrar (a nivel de grafo, sin datos):** M es
un **proyector ortogonal idempotente**, M M = M, sobre el subespacio de
R^n de las señales "constantes por bloque". Esto se sigue de la
identidad algebraica estándar P^+ P P^+ P = P^+ P (propiedad de la
pseudoinversa de Moore–Penrose) y **no depende de ningún dato de entrada**:
es una propiedad de las matrices P y R solas, verificable directamente
sobre su representación en el grafo de cómputo (análogo a la
**Estrictificación de Mac Lane en tiempo de compilación** del Apéndice V2.0
del paper POLYDIM: la coherencia de la composición se certifica
estáticamente, sin necesidad de ejecutar la cadena con datos concretos).

**Propiedad empírica a medir (sí depende de los datos):** el error de
reconstrucción real,
ε(x) = ||x - Mx||_2 = ||(I-M)x||_2,
que es exactamente la energía de la componente de x que varía *dentro* de
cada bloque (la parte que el promedio destruye). Analíticamente,
ε(x)^2 = Σ_bloques Σ_(i∈bloque) (x_i - x̄_bloque)^2,
donde x̄_bloque es el promedio del bloque. Esta cantidad depende de la
"rugosidad" (variación de alta frecuencia) de x relativa al tamaño de
bloque k, no de la estructura de P o R.

## 4. Razonamiento

- Se separa la propiedad **categórica/estructural** (idempotencia de la
  composición, verificable en la representación intermedia del grafo, sin
  correr datos) de la propiedad **geométrica/empírica** (fidelidad de la
  reconstrucción, que sí depende de los datos). El enunciado original las
  mezclaba implícitamente — separar esto es necesario para que la tarea
  tenga una respuesta bien definida, y evita afirmar como "demostrado" algo
  que en realidad es contingente a los datos.
- Se eligió la pseudoinversa de Moore–Penrose porque es, por definición
  matemática estándar, la reconstrucción lineal óptima en norma mínima /
  mínimos cuadrados — no es una elección arbitraria ni inventada para la
  tarea.
- Se descartó medir el error de reconstrucción de forma abstracta ("en
  general") porque no existe tal cosa: sin restringir la clase de señales
  (o sin una distribución de probabilidad sobre ellas), la pregunta
  "¿cuánta información se pierde?" no tiene una única respuesta numérica.
  Por eso se la trata empíricamente, con tres señales de prueba
  representativas (suave, rugosa, ruido blanco) que cubren el espectro de
  posibles respuestas.
- Alternativa descartada: usar directamente el operador `MIX`/rotor de
  Clifford del paper POLYDIM en vez de una pseudoinversa estándar. Se
  prefirió la formulación lineal estándar porque el pooling promedio ya es,
  por construcción, una transformación lineal (no requiere maquinaria de
  álgebra geométrica), y usar la pseudoinversa permite una verificación
  exacta y cerrada (identidad algebraica) en vez de una aproximación.

## 5. Desarrollo y código

Ver `pooling_reconstruction_test.py` adjunto. Resumen del método:

1. Se construye P (matriz de pooling promedio en bloques de tamaño k).
2. Se calcula R = P^+ con `numpy.linalg.pinv`.
3. **Chequeo de grafo (sin datos):** se calcula M = RP y se verifica
   numéricamente que M M ≈ M (tolerancia 1e-9), y que
   rango(M) = m.
4. **Medición empírica (con datos):** se generan tres señales de prueba
   (suave, rugosa, ruido blanco) del mismo largo n, se aplica M a cada
   una, y se compara el error medido ||x-Mx|| contra la fórmula cerrada de
   energía intra-bloque.

### Resultado de la corrida de sanity-check (interna, no reemplaza la corrida del alumno)

```
Chequeo estructural (grafo, sin datos): max|M@M - M| = 5.551e-17  -> idempotente: True
rango(M) = 10  (esperado = m = 10)
señal suave   : error medido = 1.96139   error teorico (intra-bloque) = 1.96139
señal rugosa  : error medido = 5.32607   error teorico (intra-bloque) = 5.32607
señal ruido   : error medido = 5.94131   error teorico (intra-bloque) = 5.94131
```

La idempotencia estructural se confirma hasta precisión de máquina (la
diferencia 5.5e-17 es error de punto flotante, no un fallo del modelo). El
error de reconstrucción medido coincide exactamente (a la precisión
mostrada) con la fórmula teórica de energía intra-bloque en las tres
señales, y crece con la rugosidad de la señal, tal como predice el modelo.

## 6. Conclusión

**La propiedad estructural sí se cumple exactamente y de forma
incondicional:** la cadena "pooling promedio → reconstrucción óptima lineal"
es una composición coherente e idempotente, verificable en la representación
del grafo de cómputo sin necesidad de ejecutar la cadena sobre ningún dato —
esto es lo único que se puede certificar de forma estática, y en ese sentido
sí es análogo a la Estrictificación de Mac Lane en tiempo de compilación del
paper POLYDIM.

**La fidelidad de la reconstrucción, en cambio, no es una propiedad
estructural sino empírica**, y depende enteramente del contenido de
frecuencias de la señal relativo al tamaño de bloque: para señales suaves la
reconstrucción es casi perfecta, para señales con variación fina dentro de
cada bloque (o ruido) la pérdida es sustancial y no hay forma de evitarla con
una reconstrucción lineal — es información que el pooling promedio destruye
irreversiblemente, sin importar qué tan "óptima" sea la reconstrucción
posterior. Esta distinción es la que el enunciado original no dejaba clara,
y es el resultado central de esta tarea.

## 7. Instrucciones para correr el código vos mismo

```bash
pip install numpy --break-system-packages   # si hace falta
python3 pooling_reconstruction_test.py
```

No hace falta ninguna otra dependencia; corre en menos de un segundo. Probá
cambiar `k` (tamaño de bloque) o agregar tus propias señales de prueba para
ver con tus propios ojos cómo cambia el error de reconstrucción según la
"rugosidad" de la señal relativa al tamaño de bloque.

## 8. Resumen estructurado

```json
{
  "alumno": null,
  "tarea": 40,
  "tipo_problema": "algebra_lineal / analisis_de_señales / coherencia_composicional",
  "estado": "resuelto",
  "resultado_resumen": "Se demuestra estructuralmente (sin datos, a nivel de grafo) que pooling promedio seguido de reconstruccion optima lineal (pseudoinversa) es una composicion idempotente; y se mide empiricamente que la fidelidad de la reconstruccion depende del contenido de frecuencias de la señal, siendo casi perfecta para señales suaves y con perdida sustancial para señales rugosas o ruido.",
  "metrica_estandar_usada": "Norma L2 del error de reconstruccion ||x - Mx||, comparada contra la energia intra-bloque teorica"
}
```
