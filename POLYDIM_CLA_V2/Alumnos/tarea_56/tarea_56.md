# Tarea 56 — Escalado asintótico de FLOPs y VRAM en una cadena de operaciones (Dynamo)

## 1. Enunciado

### 1.1 Original (banco maestro, Escuadrón 3, Test 56)

> "Actúa como ingeniero de PyTorch Dynamo. Diseña una operación encadenada que demuestre a nivel de volcado intermedio de grafo que el consumo de VRAM y FLOPs gastados durante la inferencia asintótica."

### 1.2 Nota de reconstrucción

El fragmento original está gramaticalmente roto: la cláusula final ("que el consumo de VRAM y FLOPs gastados durante la inferencia asintótica") es un sintagma nominal sin verbo — no dice **qué** hay que demostrar sobre ese consumo. Esto es consistente con el origen del banco (combinación de plantillas rol + fragmento técnico) señalado en la skill de este curso.

A diferencia de otros enunciados del mismo banco (que asumen una conclusión, del tipo "demostrá por qué X ahorra cómputo sin perjudicar Y"), este fragmento **no** contiene una afirmación a demostrar como cierta — es solo un sustantivo suelto ("el consumo de VRAM y FLOPs... asintótica"). Por lo tanto la reconstrucción no tiene que neutralizar una premisa no probada, sino completar la pregunta de investigación:

**Versión reformulada:** Diseñar una cadena de operaciones (bloques de self-attention, que son el caso estándar con comportamiento asintótico no trivial) y, usando el volcado del grafo FX intermedio que genera TorchDynamo al trazar el modelo, medir empíricamente cómo escalan el consumo de VRAM/memoria y los FLOPs a medida que crece el tamaño de la entrada (largo de secuencia *n*), comparando el exponente de escalado empírico contra el orden teórico de las operaciones involucradas.

Esta es una pregunta abierta: el desarrollo no da por sentado ningún resultado, lo mide.

## 2. Modelo matemático

Cadena de $L$ bloques idénticos de atención (proyección QKV + atención + proyección de salida + MLP con expansión ×4), con largo de secuencia $n$, dimensión de embedding $d$, y $h$ cabezas.

**FLOPs por bloque** (orden dominante, ignorando softmax y normalizaciones que son $O(n)$/$O(nd)$):

$$
\text{FLOPs}_{\text{bloque}}(n, d) = \underbrace{4nd^2}_{\text{QKV + salida}} + \underbrace{2n^2d}_{\text{scores} + \text{contexto}} + \underbrace{8nd^2}_{\text{MLP}} = 12nd^2 + 2n^2d
$$

$$
\text{FLOPs}(n) = L \cdot (12nd^2 + 2n^2d)
$$

Para $n \gg d$, el término $2n^2d$ domina, por lo que:

$$
\text{FLOPs}(n) = \Theta(n^2) \quad \text{cuando } n \to \infty \text{ con } d \text{ fijo}
$$

**Memoria (activaciones + matriz de atención):** la matriz de scores por cabeza ocupa $O(n^2)$ elementos, mientras que las activaciones intermedias ocupan $O(nd)$. Para $n \gg d$:

$$
\text{Memoria}(n) = \Theta(n^2)
$$

**Hipótesis a contrastar empíricamente:** el exponente empírico de la pendiente log-log de FLOPs medidos vs. $n$, y de memoria pico vs. $n$, debería aproximarse a $2$ a medida que $n$ crece por encima de $d$.

## 3. Razonamiento

- Se eligió **self-attention** como operación de la cadena porque es el ejemplo canónico con comportamiento asintótico cuadrático bien documentado (Vaswani et al., 2017), y porque permite contrastar teoría vs. medición de forma clara.
- Para el "volcado del grafo intermedio" se usa `torch._dynamo.explain`, que expone directamente el grafo FX que Dynamo produce al trazar el modelo (incluyendo graph breaks), en lugar de TorchScript (`torch.jit.trace`), que no refleja cómo opera específicamente Dynamo — el enunciado pide explícitamente "a nivel de volcado intermedio de grafo" de Dynamo, no de otro backend.
- Para contar FLOPs se usa `torch.utils.flop_counter.FlopCounterMode`, nativo de PyTorch (≥2.1) y basado en hooks sobre los kernels de ATen, en vez de estimarlos manualmente o con librerías externas (fvcore), para minimizar dependencias y errores de cómputo manual.
- Para memoria se usa `torch.cuda.max_memory_allocated()` cuando hay GPU. **Se documenta explícitamente la limitación:** en un entorno sin GPU (como el usado para preparar este material), se cae a `tracemalloc`, que mide picos de memoria de Python/CPU, no el allocator de CUDA. El orden de magnitud relativo (cómo escala con $n$) debería mantenerse, pero los valores absolutos no son comparables a VRAM real — el alumno debe correr esto en su propia GPU si quiere la medición real de VRAM.

## 4. Desarrollo y código

El código completo está en `tarea_56.py` (mismo directorio). Estructura:

1. `AttentionBlock`: un bloque estándar de self-attention + MLP.
2. `ChainedModel`: encadena `n_layers` bloques — esta es la "operación encadenada" pedida.
3. `theoretical_flops(n, d, L)`: implementa la fórmula de la Sección 2.
4. `dump_intermediate_graph(model, x)`: llama a `torch._dynamo.explain` y **imprime el grafo FX intermedio** (las primeras líneas), cumpliendo el requisito de "volcado a nivel de grafo".
5. `measure_flops` / `measure_peak_memory`: miden FLOPs y memoria pico reales para distintos $n$.
6. `run_experiment()`: barre $n \in \{128, 256, 512, 1024, 2048\}$, imprime FLOPs medidos vs. teóricos y memoria pico, y ajusta una recta en escala log-log (`np.polyfit`) para obtener el exponente empírico de escalado de FLOPs y de memoria respecto de $n$.

**No se ejecutó el experimento completo como parte de esta entrega** (solo se verificó que el archivo compila sin errores de sintaxis con `python3 -m py_compile`, sin importar `torch` — no se validaron los valores numéricos). La corrida real, con los resultados y el ajuste log-log, la debe hacer el alumno en su propia cuenta.

## 5. Conclusión

El modelo teórico predice un exponente de escalado $\approx 2$ tanto para FLOPs como para memoria, cuando $n$ crece por encima de $d$ (en este script, $d=256$, así que conviene mirar sobre todo los puntos con $n \geq 1024$ para que la aproximación asintótica sea razonable). 

**Esto todavía no está comprobado — es una predicción, no un resultado.** Al correr `tarea_56.py`, el alumno va a obtener los exponentes empíricos reales y podrá comparar:

- Si el exponente medido de FLOPs se acerca a 2 → el modelo teórico de la Sección 2 queda respaldado empíricamente para este rango de $n$.
- Si se aleja de 2 (por ejemplo, más cercano a 1) → probablemente el rango de $n$ elegido todavía no es suficientemente grande frente a $d$ para que domine el término cuadrático, o hay efectos de overhead constante que conviene reportar y discutir.
- Si el exponente de memoria difiere del de FLOPs → puede deberse a que el allocator (CUDA o el approximate de `tracemalloc`) no libera/reutiliza memoria de forma uniforme, algo que vale la pena inspeccionar con el volcado del grafo (¿cuántos graph breaks hay? ¿se fusionan los kernels?).

El alumno debe interpretar sus propios números — no está anticipado ni fabricado ningún resultado acá.

## 6. Instrucciones para ejecutar (a cargo del alumno)

```bash
pip install torch numpy   # requiere torch >= 2.1 (idealmente 2.2+)
python3 tarea_56.py
```

- Si tenés GPU con CUDA disponible, el script la va a usar automáticamente y vas a obtener la medición real de VRAM vía `torch.cuda.max_memory_allocated()`.
- Si corrés en CPU, vas a ver la advertencia implícita del fallback a `tracemalloc` — interpretá esos números como una aproximación de orden de magnitud, no como VRAM real.
- Prestá atención a la salida de `dump_intermediate_graph`: ahí está el volcado del grafo FX intermedio que pide el enunciado. Si querés ver el grafo completo (no solo las primeras 10 líneas), editá el script para imprimir todas las líneas de `str(g)`.
