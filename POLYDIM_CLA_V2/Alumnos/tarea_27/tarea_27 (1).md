# Tarea 27 — POLYDIM (Escuadrón 1: Relajación Topológica y Homotopía / TDA)

## 1. Enunciado

**Original (banco maestro, Test 27), citado literalmente:**
> "Actúa como ingeniero de Teoría de la Información. Simula en Python un entorno de agentes que modifican asíncronamente un estado tensor central para el consumo de VRAM y FLOPs gastados durante la inferencia asintótica."

### Nota de reconstrucción (obligatoria — ver metodología del curso)

El enunciado, como está escrito, es agramatical: la cláusula final ("para el consumo de VRAM y FLOPs...") no completa el verbo "modifican...para", como si el banco hubiera pegado dos fragmentos de plantillas distintas:

- **Fragmento A (tema):** *"un entorno de agentes que modifican asíncronamente un estado tensor central"*
- **Fragmento B (métrica a medir):** *"el consumo de VRAM y FLOPs gastados durante la inferencia asintótica"*

Reconstruí el enunciado uniendo ambos fragmentos como **tema + métrica de evaluación**, en vez de forzar una relación causal que el texto no expresa:

> **Enunciado reconstruido:** Actuando como ingeniero de Teoría de la Información/Sistemas Distribuidos, simular en Python un entorno de **N agentes que modifican asíncronamente y sin bloqueos (lock-free) un tensor central compartido** (el "estado latente" de POLYDIM), y **medir empíricamente cómo escalan el consumo de memoria (proxy de VRAM) y los FLOPs** en función de N, a medida que el sistema se acerca a su régimen asintótico de inferencia.

No hay ninguna conclusión dada por sentada en el enunciado reconstruido (a diferencia de otros tests del banco tipo "demostrá por qué X ahorra cómputo sin perjudicar Y"), así que no aplica la salvedad del paso 4 sobre premisas no demostradas — el enunciado reconstruido es una pregunta abierta y empírica.

## 2. Modelo matemático estándar

El "entorno de agentes que modifican asíncronamente un estado tensor central sin lock" es exactamente el escenario formalizado por **Hogwild!** (Recht, Ré, Wright, Niu, *NeurIPS 2011*, "Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent"):

- Tensor central $\theta_t \in \mathbb{R}^d$ (el estado latente compartido).
- $N$ agentes concurrentes; cada uno, en su paso local, muestrea un subconjunto disperso de coordenadas activas $e \subset \{1,\dots,d\}$, $|e| = s \ll d$, y aplica:
$$\theta_{t+1}[e] \leftarrow \theta_t[e] - \eta \, g_e\big(\theta_{t-\tau}\big)$$
  donde $\tau$ es la **obsolescencia (staleness)**: como no hay lock, el agente puede estar leyendo una versión de $\theta$ ya modificada parcialmente por otros agentes desde entonces.
- **Teorema (Hogwild!, informal):** si $f$ es suave y fuertemente convexa, si las actualizaciones son dispersas (probabilidad de solapamiento de coordenadas acotada por $\rho$) y la obsolescencia $\tau$ está acotada, entonces para un paso $\eta$ suficientemente pequeño el esquema converge al mismo orden asintótico $O(1/t)$ que el SGD secuencial, **sin necesidad de locks**.
- **Condición crítica:** esta garantía depende de la **dispersión** ($s \ll d$). Si las actualizaciones fueran densas (cada agente toca todas las coordenadas), las escrituras concurrentes pueden perderse (race condition real) y el teorema no aplica — esto es justamente lo que el código de la Sección 4 permite investigar variando $s$.

**Modelo de costo (FLOPs y memoria/VRAM):**
- FLOPs por actualización: $\approx 4s$ (gather de $s$ coordenadas, resta de gradiente sintético, escalado por $\eta$, scatter de vuelta). FLOPs totales tras $T$ actualizaciones: $O(T \cdot s)$, **independiente de $d$ y de $N$** (solo depende de cuántas actualizaciones totales se hacen).
- Memoria (proxy de VRAM), **dos implementaciones a comparar**:
  1. **Tensor verdaderamente compartido** (`multiprocessing.shared_memory`): un único buffer de $d$ floats en memoria física, sin importar cuántos agentes (procesos) lo referencien. Predicción: memoria $= O(d)$, constante en $N$.
  2. **Copia ingenua por proceso** (patrón común y a menudo no advertido al paralelizar con `multiprocessing`/`fork` sin memoria compartida explícita): cada proceso agente recibe *su propia copia* del tensor. Predicción: memoria total $= O(N \cdot d)$, crece linealmente con el número de agentes.

Esta comparación es la pregunta empírica central de la tarea: **¿el "estado tensor central" se comporta como memoria compartida real (VRAM constante) o como una duplicación oculta (VRAM que escala con N)?**

## 3. Razonamiento

- Se descartó modelar esto como un simple *parameter server* con locks (mutex/semáforos) porque el enunciado original pide explícitamente "modifican asíncronamente" — la esencia de Hogwild! es precisamente evitar el lock, aceptando staleness acotada a cambio de paralelismo real.
- Se usó `multiprocessing` en lugar de `threading` porque en CPython el GIL serializa los hilos de Python puro; para simular escritura concurrente real sobre un tensor compartido (y así medir el efecto de VRAM/memoria de forma honesta) se necesitan procesos con memoria compartida explícita (`shared_memory`), que es además el mecanismo real más cercano a cómo un acelerador (GPU) comparte un buffer de tensor entre kernels concurrentes.
- Se midieron los FLOPs de forma **analítica** (contando operaciones por diseño del algoritmo), no vía profiler de hardware, porque no hay GPU en este entorno; esto es consistente con el "Smoking Gun Prototype" mencionado en el paper (validación en una VM mínima, no en silicio de producción).
- La memoria se mide en bytes reales del buffer (`shm.size` para el caso compartido, suma de `nbytes` reportados por cada proceso en el caso naive) — es una medida honesta y verificable, no una estimación teórica.

## 4. Desarrollo y código

Ver `tarea_27.py` (funcional, sin pseudocódigo). Resumen de su estructura:

- `run_shared_experiment(d, n_agents, sparsity_s, n_updates_per_agent)`: crea **un** tensor en `shared_memory`, lanza `n_agents` procesos que lo leen/escriben sin lock, devuelve tiempo de pared, FLOPs totales, bytes de memoria compartida y si el tensor final quedó finito (sin NaN/Inf, indicador indirecto de estabilidad pese a las race conditions).
- `run_naive_experiment(...)`: la misma lógica pero cada proceso recibe su propia copia (`bytearray` local), simulando el patrón de paralelización "naive" — mide la memoria total sumando lo que cada proceso reporta poseer.
- `scaling_study(...)`: barre distintos valores de $N$ (número de agentes) e imprime la tabla comparativa — este es el experimento asintótico que el alumno debe correr y observar.

**Corrida interna de sanity-check** (solo para verificar que no hay errores de sintaxis/runtime obvios, con parámetros diminutos $d=200$, 10 actualizaciones, $N\in\{1,2\}$ — **esto no es la comprobación final**):

```
   N |  shared_mem(B) |   naive_mem(B) |  shared_t(s) | naive_t(s) |      flops | theta_ok
   1 |           1600 |           1600 |       0.0333 |     0.0208 |        200 |     True
   2 |           1600 |           3200 |       0.0083 |     0.0173 |        400 |     True
```

Incluso a esta escala minúscula ya se ve el patrón esperado: `shared_mem` se mantiene constante (1600 B = 200×8 bytes) mientras `naive_mem` duplica al pasar de 1 a 2 agentes. **Esto no reemplaza la corrida real que el alumno debe hacer con parámetros representativos** (ver instrucciones abajo).

## 5. Conclusión (a verificar por el alumno, no dada por sentada aquí)

El modelo predice:
- **Memoria:** `shared_mem_bytes` constante en $N$ (⁠$O(d)$); `naive_mem_bytes` creciendo linealmente en $N$ (⁠$O(N \cdot d)$). El alumno debe correr `scaling_study` con $N$ más grande (por ejemplo `agent_counts=(1,2,4,8,16)`) y $d$ representativo (ej. $d=10^5$–$10^6$) para confirmar la pendiente lineal en el caso naive vs. la línea plana en el caso compartido.
- **FLOPs:** deberían escalar linealmente con el número total de actualizaciones ($N \times$ `n_updates_per_agent`), independientemente de si la memoria es compartida o no — es un chequeo de consistencia interna del propio contador.
- **Estabilidad bajo race conditions:** con dispersión $s \ll d$ (Hogwild!), `theta_has_nan_or_inf` debería mantenerse `False` incluso con muchos agentes escribiendo sin lock. Si el alumno sube $s$ hasta acercarse a $d$ (actualizaciones densas), el teorema de Hogwild! deja de aplicar — vale la pena que lo pruebe y observe si aparece inestabilidad o pérdida sistemática de actualizaciones (comparar `theta_norm_final` contra una corrida secuencial de referencia sin concurrencia, que no está incluida aquí y queda como extensión).
- El **tiempo de pared** (`wall_time_s`) no tiene una predicción teórica limpia en este experimento (depende de overhead de `fork`/`spawn` de procesos, mucho más caro que el cómputo real para $d,s$ pequeños) — el alumno debe interpretar con cautela: para conclusiones sobre "inferencia asintótica" real, el número de actualizaciones por agente debe ser mucho mayor que el overhead de arranque de procesos.

## 6. Instrucciones para correr

```bash
cd tarea_27
python3 tarea_27.py
```

Sin dependencias externas (solo `numpy`, incluido en la mayoría de entornos científicos; si falta: `pip install numpy`). Requiere Python ≥ 3.8 por `multiprocessing.shared_memory`.

Para el experimento asintótico completo, editar en `tarea_27.py` (función `scaling_study`, al final del archivo) los parámetros `d`, `sparsity_s`, `n_updates_per_agent` y `agent_counts` a valores más representativos, y graficar `naive_mem_bytes` vs. `shared_mem_bytes` en función de `N` para confirmar (o refutar) la hipótesis $O(N \cdot d)$ vs. $O(d)$.

```json
{
  "alumno": null,
  "tarea": 27,
  "tipo_problema": "Concurrencia asíncrona sin lock sobre tensor compartido (Hogwild!) + análisis de costo (FLOPs/memoria) asintótico",
  "estado": "resuelto",
  "resultado_resumen": "Se reconstruyó el enunciado como estudio de escalado de memoria (VRAM proxy) y FLOPs en un esquema Hogwild!-like de agentes asíncronos sin lock sobre un tensor central, comparando memoria compartida real (O(d)) vs. copia ingenua por proceso (O(N·d)).",
  "metrica_estandar_usada": "FLOPs analíticos (conteo de operaciones), bytes de memoria (RSS/tamaño de buffer), tiempo de pared"
}
```
