# Tarea 37 — POLYDIM (Escuadrón 2: Álgebra de Clifford y Rotores / Alineación Espectral)

## 1. Enunciado

**Original (banco maestro, Test 37), citado literalmente:**
> "Actúa como DevOps y SRE de IA. Utiliza la librería POT o Sinkhorn para alinear estadísticamente las distribuciones de embeddings de dos modelos de lenguaje."

### Nota sobre el enunciado

A diferencia de otros tests del banco (p. ej. la Tarea 27), este enunciado **está gramaticalmente completo y bien formado** — no hay fragmentos de plantilla pegados de forma agramatical. No hace falta reconstrucción de texto. Lo único que hice fue precisar matemáticamente qué significa "alinear estadísticamente" (sin asumir una conclusión no demostrada): lo interpreto como **alineación no supervisada de dos espacios de embeddings, sin correspondencia palabra-a-palabra conocida entre ellos**, que es el caso de interés real (si la correspondencia ya fuera conocida, el problema sería trivial — un simple ajuste supervisado).

## 2. Modelo matemático estándar

El problema de "alinear estadísticamente las distribuciones de embeddings de dos modelos de lenguaje" sin supervisión directa es exactamente el que aborda **Wasserstein Procrustes** (Grave, Joulin & Berthet, *AISTATS 2019*, "Unsupervised Alignment of Embeddings with Wasserstein Procrustes"), que combina:

1. **Transporte Óptimo Entrópico (Sinkhorn, Cuturi 2013):** dado un candidato de rotación ortogonal $Q$, se busca el acoplamiento (plan de transporte) $\pi^\star$ entre las nubes de puntos $\{Qx_i\}_{i=1}^n$ (embeddings del modelo A rotados) e $\{y_j\}_{j=1}^m$ (embeddings del modelo B), resolviendo:
$$\pi^\star = \arg\min_{\pi \in \Pi(\mu,\nu)} \sum_{i,j} \pi_{ij} \, C_{ij} - \varepsilon H(\pi)$$
   donde $C_{ij} = \|Qx_i - y_j\|_2^2$, $H(\pi)$ es la entropía de Shannon del acoplamiento, y $\varepsilon > 0$ es el peso de la regularización entrópica. Este problema se resuelve de forma eficiente y diferenciable con el **algoritmo de Sinkhorn** (escalado iterativo de matrices), implementado en la librería `POT` como `ot.sinkhorn`.
2. **Procrustes ortogonal:** dado el acoplamiento (blando) $\pi^\star$, se re-estima la rotación óptima
$$Q^\star = \arg\min_{Q \in O(d)} \sum_{i,j} \pi^\star_{ij} \|Qx_i - y_j\|_2^2 = \arg\min_{Q\in O(d)} \|QX^\top - Y_{\text{soft}}^\top\|_F$$
   cuya solución cerrada es $Q^\star = UV^\top$ donde $X^\top Y_{\text{soft}} = U\Sigma V^\top$ (SVD).
3. Se **alterna** (1) y (2) hasta convergencia (esquema tipo *block coordinate descent*, sin garantía de óptimo global por la no convexidad conjunta, pero empíricamente estable con inicialización razonable).

**Métrica estándar de validación (verificable empíricamente):** precisión de recuperación por vecino más cercano (*nearest-neighbor retrieval accuracy*, top-1 y top-5) contra un conjunto de pares de referencia. Es la métrica canónica usada en la literatura de alineación de espacios de embeddings (word translation / embedding alignment).

## 3. Razonamiento

- Se descartó un ajuste **supervisado** directo (mínimos cuadrados / Procrustes clásico con pares conocidos) porque el enunciado pide alinear *distribuciones*, no puntos correspondientes conocidos — esto apunta al régimen no supervisado, que es matemáticamente más interesante y es exactamente donde el transporte óptimo aporta valor (encuentra la correspondencia y la alineación simultáneamente).
- Se usó `ot.sinkhorn` de la librería **POT** tal como pide el enunciado ("Utiliza la librería POT o Sinkhorn"), en lugar de reimplementar Sinkhorn desde cero, para no duplicar una implementación ya optimizada y numéricamente estable (POT normaliza internamente, maneja underflow, etc.).
- Se generaron embeddings **sintéticos** con una rotación ortogonal oculta $Q_{\text{true}}$ conocida por el generador (pero no por el algoritmo) para poder definir un **ground truth verificable**: con modelos de lenguaje reales no hay forma de saber si la alineación encontrada es "correcta" sin pares de referencia externos; con datos sintéticos sí se puede medir el error exacto de recuperación.
- Se estudió el efecto del parámetro de regularización entrópica $\varepsilon$: valores grandes de $\varepsilon$ dan acoplamientos más "difusos" (Sinkhorn se acerca a la solución de transporte uniforme) y por ende una peor especificidad de la alineación; valores muy pequeños de $\varepsilon$ pueden causar inestabilidad numérica (under/overflow) en el escalado de Sinkhorn. Esto conecta directamente con la mención en el paper POLYDIM (Sección 9, punto 3) de "Entropic Optimal Transport y Descomposición de Hodge" para el protocolo `ALIGN`.

## 4. Desarrollo y código

Ver `tarea_37.py` (funcional, sin pseudocódigo; usa `numpy`, `scipy` y `pot`). Estructura:

- `make_synthetic_embeddings(n, d, noise_std, seed)`: genera embeddings del "modelo A" ($X$) y del "modelo B" ($Y = X Q_{\text{true}} + \text{ruido}$, filas permutadas al azar), junto con el `ground_truth_perm` para evaluación.
- `sinkhorn_coupling(Xq, Y, epsilon)`: llama a `ot.sinkhorn` (POT) con masas uniformes y costo cuadrático normalizado; devuelve el plan de transporte y su costo.
- `procrustes_from_coupling(X, Y, pi)`: resuelve el Procrustes ortogonal vía SVD dado el acoplamiento actual.
- `wasserstein_procrustes(X, Y, epsilon, n_iters, seed)`: alterna Sinkhorn ↔ Procrustes, registrando el historial de costo y residuo por iteración.
- `nearest_neighbor_accuracy(X, Y, Q, ground_truth_perm, k)`: calcula precisión top-1 y top-$k$ de recuperación bajo la rotación $Q$ encontrada.
- `epsilon_study(...)`: barre distintos valores de $\varepsilon$ e imprime la tabla comparativa — este es el experimento que el alumno debe correr y observar.

**Corrida interna de sanity-check** (parámetros diminutos $n=60$, $d=10$, 6 iteraciones, $\varepsilon \in \{0.2, 0.05\}$ — **no es la comprobación final**, solo verifica ausencia de errores de sintaxis/runtime):

```
 epsilon |  sinkhorn_cost_final |   procrustes_res_final |  top1_acc |  top5_acc
   0.200 |             0.230905 |              19.524407 |     0.300 |     0.600
   0.050 |             0.137416 |              14.004651 |     0.200 |     0.483
```

A esta escala tan chica (pocos puntos, dimensión baja, pocas iteraciones) el resultado no es concluyente por sí solo — nótese incluso que aquí `top1_acc` bajó al reducir $\varepsilon$, lo opuesto a lo que predice la teoría en el régimen asintótico. Esto es esperable con $n=60$ y solo 6 iteraciones: la muestra es demasiado chica y el algoritmo no llegó a converger. **El alumno debe correr con parámetros representativos** (ver instrucciones) antes de sacar conclusiones.

## 5. Conclusión (a verificar por el alumno, no dada por sentada aquí)

Lo que el modelo predice, y que el alumno debe confirmar corriendo `epsilon_study` con parámetros más grandes (mayor $n$, $d$, más iteraciones):

- **Efecto de $\varepsilon$:** se espera que exista un valor intermedio de $\varepsilon$ que maximice `top1_acc`/`top5_acc` — ni demasiado grande (acoplamiento difuso, pierde especificidad) ni demasiado pequeño (inestabilidad numérica de Sinkhorn, y/o mínimos locales pobres en la alternancia con Procrustes). Esta es una hipótesis a confirmar empíricamente, no un hecho garantizado por la teoría de Wasserstein Procrustes (el paper original de Grave et al. no da una fórmula cerrada para el $\varepsilon$ óptimo).
- **Convergencia:** `sinkhorn_cost` y `procrustes_residual` deberían decrecer monótonamente (o al menos no-crecer sistemáticamente) a lo largo de las iteraciones cuando el algoritmo converge; si no lo hacen (oscilan o divergen), es indicio de $\varepsilon$ mal elegido o de mala inicialización de $Q$ — el alumno puede probar varias semillas (`seed`) para ver la sensibilidad a la inicialización, dado que el problema conjunto no es convexo.
- **Límite del método:** con ruido (`noise_std`) creciente entre los dos "modelos", la precisión de recuperación debería degradarse — esto no está demostrado en este documento, es una predicción a verificar corriendo el script con distintos `noise_std`.
- Si el alumno observa que el algoritmo **no converge** o da `top1_acc` cercano al azar (~$1/n$) incluso con parámetros grandes, eso también es un resultado válido a documentar: indicaría que la hipótesis de una rotación ortogonal simple no captura la relación real entre los dos espacios de embeddings (que podría requerir, por ejemplo, Gromov-Wasserstein en vez de OT clásico, si los espacios no son ni siquiera comparables bajo una isometría).

## 6. Instrucciones para correr

```bash
cd tarea_37
pip install pot   # si no está instalado (Python Optimal Transport)
python3 tarea_37.py
```

Requiere `numpy`, `scipy` y `pot`. Para el estudio completo, editar en `tarea_37.py` (función `epsilon_study`, al final del archivo) los parámetros `n` (número de embeddings), `d` (dimensión), `epsilons` y `n_iters` a valores más representativos (por ejemplo `n=5000`, `d=300` — dimensiones típicas de embeddings de modelos de lenguaje reales), y opcionalmente reemplazar `make_synthetic_embeddings` por embeddings reales extraídos de dos modelos de lenguaje distintos (p. ej. las matrices de embedding de dos tokenizers/vocabularios compartidos).

```json
{
  "alumno": null,
  "tarea": 37,
  "tipo_problema": "Alineación no supervisada de distribuciones de embeddings vía Transporte Óptimo Entrópico (Sinkhorn/POT) + Procrustes ortogonal (Wasserstein Procrustes)",
  "estado": "resuelto",
  "resultado_resumen": "Se implementó el esquema de Wasserstein Procrustes (Grave et al. 2019), alternando Sinkhorn (POT) y Procrustes ortogonal vía SVD, con evaluación por precisión de recuperación de vecino más cercano (top-1/top-5) sobre embeddings sintéticos con ground truth conocido.",
  "metrica_estandar_usada": "Costo de Sinkhorn (Wasserstein regularizado), residuo de Procrustes (norma de Frobenius), precisión top-1/top-5 de recuperación por vecino más cercano"
}
```
