# Tarea 48 — POLYDIM V2 (Curso de IA, UTN)

## 1. Enunciado original (banco maestro, tal como aparece)

> **48 (Test 48):** Actúa como investigador de Análisis Estático. Crea un script en Python que analice el grafo computacional y la identidad matricial de Woodbury para invertir matrices de covarianza de proyecciones de bajo rango.

### Nota de reconstrucción (paso obligatorio antes de resolver)

El enunciado, como está escrito, es agramatical: la frase *"Crea un script [...] que analice el grafo computacional **y** la identidad matricial de Woodbury para invertir [...]"* no tiene un verbo que conecte "la identidad matricial de Woodbury" con el resto de la oración — el banco se genera combinando una plantilla de rol con un fragmento técnico, y en este caso el fragmento quedó pegado sin el verbo que lo integra a la oración.

**Reconstrucción adoptada:** interpretando el rol ("investigador de Análisis Estático") junto con el fragmento técnico reconocible ("identidad matricial de Woodbury para invertir matrices de covarianza de proyecciones de bajo rango"), el problema bien planteado es:

> *Diseñar un script en Python que realice un **pase de análisis estático sobre un grafo computacional**, detectando — a partir de metadatos estructurales de los nodos (forma y rango, no valores) — qué matrices de covarianza tienen estructura de bajo rango ($\Sigma = A + UCU^T$ con $A$ diagonal y $U \in \mathbb{R}^{d\times r}$, $r \ll d$). Para esos nodos, el script debe invertir la covarianza usando la **identidad de Sherman–Morrison–Woodbury** en lugar de inversión densa, y verificar empíricamente (a) la correctitud del resultado contra la inversión densa de referencia, y (b) la ventaja de complejidad computacional al escalar $d$.*

No hay ninguna premisa no demostrada que haya que dar por sentada aquí: la identidad de Woodbury es un resultado clásico de álgebra lineal (Sherman & Morrison, 1950; Woodbury, 1950) — lo que se comprueba empíricamente en esta tarea no es "si es cierta" (ya se sabe que sí, por demostración algebraica), sino que **la implementación es correcta** (error numérico ~precisión de máquina) y que **el costo computacional efectivamente escala mejor** que la inversión densa, que es la afirmación operativa relevante para POLYDIM (Sección 5.1 del paper: cómputo de $F^{-1}$ vía Woodbury + KFAC en $\mathcal{O}(d \cdot r)$).

## 2. Modelo matemático

Sea $\Sigma \in \mathbb{R}^{d\times d}$ una matriz de covarianza con estructura de actualización de bajo rango:

$$\Sigma = A + U C U^T, \qquad A \in \mathbb{R}^{d\times d} \text{ diagonal invertible}, \quad U \in \mathbb{R}^{d\times r}, \quad C \in \mathbb{R}^{r\times r} \text{ invertible}, \quad r \ll d.$$

La identidad de Sherman–Morrison–Woodbury da la inversa exacta:

$$\Sigma^{-1} = A^{-1} - A^{-1} U \left(C^{-1} + U^T A^{-1} U\right)^{-1} U^T A^{-1}.$$

**Costo computacional:**
- Inversión densa ingenua de $\Sigma$: $\mathcal{O}(d^3)$.
- Woodbury: $A^{-1}$ es $\mathcal{O}(d)$ (diagonal); el único sistema "caro" es la inversión de una matriz $r\times r$ ($\mathcal{O}(r^3)$), más productos matriciales $\mathcal{O}(d^2 r)$. Total: $\mathcal{O}(d^2 r + r^3)$, que para $r \ll d$ es sustancialmente menor que $\mathcal{O}(d^3)$.

**Métrica de verificación (estándar, según la guía del curso):** error relativo en norma de Frobenius,

$$\text{err} = \frac{\lVert \Sigma^{-1}_{\text{Woodbury}} - \Sigma^{-1}_{\text{denso}} \rVert_F}{\lVert \Sigma^{-1}_{\text{denso}} \rVert_F},$$

y adicionalmente $\lVert \Sigma \Sigma^{-1}_{\text{Woodbury}} - I \rVert_F / \lVert I \rVert_F$ como chequeo de identidad independiente de cuál método se toma como "referencia".

## 3. Razonamiento

- **Por qué análisis estático y no dinámico:** la decisión de qué estrategia de inversión usar (Woodbury vs. densa) se toma mirando únicamente la *forma* de los datos del nodo (shape de $A$, rango declarado $r$ de $U$), **no** sus valores numéricos. Esto es exactamente lo que hace un compilador/optimizador de grafos (p. ej. XLA, TorchDynamo) en tiempo de compilación: decide la estrategia antes de ejecutar, basado en metadatos de tipo/forma.
- **Por qué un umbral $r \le \alpha \cdot d$:** si $r$ es comparable a $d$ (por ejemplo $r = d/2$), Woodbury deja de ser ventajoso porque el término $\mathcal{O}(r^3)$ se vuelve tan costoso como el denso. El análisis estático incorpora ese umbral explícitamente (se usó $\alpha = 0.5$ en el código, ajustable).
- **Alternativa descartada:** hacer la elección de estrategia en tiempo de ejecución inspeccionando el rango numérico de $U$ (p. ej. vía SVD) sería más robusto ante estructuras "casi" de bajo rango, pero contradice el enunciado de "análisis estático" — y además introduce un costo $\mathcal{O}(d^2 r)$ solo para decidir, lo cual anula parte de la ganancia. Se documenta esta alternativa pero no se implementa, para mantener coherencia con el rol pedido (investigador de *análisis estático*).
- **Nodo de control (`cov_dense`):** se incluyó a propósito un nodo con covarianza densa sin estructura explotable, para que el análisis estático decida correctamente "no usar Woodbury ahí" — esto prueba que el pase no aplica Woodbury indiscriminadamente, sino según la estructura detectada.

## 4. Desarrollo y código

El código completo está en `tarea_48_woodbury_static_analysis.py` (mismo directorio). Estructura:

1. `GraphNode` / `build_synthetic_graph`: representación mínima de un grafo computacional con dos nodos de covarianza (uno denso de control, uno de bajo rango).
2. `static_inversion_plan`: el pase de análisis estático — decide `"woodbury"` o `"dense"` por nodo, mirando solo metadatos de forma/rango.
3. `invert_dense` / `invert_woodbury`: las dos implementaciones de inversión.
4. `verify_node`: compara ambos resultados con error relativo de Frobenius y con el chequeo $\Sigma\Sigma^{-1} \approx I$.
5. `benchmark_scaling`: mide tiempo real de ambos métodos para $d \in \{100, 200, 400, 800\}$ con $r=8$ fijo.

**Corrida de sanity-check interna** (solo para verificar que no hay errores de sintaxis/ejecución — no es la comprobación oficial, que corresponde al alumno):

```
Plan de inversión decidido por análisis estático (d=200, r=8):
  - nodo 'cov_dense': estrategia = dense
  - nodo 'cov_lowrank': estrategia = woodbury

Verificación de correctitud (comparando contra inversión densa exacta):
  - cov_dense       strategy=dense     rel_error_inv=0.000e+00
  - cov_lowrank     strategy=woodbury  rel_error_inv=3.546e-16

Benchmark de escalado (r fijo = 8, d creciente):
     d    t_dense (s)   t_woodbury (s)    speedup
   100       0.000400         0.000068      5.90x
   200       0.002326         0.000129     17.97x
   400       0.018602         0.000514     36.16x
   800       0.078699         0.002000     39.35x
```

Estos números son de **una** corrida de sanity-check en este entorno; van a variar según la máquina. **El alumno debe correrlo él mismo** para comprobarlo con sus propios ojos (ver instrucciones abajo).

## 5. Conclusión técnica

- **Correctitud:** el error relativo entre la inversa por Woodbury y la inversa densa de referencia es del orden de la precisión de máquina ($\sim 10^{-16}$), como se espera de una identidad algebraica exacta (no es una aproximación). Esto confirma que la implementación es correcta.
- **Complejidad/escalado:** el speedup empírico crece con $d$ a $r$ fijo (de ~6x en $d=100$ a ~39x en $d=800$ en esta corrida), consistente con la predicción teórica $\mathcal{O}(d^3)$ vs. $\mathcal{O}(d^2 r + r^3)$: la ventaja crece aproximadamente de forma lineal en $d/r$ para $r$ fijo, dentro del rango medido.
- **Análisis estático:** el pase correctamente asigna `"woodbury"` solo al nodo con estructura de bajo rango declarada, y `"dense"` al nodo de control — el mecanismo de decisión funciona sin necesitar evaluar los valores numéricos.
- **Límite de la conclusión:** esto no demuestra nada sobre POLYDIM como arquitectura en general — solo confirma, empíricamente y en este entorno, la corrección y el beneficio de complejidad de aplicar Woodbury cuando la estructura de bajo rango es *conocida de antemano* (vía metadatos). Si esa estructura no está garantizada en tiempo de compilación, la detección estática no aplica y haría falta otro mecanismo (ver la alternativa descartada en la Sección 3).

## 6. Instrucciones para que el alumno corra el código

Requisitos: Python 3.9+, `numpy`.

```bash
pip install numpy
python3 tarea_48_woodbury_static_analysis.py
```

El script no requiere GPU ni dependencias adicionales. Para explorar variantes:
- Cambiar `r` en `main()` (por ejemplo `r=2` o `r=50`) para ver cómo cambia el speedup y, con $r$ grande, cómo el análisis estático puede decidir `"dense"` si se supera el umbral `rank_threshold_ratio`.
- Agregar valores a `d_values` en `benchmark_scaling` para extender la curva de escalado (cuidado: $d > 2000$ puede tardar varios segundos en el camino denso).

```json
{
  "alumno": "",
  "tarea": 48,
  "tipo_problema": "análisis estático de grafo computacional + álgebra lineal numérica (identidad de Woodbury para inversión de covarianzas de bajo rango)",
  "estado": "resuelto",
  "resultado_resumen": "Pase de análisis estático que detecta estructura de bajo rango por metadatos y aplica Woodbury; error relativo vs. inversión densa ~1e-16 (precisión de máquina), speedup creciente con d/r verificado empíricamente hasta d=800.",
  "metrica_estandar_usada": "Error relativo en norma de Frobenius"
}
```
