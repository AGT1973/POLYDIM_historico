# Tarea 36 — POLYDIM V2 (Sinkhorn + Identidad de Woodbury)

## 1. Enunciado original (banco maestro)

> "36 (Test 36): Actúa como experto en Categorical Deep Learning. Utiliza la
> librería POT o Sinkhorn para alinear estadísticamente la identidad
> matricial de Woodbury para invertir matrices de covarianza de proyecciones
> de bajo rango."

**Nota de reconstrucción.** Este enunciado combina dos plantillas que no
encajan tal cual: la plantilla de acción ("Utiliza POT/Sinkhorn para alinear
estadísticamente **X**") espera como **X** algo alineable — dos
distribuciones, dos nubes de puntos, dos conjuntos de embeddings — pero al
banco le tocó como **X** un fragmento que en realidad describe *otra*
técnica matemática ("la identidad de Woodbury para invertir matrices de
covarianza de bajo rango"), no un objeto alineable. Literalmente no tiene
sentido "alinear con Sinkhorn" una identidad matricial.

Lo que sí tiene sentido, y conecta ambos fragmentos de forma coherente, es
usar la identidad de Woodbury **dentro** del cálculo de Sinkhorn: el costo
de transporte óptimo (Mahalanobis) requiere la matriz de precisión
$\Sigma^{-1}$, y si $\Sigma$ tiene estructura de bajo rango + diagonal (como
las covarianzas/Fisher del propio paper POLYDIM, sección 5.1: *"optimizing
the inverse of the Fisher Tensor via the Woodbury identity and KFAC"*),
invertirla directamente es $O(d^3)$ — pero con Woodbury es $O(d\,r + r^3)$.
Esta es exactamente la clase de situación donde ambas piezas del enunciado
se necesitan mutuamente, en vez de ser instrucciones independientes.

**Versión reconstruida y investigable:**

> Implementar el algoritmo de Sinkhorn (transporte óptimo entrópico, en el
> espíritu de la librería POT) para alinear estadísticamente dos nubes de
> puntos (dos conjuntos de embeddings), usando como costo una distancia de
> Mahalanobis cuya matriz de precisión $\Sigma^{-1}$ —con $\Sigma$ de
> estructura diagonal + bajo rango— se calcula eficientemente mediante la
> identidad de Woodbury, evitando formar o invertir la matriz densa completa.

## 2. Modelo matemático

**Covarianza de bajo rango + diagonal.** $\Sigma = D + UU^\top$, con
$D\in\mathbb{R}^{d\times d}$ diagonal positiva y $U\in\mathbb{R}^{d\times r}$,
$r\ll d$ (estructura típica de aproximaciones de Fisher/K-FAC, sección 5.1
del paper). La identidad de Woodbury da su inversa sin pasar por $O(d^3)$:

$$\Sigma^{-1} = D^{-1} - D^{-1}U\big(I_r + U^\top D^{-1}U\big)^{-1}U^\top D^{-1}$$

El único sistema que hay que resolver es de tamaño $r\times r$
($I_r + U^\top D^{-1}U$), de costo $O(r^3)$; aplicar $\Sigma^{-1}$ a un lote
de $k$ vectores cuesta $O(dr k + r^3)$ en vez de $O(d^3 + d^2k)$.

**Costo de Mahalanobis entre dos nubes de puntos.** Dadas
$X=\{x_i\}_{i=1}^n$, $Y=\{y_j\}_{j=1}^m\subset\mathbb{R}^d$:

$$C_{ij} = (x_i-y_j)^\top \Sigma^{-1} (x_i-y_j)$$

Por bilinealidad, toda la matriz $C\in\mathbb{R}^{n\times m}$ se arma sin
iterar par por par:

$$C = \mathrm{diag}(X\Sigma^{-1}X^\top)\mathbf{1}^\top + \mathbf{1}\,\mathrm{diag}(Y\Sigma^{-1}Y^\top)^\top - 2X\Sigma^{-1}Y^\top$$

donde $X\Sigma^{-1}X^\top$, $Y\Sigma^{-1}Y^\top$ y $X\Sigma^{-1}Y^\top$ se
calculan aplicando `woodbury_apply_inverse` a $X^\top$ e $Y^\top$ como lotes
de columnas.

**Sinkhorn (transporte óptimo entrópico).** Con marginales
$a\in\Delta^{n-1}$, $b\in\Delta^{m-1}$ (uniformes en la implementación):

$$\min_{P\in U(a,b)} \langle P,C\rangle - \varepsilon H(P), \qquad U(a,b)=\{P\ge0: P\mathbf{1}=a,\ P^\top\mathbf{1}=b\}$$

se resuelve con la iteración de Sinkhorn–Knopp sobre $K=\exp(-C/\varepsilon)$:

$$u\leftarrow a\oslash(Kv), \qquad v\leftarrow b\oslash(K^\top u), \qquad P=\mathrm{diag}(u)\,K\,\mathrm{diag}(v)$$

El plan $P$ resultante es la "alineación estadística" entre $X$ e $Y$.

## 3. Razonamiento

Se descartó resolver el enunciado literal ("alinear con Sinkhorn una
identidad matricial") porque no es un objeto sobre el que Sinkhorn pueda
operar — Sinkhorn alinea *distribuciones*, no identidades algebraicas. En
vez de forzar una interpretación sin sentido, se identificó dónde
matemáticamente Woodbury y Sinkhorn **sí interactúan de forma natural**: el
costo de Mahalanobis que alimenta a Sinkhorn necesita $\Sigma^{-1}$, y
cuando $\Sigma$ viene de una estimación de covarianza de bajo rango (como
las que aparecen en el paper para el tensor de Fisher), Woodbury es
precisamente la herramienta que hace factible ese cálculo a escala. Se
prefirió esta lectura, ligada a la sección 5.1 del paper (Woodbury + KFAC
para $F^{-1}$) y a la sección 9.3 (ALIGN vía transporte óptimo entrópico),
por sobre inventar una alineación sin significado matemático.

Se verificó explícitamente, con una inversión directa $O(d^3)$ como
referencia, que la ruta Woodbury da el mismo resultado que la ruta directa
— tanto para la aplicación de $\Sigma^{-1}$ a un lote de vectores como para
la matriz de costos completa — antes de confiar en ella dentro de Sinkhorn.

## 4. Desarrollo y código

**`sinkhorn_woodbury_align.py`** implementa:
1. `woodbury_apply_inverse`: aplica $\Sigma^{-1}$ a un lote de vectores sin
   formar la matriz densa, usando `np.linalg.solve` sobre el sistema
   $r\times r$ (nunca se invierte explícitamente, ni el sistema pequeño ni
   $\Sigma$).
2. `direct_inverse_apply`: referencia lenta ($O(d^3)$) para verificación.
3. `mahalanobis_cost_matrix`: arma $C$ completa vía la identidad bilineal,
   reutilizando `woodbury_apply_inverse` sobre $X^\top$ e $Y^\top$.
4. `sinkhorn`: iteración de Sinkhorn–Knopp estándar con normalización y
   criterio de parada por convergencia de $u$.
5. `main`: genera dos nubes de puntos sintéticas (gaussianas con la misma
   estructura de covarianza bajo rango + diagonal, desplazadas entre sí),
   corre las tres verificaciones (Woodbury vs. directo, costo vía Woodbury
   vs. directo, marginales del plan de Sinkhorn) y reporta el costo de
   transporte final.

**Nota sobre ejecución en este entorno:** el script se corrió acá como
sanity-check interno — las tres verificaciones (`assert`) pasaron sin error
(errores numéricos en el orden de $10^{-8}$–$10^{-15}$, atribuibles solo a
precisión de punto flotante). Igualmente, la comprobación que cuenta la
tenés que hacer vos corriéndolo en tu propia cuenta, y experimentando con
otros valores de `d`, `r`, `eps` para ver cómo cambia el comportamiento.

## 5. Conclusión

- La identidad de Woodbury permite aplicar $\Sigma^{-1}$ (con $\Sigma$
  diagonal + bajo rango) sin nunca formar una matriz densa $d\times d$,
  reduciendo el costo de $O(d^3)$ a $O(dr+r^3)$ — confirmado numéricamente
  contra la inversión directa (error $\sim 10^{-15}$ en la aplicación a
  vectores, $\sim 10^{-13}$ en la matriz de costos completa).
- Ese ahorro es precisamente lo que hace viable meter la matriz de precisión
  dentro del costo de una distancia de Mahalanobis usada por Sinkhorn: sin
  Woodbury, cada evaluación de $\Sigma^{-1}$ escalaría mal con la dimensión
  del embedding $d$.
- El plan de transporte de Sinkhorn convergió a las marginales impuestas
  (error $<10^{-3}$ en ambos casos) usando ese costo, lo que valida el
  encadenamiento completo Woodbury → costo de Mahalanobis → Sinkhorn como
  una forma coherente de "alinear estadísticamente" dos conjuntos de
  embeddings con una métrica no euclidiana barata de evaluar.

## 6. Instrucciones para correr

```bash
pip install numpy --break-system-packages   # si numpy no está instalado
python3 sinkhorn_woodbury_align.py
```

No requiere la librería POT (todo Sinkhorn está implementado a mano en
`sinkhorn`), pero si querés comparar contra la implementación de referencia
de la librería, podés instalar `pip install pot --break-system-packages` y
contrastar tu plan `P` con `ot.sinkhorn(a, b, C, reg=eps)` sobre la misma
matriz de costos `C` que devuelve `mahalanobis_cost_matrix`. Jugá también
con `eps` (regularización entrópica): valores muy chicos hacen que Sinkhorn
tarde más en converger o sea numéricamente inestable — es una buena forma de
ver en carne propia el trade-off que menciona la sección 9 del paper entre
rigor exacto y tratabilidad numérica.
