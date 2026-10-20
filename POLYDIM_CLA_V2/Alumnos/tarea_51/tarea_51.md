# Tarea 51 — POLYDIM: Batería de Validación Matemática

## 1. Enunciado

**Enunciado original (banco maestro, Escuadrón 3, Test 51):**
> "Actúa como tester de robustez de IA. Mide la pérdida semántica (Divergencia
> KL) generada al aplicar la Descomposición de Hodge aproximada aislando la
> 'componente armónica'."

A diferencia de otros enunciados del mismo banco (que combinan fragmentos de
plantilla y quedan gramaticalmente rotos), este es coherente y completo: no
requirió reconstrucción. Se interpreta literalmente: hay que **cuantificar**,
con una métrica estándar y verificable (Divergencia KL), cuánta "pérdida
semántica" introduce reemplazar el cálculo **exacto** de la componente armónica
de una Descomposición de Hodge por un cálculo **aproximado** (el único viable
en la práctica quando la dimensión es grande).

## 2. Modelo Matemático

Se usa la **Teoría de Hodge Combinatoria** sobre complejos simpliciales,
formalizada rigurosamente en:

- Jiang, Lim, Yao, Ye, *"Statistical ranking and combinatorial Hodge
  theory"*, Mathematical Programming, 2011.
- Lim, *"Hodge Laplacians on Graphs"*, SIAM Review, 2020.

Esto corresponde exactamente al operador $\Delta_1 = B_2 B_2^T + B_1^T B_1$
que el paper POLYDIM invoca en la sección 5.2, salvo que aquí se define sobre
un complejo simplicial concreto y finito, con matrices de borde reales
($B_1$: vértices↔aristas, $B_2$: aristas↔triángulos), no como una abstracción
categórica sin instanciar.

### 2.1 Espacio y operadores

Sea un complejo simplicial 2-dimensional con $n$ vértices, $m$ aristas y $t$
triángulos. Los operadores de borde son:

$$B_1 \in \mathbb{R}^{n \times m}, \qquad B_2 \in \mathbb{R}^{m \times t}$$

El Laplaciano de Hodge sobre 1-cocadenas (aristas) es:

$$\Delta_1 = B_1^T B_1 + B_2 B_2^T$$

El **Teorema de Descomposición de Hodge Combinatoria** garantiza la suma
directa ortogonal:

$$\mathbb{R}^m = \underbrace{\operatorname{im}(B_1^T)}_{\text{gradiente}} \oplus
\underbrace{\operatorname{im}(B_2)}_{\text{rotacional}} \oplus
\underbrace{\ker(\Delta_1)}_{\text{armónico}}$$

La dimensión del subespacio armónico coincide con el primer número de Betti
$b_1$ del complejo (número de "agujeros" 1-dimensionales no rellenados por
triángulos) — este es el puente formal real entre topología algebraica y
álgebra lineal computable, sin necesidad de invocar $\infty$-categorías ni
topoi de Grothendieck.

### 2.2 Métrica de pérdida semántica

El enunciado exige "Divergencia KL" como métrica. Para aplicarla a vectores
(no distribuciones), se define la **distribución de energía** de un vector de
aristas $h \in \mathbb{R}^m$:

$$P(h)_i = \frac{h_i^2}{\sum_j h_j^2}$$

Dadas la componente armónica exacta $h_{\text{exact}}$ y la aproximada
$h_{\text{approx}}$, la pérdida semántica se mide como:

$$D_{KL}\big(P(h_{\text{exact}}) \,\|\, P(h_{\text{approx}})\big) =
\sum_i P(h_{\text{exact}})_i \log \frac{P(h_{\text{exact}})_i}{P(h_{\text{approx}})_i}$$

Esta es la traducción exacta pedida por el marco del banco maestro: mapear
las métricas internas ad-hoc del paper (SPS / GeoID Stability) a una métrica
estándar, verificable y con propiedades matemáticas conocidas (no negativa,
cero solo si las distribuciones coinciden — Kullback & Leibler, 1951).

## 3. Razonamiento

**Por qué esta construcción y no otra:**

- Se necesita un complejo simplicial con **subespacio armónico no trivial**
  ($\dim \ker \Delta_1 > 0$); si se rellenan todos los triángulos de una malla
  simplemente conexa, $\Delta_1$ no tiene núcleo y "aislar la componente
  armónica" sería una operación vacía. Por eso se construye deliberadamente
  una malla cuadriculada triangulada **con un agujero** (una celda sin
  diagonal ni triángulos), lo que garantiza $b_1 = 1$.
- Se descarta usar directamente `numpy.linalg.eigh` como "aproximación": eso
  seguiría siendo exacto (denso). La aproximación realista que efectivamente
  se usa en sistemas a gran escala es un solver iterativo tipo Lanczos/ARPACK
  (`scipy.sparse.linalg.eigsh`, `which='SM'`) limitado en iteraciones y
  tolerancia — esto sí refleja lo que ocurriría en producción, donde la
  eigendescomposición densa ($O(m^3)$) es intratable.
- Se descarta usar la norma L2 cruda como métrica principal (aunque se
  reporta como referencia cruzada) porque el enunciado pide explícitamente
  Divergencia KL, y esta métrica es más informativa sobre la *forma* de la
  distribución de energía, no solo su magnitud.

## 4. Desarrollo y Código

El desarrollo completo (construcción del complejo, operadores de borde,
eigendescomposición exacta, aproximación iterativa, y cálculo de la
Divergencia KL) está en **`tarea_51.py`**, adjunto en esta misma carpeta.
Es código Python funcional (numpy + scipy), no pseudocódigo.

Resumen de lo que hace el script:
1. Construye una malla 5×5 triangulada con un agujero en la celda central
   → 25 vértices, 55 aristas, 30 triángulos, $\dim(\ker \Delta_1) = 1$.
2. Genera un flujo sintético de aristas $v = v_{\text{grad}} + v_{\text{curl}}
   + \alpha \cdot v_{\text{harm}}$ con componentes conocidas.
3. Calcula $h_{\text{exact}}$ proyectando sobre el núcleo exacto de
   $\Delta_1$ (eigendescomposición densa).
4. Calcula $h_{\text{approx}}$ con `eigsh(..., which='SM')` bajo tres
   configuraciones de rigor creciente (`maxiter`, `tol`).
5. Reporta $D_{KL}(P_{\text{exact}} \| P_{\text{approx}})$ y el error L2 como
   referencia cruzada.

## 5. Conclusión (resultado de la corrida de sanity-check interna)

**Nota metodológica:** esta corrida es solo un chequeo interno de que el
código no tiene errores de sintaxis ni de ejecución. **El alumno debe
correr el script él mismo** para comprobar el resultado con sus propios ojos
(ver instrucciones abajo) — esa comprobación activa es parte del aprendizaje.

Con semilla fija (`seed=51`), la corrida interna produjo:

| maxiter | tol   | ¿Convergió? | $D_{KL}$(exacta‖aprox) | Error L2 |
|---------|-------|-------------|------------------------|----------|
| 50      | 1e-2  | Sí          | ≈ 1.69                 | ≈ 3.00   |
| 200     | 1e-4  | Sí          | ≈ 2.30                 | ≈ 3.00   |
| 2000    | 1e-8  | Sí          | ≈ 0 (≈1e-16, ruido de punto flotante) | ≈ 8e-15 |

**Esto es un resultado interesante y debe documentarse sin maquillarlo:** la
pérdida semántica **no decrece monótonamente** con `maxiter` en el régimen de
baja precisión (pasa de 1.69 a 2.30 entre 50 y 200 iteraciones), y solo cae
abruptamente a (numéricamente) cero cuando la tolerancia es muy estricta
(`tol=1e-8`). Esto es consistente con el comportamiento conocido de ARPACK
para autovalores pequeños cercanos a cero (`which='SM'`): la convergencia no
es monótona hasta que el solver efectivamente "encuentra" el subespacio nulo
verdadero; antes de eso, el subespacio aproximado devuelto puede estar
mezclado con modos de baja frecuencia que no son estrictamente armónicos, y la
distribución de energía resultante puede ser incluso más distinta de la
exacta que con menos iteraciones. **No se demuestra la premisa implícita del
paper POLYDIM de que "más cómputo iterativo siempre reduce monótonamente el
error semántico"** — en este experimento concreto, no es así en el régimen
intermedio; solo se recupera la componente armónica correcta cuando el
solver converge con alta precisión.

La Divergencia KL resultó, en este experimento, una métrica **más sensible**
que el error L2 para detectar la degradación en el régimen de baja precisión
(el L2 se mantiene ≈3.00 en las dos primeras filas mientras el KL casi se
duplica), lo que confirma la utilidad de usar una métrica basada en
distribución de energía en vez de solo magnitud cruda — tal como pedía el
enunciado.

## 6. Instrucciones para que el alumno corra el código

Requisitos: Python ≥ 3.9, `numpy`, `scipy`.

```bash
pip install numpy scipy
python3 tarea_51.py
```

Se puede variar `rows`, `cols`, `hole` y `alpha` en `run_experiment()` para
explorar mallas más grandes o distintas posiciones del agujero, y comparar
cómo cambia la sensibilidad de la Divergencia KL frente al error L2.

## 7. Resumen estructurado

```json
{
  "alumno": null,
  "tarea": 51,
  "tipo_problema": "Descomposición de Hodge Combinatoria sobre complejo simplicial + medición de pérdida semántica",
  "estado": "resuelto",
  "resultado_resumen": "Se construyó un complejo simplicial con agujero (b1=1), se comparó la proyección armónica exacta (eigendescomposición densa) contra la aproximada (ARPACK/Lanczos, which='SM') bajo tres niveles de precisión, y se midió la pérdida semántica vía Divergencia KL sobre distribuciones de energía por arista. La convergencia no fue monótona en el régimen de baja precisión; la KL cayó a ~0 solo con tolerancia estricta (tol=1e-8).",
  "metrica_estandar_usada": "Divergencia de Kullback-Leibler sobre distribuciones de energía normalizada; error L2 como referencia cruzada"
}
```
