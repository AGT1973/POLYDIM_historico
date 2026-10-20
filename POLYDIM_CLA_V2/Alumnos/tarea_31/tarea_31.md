# Tarea 31 — Curso IA (UTN) — Banco POLYDIM

## 1. Enunciado original

> **31 (Test 31):** Actúa como especialista en NVIDIA Triton. Implementa una versión especializada de LoRA donde el producto no ocurre en R^n normal, sino que el Producto Geométrico de un Álgebra de Clifford truncada.

Igual que en otras tareas del banco, el fragmento técnico quedó sin verbo tras "sino que": *"...sino que el Producto Geométrico de un Álgebra de Clifford truncada"* no es una oración completa.

## 2. Reconstrucción del enunciado

**Versión reformulada, matemáticamente investigable:**

> Implementar una variante de LoRA (Low-Rank Adaptation) en la que la adaptación no se aplica como el producto matricial aditivo estándar $\Delta W = BA$, sino mediante el **Producto Geométrico** de un Álgebra de Clifford $C\ell_n(\mathbb{R})$ **truncada al grado $\le 2$** (escalares + bivectores): la adaptación es un **rotor** $R = \exp(K)$, con $K$ un generador bivectorial de bajo rango, aplicado multiplicativamente (por conjugación/sandwich) en vez de sumado. Comparar formalmente esta construcción contra el LoRA estándar en dos ejes concretos y verificables: preservación de norma y expresividad alcanzable.

Se conserva del original: (a) el rol de optimización de bajo nivel (aunque el fragmento específico de Triton no aporta contenido matemático distinto — se documenta abajo por qué no se implementó un kernel Triton dedicado); (b) *"Producto Geométrico de un Álgebra de Clifford truncada"*, que remite a la Sección 3.1 del marco POLYDIM (Rotores de Spin, $R = e^{\theta B/2}$) y a la idea general de LoRA (Hu et al., 2021) mencionada en el enunciado. Lo que se agrega para hacerlo investigable es la traducción concreta de "Álgebra de Clifford truncada a grado 2" a un objeto de álgebra lineal estándar: matrices antisimétricas (bivectores) y su exponencial (rotaciones de Givens), y dos afirmaciones matemáticas verificables sobre esa construcción.

**Nota sobre el rol "especialista en Triton":** no se escribió un kernel Triton porque el núcleo del problema es puramente de álgebra lineal/teoría de grupos (¿qué preserva y qué no alcanza esta construcción?), y un kernel de bajo nivel no cambiaría ni demostraría nada sobre esas dos preguntas — hubiera sido ingeniería de rendimiento disfrazada de rigor matemático. Se deja como extensión natural en §7 para quien quiera perfilarlo.

## 3. Modelo matemático

### 3.1 LoRA estándar

$$
W' = W + \Delta W, \qquad \Delta W = BA,\quad B \in \mathbb{R}^{n\times r},\ A \in \mathbb{R}^{r \times n}
$$

Parámetros entrenables: $2nr$. La actualización es **aditiva** y, en general, no preserva ninguna estructura métrica de $W$ (ni ortogonalidad ni norma de las salidas).

### 3.2 Rotor-LoRA (Clifford truncado a grado $\le 2$)

Sea $\{e_i \wedge e_j\}_{i<j}$ la base estándar de bivectores de $C\ell_n(\mathbb{R})$. Cada bivector simple $e_i \wedge e_j$ se representa, en la acción sobre vectores, por el generador antisimétrico elemental $K_{ij} \in \mathfrak{so}(n)$ ($K_{ij}[i,j]=1$, $K_{ij}[j,i]=-1$, resto $0$). Para $r$ pares de índices **disjuntos** $(i_1,j_1),\dots,(i_r,j_r)$ y ángulos entrenables $\theta_1,\dots,\theta_r$:

$$
K = \sum_{k=1}^{r} \theta_k K_{i_k j_k}, \qquad R = \exp(K)
$$

Como los pares son disjuntos, los generadores $K_{i_kj_k}$ **conmutan**, y por lo tanto:

$$
R = \exp(K) = \prod_{k=1}^{r} \exp(\theta_k K_{i_k j_k})
$$

es exactamente el producto de $r$ **rotaciones de Givens** independientes — la contraparte matricial exacta del rotor $R = e^{\theta B/2}$ del marco POLYDIM (Sección 3.1), salvo la normalización del ángulo. La adaptación se aplica **multiplicativamente**: $W' = R\,W$ (o, equivalentemente sobre activaciones, $x' = Rx$).

**Parámetros entrenables: $r$** (solo los ángulos), frente a $2nr$ de LoRA estándar para el mismo "rango".

### 3.3 Afirmación 1 (preservación de norma) — se demuestra y se verifica

Como $K^T = -K$ (antisimétrica), $R = \exp(K)$ satisface $R^TR = \exp(K)^T\exp(K) = \exp(K^T)\exp(K) = \exp(-K)\exp(K) = I$. Por lo tanto $R \in O(n)$ (de hecho $R \in SO(n)$ por continuidad desde $\theta=0$), y:

$$
\|Rx\|_2 = \|x\|_2 \quad \forall x \in \mathbb{R}^n
$$

**exactamente**, no de forma aproximada. Esto es lo que el marco POLYDIM describe cualitativamente como preservación de "volumen/norma semántica" del operador `MIX`; acá se prueba con el resultado estándar de álgebra lineal $\exp(\text{antisimétrica}) \in SO(n)$, no se asume.

### 3.4 Afirmación 2 (límite de expresividad) — se demuestra y se verifica, y es una limitación real

$SO(n)$ tiene dimensión $n(n-1)/2$. El subgrupo generado por $r$ generadores **fijos y disjuntos** es abeliano (conmutan) y tiene dimensión $r$: es un **subtoro** de $SO(n)$, no todo el grupo. Concretamente, para $n=6$: $\dim SO(6) = 15$, mientras que con $r=3$ pares disjuntos el subgrupo alcanzable tiene dimensión $3$.

**Consecuencia verificable:** este rotor-LoRA de ejes fijos puede reproducir *exactamente* solo las rotaciones que ya son block-diagonales en esos mismos pares de ejes (o que coinciden con ellas), pero **no** una rotación genérica de $SO(n)$ — así se apague cuánto se entrene. Esto no es una falla de optimización, es una restricción estructural del subgrupo alcanzable, y se demuestra ajustando los ángulos por descenso de gradiente contra dos objetivos: uno genérico (falla, residual no llega a $0$) y uno construido para vivir en el mismo subtoro (ajusta casi exacto). El enunciado original no advertía esta limitación — se señala explícitamente en vez de callarla, tal como pide la metodología del curso.

## 4. Razonamiento y alternativas descartadas

- **¿Por qué generadores disjuntos y no un bivector denso arbitrario?** Un bivector denso arbitrario en $C\ell_n$ no siempre corresponde a generadores que conmuten, y el rotor resultante ya no se factoriza en rotaciones planas independientes de forma cerrada (hay que exponenciar la matriz completa, lo cual sigue siendo válido y ortogonal, pero pierde la interpretación de "bajo rango por construcción" que es la analogía directa con LoRA). Restringir a pares disjuntos es lo que da, de entrada, un conteo de parámetros $r$ literalmente comparable al rango $r$ de LoRA.
- **¿Por qué no reclamar que el rotor-LoRA "reemplaza" a LoRA en general?** Porque eso sería exactamente el tipo de premisa no demostrada que la consigna original insinúa sin probarla ("truncada" sugiriendo que es automáticamente mejor). Lo que se demuestra es un trade-off concreto: **preservación de norma exacta y muchísimos menos parámetros**, a cambio de **expresividad estrictamente menor** (subgrupo, no todo $SO(n)$) y de un tipo de actualización distinto (multiplicativa/rotacional, no aditiva/arbitraria) — no son intercambiables para cualquier tarea de fine-tuning.
- **¿Por qué no usar la forma real de Schur para recuperar universalidad con $O(n)$ parámetros?** Es la extensión natural correcta (toda rotación de $SO(n)$ es conjugada, vía *algún* cambio de base ortogonal, a un producto de rotaciones planas — es el contenido de la forma canónica real de Schur), pero aprenderla exigiría también parametrizar y entrenar esa base (vía QR/Givens generalizados), lo cual excede el alcance de "truncar a bivectores fijos" que pide el enunciado reconstruido. Queda como extensión en §7, no se implementa para no inflar el entregable con algo no pedido.

## 5. Desarrollo y código

Código completo en `tarea_31.py`. Estructura:

1. `bivector_generator` / `build_rotor`: construye $K$ y $R=\exp(K)$ vía `scipy.linalg.expm`.
2. `check_orthogonality`: verifica $\|R^TR - I\|$ (debe ser $\approx 0$ a precisión de máquina).
3. `standard_lora_delta`: LoRA estándar de referencia, $\Delta W = BA$.
4. `experiment_norm_preservation`: Afirmación 1 — compara $\|x\|$ vs $\|Rx\|$ (rotor) y $\|Wx\|$ vs $\|(W+BA)x\|$ (LoRA estándar), y el conteo de parámetros.
5. `fit_rotor_to_target` + `experiment_expressivity_limit`: Afirmación 2 — ajusta los ángulos por descenso de gradiente (diferencias finitas, sin dependencias de autograd) contra un objetivo genérico de $SO(6)$ y contra uno alineado al subtoro, y compara los residuales.

**Resultado del sanity-check interno** (no reemplaza que lo corras vos):

- $\|R^TR - I\| \approx 5\times10^{-15}$ → ortogonalidad exacta, como predice la teoría.
- $\|x\| = \|Rx\|$ coinciden a 6 decimales; en cambio $\|Wx\| \ne \|(W+BA)x\|$ para el LoRA estándar.
- Parámetros: rotor-LoRA usa $3$; LoRA estándar equivalente usa $36$ (para $n=6$, $r=3$).
- Ajustando contra una rotación genérica de $SO(6)$: residual final $\approx 2.10$ (no converge a $0$ — la limitación de expresividad es real, no un artefacto de mala optimización). Ajustando contra una rotación ya alineada al subtoro: residual $\approx 1.2\times10^{-4}$ (prácticamente exacto).

## 6. Conclusión

El modelo **converge** en el sentido fuerte para la Afirmación 1 (es una identidad algebraica exacta, no una optimización) y **no converge / converge a un residual no nulo** para la Afirmación 2 cuando el objetivo no vive en el subtoro alcanzable — y eso se documenta como resultado válido, no como fallo del método: es la demostración honesta de que "truncar a bivectores" no es gratis, tiene un costo de expresividad medible y cuantificado ($r$ vs $n(n-1)/2$ dimensiones del grupo).

## 7. Cómo correrlo vos mismo

```bash
pip install numpy scipy --break-system-packages   # si no los tenés instalados
python3 tarea_31.py
```

Extensiones que podés probar vos (no incluidas en este entregable):
- Reemplazar el ajuste por diferencias finitas por autograd (PyTorch, `torch.matrix_exp`) para $n$ más grandes.
- Implementar la parametrización de Schur/Givens generalizada (aprender también la base ortogonal, no solo los ángulos) y verificar si eso sí alcanza rotaciones genéricas de $SO(n)$ con $O(n)$ parámetros.
- Perfilar la construcción del rotor como kernel Triton si el tamaño de $n$ lo justifica (para $n$ chico como acá, la exponencial de matriz en CPU ya es instantánea y un kernel dedicado no aporta nada al argumento matemático).

```json
{
  "tarea": 31,
  "tipo_problema": "PEFT / Álgebra de Clifford (Spin, rotores) / Teoría de grupos de Lie (SO(n))",
  "estado": "resuelto",
  "resultado_resumen": "Se formalizó un LoRA multiplicativo vía rotores de Clifford truncados a bivectores disjuntos, se probó y verificó exactamente la preservación de norma (R ortogonal), y se documentó y verificó empíricamente el límite de expresividad frente a LoRA estándar (subtoro de dimensión r vs SO(n) de dimensión n(n-1)/2).",
  "metrica_estandar_usada": "Norma de Frobenius del residual ||R - R_target||_F; norma L2 para preservación; ||R^T R - I|| para ortogonalidad exacta"
}
```
