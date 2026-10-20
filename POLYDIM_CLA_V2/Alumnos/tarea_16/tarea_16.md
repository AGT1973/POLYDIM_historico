# Tarea 16 — Kernel para descomposición de Hodge aproximada, aislando la componente armónica

**Alumno:** backup4512201@gmail.com
**Rol asignado:** DevOps y SRE de IA

---

## 1. Enunciado

### Enunciado original (banco maestro, tal cual)

> *"16 (Test 16): Actúa como DevOps y SRE de IA. Diseña un kernel en Triton ultra-optimizado para realizar la Descomposición de Hodge aproximada aislando la 'componente armónica'."*

### Nota de reconstrucción

El banco se genera combinando plantillas (rol + fragmento técnico), y varios fragmentos quedan gramaticalmente incompletos. En este caso el fragmento es comprensible pero subespecificado: no dice **sobre qué objeto matemático** se aplica la descomposición de Hodge, ni qué significa "aproximada" en este contexto, ni cuál es el rol concreto del kernel Triton dentro del cálculo completo.

Reconstrucción utilizada, explícita para que puedas comparar contra el original:

> *Actuando como ingeniero DevOps/SRE de IA, diseñar un kernel Triton que acelere el paso computacionalmente dominante (el producto Laplaciano–vector) de un algoritmo iterativo que calcula la descomposición de Hodge combinatoria de un flujo de aristas sobre un grafo, aislando su componente armónica (el subespacio ortogonal a los gradientes, asociado a los ciclos del grafo).*

Decisiones de reconstrucción que tomé, y por qué:

- **"Aproximada"**: la interpreto en el sentido de que (a) se trabaja sobre el 1-esqueleto del grafo (sin 2-simplices/triángulos), que es el caso particular de Hodge sin término de "curl", y (b) para grafos grandes el potencial se resuelve con un método iterativo (Conjugate Gradient) que converge de forma aproximada, no con inversión exacta.
- **Rol del kernel Triton**: un kernel no resuelve por sí solo un sistema lineal disperso completo de forma razonable; su rol natural es acelerar la operación interna que se repite en cada iteración de un solver (el producto Laplaciano-vector), que es memory-bound y se beneficia de fusión de operaciones. Esto es consistente con el rol "DevOps/SRE de IA" (optimizar el cómputo de un pipeline existente, no rediseñar la teoría matemática).
- **No asumo como hecho que "la descomposición aproximada" sea válida o correcta sin verificarlo**: la sección 4 de este documento comprueba explícitamente, con dos casos de control (árbol vs. grafo con ciclos), que la componente aislada efectivamente cumple la propiedad que la define (divergencia nula), en vez de simplemente afirmarlo.

---

## 2. Modelo matemático

Sea $G=(V,E)$ un grafo simple con $|V|=n$, $|E|=m$, y una orientación arbitraria fija de cada arista $e=(i,j)$.

**Operador de borde (matriz de incidencia orientada)** $B \in \mathbb{R}^{m\times n}$:

$$B[e,i] = -1,\quad B[e,j] = +1 \quad \text{para } e=(i,j)$$

Este operador define dos mapas duales:

- **Gradiente** $\text{grad}: \mathbb{R}^V \to \mathbb{R}^E$, $\quad (\text{grad}\,f)_{(i,j)} = f(j)-f(i) = (Bf)_{(i,j)}$
- **Divergencia** $\text{div}: \mathbb{R}^E \to \mathbb{R}^V$, $\quad \text{div} = B^\top$ (adjunto de grad)

**Laplaciano combinatorio de vértices**: $L_0 = B^\top B = D - A$ (grado menos adyacencia).

**Teorema (descomposición ortogonal del espacio de aristas).** Para cualquier flujo $\varphi \in \mathbb{R}^E$, existe una descomposición única

$$\varphi = \underbrace{Bs^*}_{\text{componente gradiente}} + \underbrace{h}_{\text{componente armónica}}, \qquad h := \varphi - Bs^*$$

donde $s^*$ resuelve las ecuaciones normales de mínimos cuadrados

$$L_0\, s^* = B^\top \varphi$$

(el gauge se fija anclando $s^*(\text{nodo de referencia})=0$, porque $L_0$ tiene como kernel las funciones constantes por componente conexa).

**Propiedad que caracteriza a $h$:** por construcción, $B^\top h = B^\top\varphi - L_0 s^* = 0$, es decir $h \in \ker(B^\top)$, el *cycle space* del grafo (Diestel). Cuando el complejo simplicial asociado no tiene 2-celdas, este subespacio coincide exactamente con la componente armónica de la teoría de Hodge combinatoria completa de Lim (2020) — no hay término de "curl" porque no existe un operador de borde superior $B_1$ sobre el cual definirlo.

**Dimensión de la componente armónica** (primer número de Betti):

$$\dim \ker(B^\top) = b_1 = m - n + c$$

donde $c$ es el número de componentes conexas. Si el grafo es un árbol, $b_1=0$ y $h\equiv 0$ — esta es la comprobación de control que usamos en el código.

**Extensión (Hodge completo con triángulos, opcional).** Si además se dispone de un conjunto de 2-simplices (triángulos) $T$ con borde $B_1\in\mathbb{R}^{|T|\times m}$, el Laplaciano de Hodge en aristas es

$$L_1 = B_0^\top B_0 + B_1 B_1^\top$$

y la componente armónica "verdadera" es $\ker(L_1) \subseteq \ker(B_0^\top)$ (los triángulos "rellenan" ciclos, reduciendo el subespacio armónico). Esta versión está implementada como extensión en el código (`hodge_decompose_with_triangles`), pero **no es la entrega principal** porque el fragmento del enunciado no especifica 2-simplices.

**Referencias:**
- Jiang, Lim, Yao, Ye (2011), *Statistical ranking and combinatorial Hodge theory*, Math. Programming 127(1).
- Lim (2020), *Hodge Laplacians on graphs*, SIAM Review 62(3).

---

## 3. Razonamiento y alternativas descartadas

1. **¿Por qué mínimos cuadrados sobre $L_0$ y no una fórmula cerrada?** Porque $L_0$ es singular (su kernel son las constantes), así que no existe inversa; la solución estándar en la literatura de HodgeRank es fijar el gauge (anclar un nodo) y resolver el sistema reducido, que es exactamente lo que hace `hodge_reference.py`.
2. **¿Por qué el kernel Triton ataca el producto Laplaciano-vector y no "toda" la descomposición?** Un solver iterativo (CG) hace decenas de productos matriz-vector; ese es el cuello de botella real en grafos grandes. Escribir un kernel Triton para "la descomposición completa" sería vago e improductivo — el punto de ingeniería real (rol DevOps/SRE) es optimizar la operación que se repite, no la orquestación (que queda en Python/PyTorch).
3. **Descarté** usar una factorización dispersa (Cholesky) precomputada para grafos grandes, porque el enunciado pide explícitamente un *kernel*, que tiene sentido en el contexto de un método iterativo matrix-free (donde nunca se materializa $L_0$ como matriz densa), no en un método directo.
4. **Verificación, no afirmación**: en vez de asumir que "la componente aislada es armónica", el código comprueba numéricamente $\|B^\top h\|\approx 0$ en ambos casos de prueba (sección 4) — esto es lo que exige el punto 4 de la metodología (no dar por sentado enunciados que asumen su propia conclusión).

---

## 4. Desarrollo, código y resultados

### 4.1 Implementación de referencia (NumPy/SciPy) — `hodge_reference.py`

Contiene:
- `build_incidence_matrix`: construye $B$ a partir de la lista de aristas.
- `hodge_decompose`: resuelve $L_0 s^* = B^\top\varphi$ con gauge fijo y devuelve gradiente + componente armónica + métricas de control.
- `first_betti_number`: calcula $b_1$ para comparar contra el resultado empírico.
- `hodge_decompose_with_triangles` / `build_upper_incidence`: extensión opcional con 2-simplices.

### 4.2 Validación empírica — `validar_hodge.py`

Corre dos casos de control y **ya fue ejecutado como sanity-check interno** (los resultados se muestran abajo para que los compares con tu propia corrida):

```
CASO 1: árbol (6 aristas, 7 nodos) -> b1 teórico = 0
  ||componente armónica||   = 3.55e-16   (≈ 0, correcto)
  ||B^T h|| (debe ser ~0)   = 5.93e-16
  error de reconstrucción   = 0.0

CASO 2: grafo con 2 ciclos (7 aristas, 6 nodos) -> b1 teórico = 2
  energía total   ||phi||^2      = 3.8486
  energía componente gradiente   = 3.4031
  energía componente armónica    = 0.4455
  ||B^T h|| (debe ser ~0)         = 1.17e-15   (correcto: divergencia nula)
  error de reconstrucción         = 0.0
```

Interpretación: en el árbol, la componente armónica es numéricamente cero (como predice la teoría, $b_1=0$). En el grafo con ciclos, la componente armónica captura ~11.6% de la energía total del flujo ($0.4455/3.8486$), y su divergencia es nula dentro de precisión de máquina — es decir, aísla exactamente el subespacio ortogonal a los gradientes, tal como pide el enunciado.

### 4.3 Kernel Triton — `hodge_triton_kernel.py`

Implementa:
- `_laplacian_edge_kernel`: kernel Triton que calcula $L_0 v = (D-A)v$ mediante un patrón scatter-add sobre la lista de aristas (evita materializar $L_0$ como matriz).
- `laplacian_matvec_triton`: wrapper que lanza el kernel.
- `conjugate_gradient_hodge`: resuelve $L_0 s = \text{rhs}$ con CG llamando al kernel en cada iteración — esta es la pieza que reemplazaría, en un grafo grande, al `spsolve` denso/disperso de `hodge_reference.py`.

**Importante — esto vos lo corrés, yo no lo ejecuté como parte de la entrega:** este archivo requiere una GPU NVIDIA con CUDA y el paquete `triton` instalado. No puede correr en un entorno sin GPU (como el mío al generar esta tarea). Por eso la corrección matemática **ya quedó demostrada de forma independiente** en la sección 4.2 con NumPy/SciPy puro — el kernel Triton es la capa de aceleración de ingeniería, no la fuente de verdad matemática.

Sanity-check que sí pude hacer sin GPU: compilación de sintaxis con `python3 -m py_compile hodge_triton_kernel.py` → sin errores.

---

## 5. Conclusión

- El fragmento del enunciado, aunque mal formado gramaticalmente, correspondía a un problema bien definido y con solución matemática exacta y verificable: la descomposición ortogonal del espacio de aristas de un grafo en componente gradiente + componente armónica (cycle space), vía el Laplaciano combinatorio $L_0=B^\top B$.
- La propiedad que define a la componente armónica ($B^\top h = 0$) **fue verificada empíricamente**, no asumida, en dos casos de control (árbol y grafo con ciclos), con error de reconstrucción exactamente cero y norma de divergencia en el orden de $10^{-15}$–$10^{-16}$ (precisión de máquina).
- El diseño del kernel Triton no intenta "resolver todo en Triton": ataca específicamente la operación memory-bound que domina el costo en grafos grandes (el producto Laplaciano-vector), dejando la orquestación del solver iterativo en Python/PyTorch — esa es la decisión de ingeniería que corresponde al rol DevOps/SRE de IA.

---

## 6. Instrucciones para que corras vos mismo el código

**Parte 1 (sin GPU, corre en cualquier máquina):**

```bash
pip install numpy scipy
python3 hodge_reference.py   # solo define funciones, no imprime nada por sí solo
python3 validar_hodge.py     # corre los 2 casos de control y muestra los resultados
```

Deberías obtener los mismos números que en la sección 4.2 (o muy cercanos, dado que `phi` se genera con una semilla fija `rng.default_rng(16)`).

**Parte 2 (requiere GPU NVIDIA + CUDA):**

```bash
pip install torch triton
python3 hodge_triton_kernel.py
```

Esto va a lanzar el kernel Triton sobre el mismo grafo de 6 nodos / 2 ciclos del caso 2, y va a imprimir el potencial $s$ recuperado por Conjugate Gradient. Si querés comparar contra la referencia NumPy, podés calcular `res.s` en `hodge_reference.py` para el mismo grafo y flujo, y verificar que ambos coincidan (salvo la constante de gauge, que en ambos casos está fijada en el nodo 0).

**Si querés ir más allá (extensión con triángulos):** revisá `hodge_decompose_with_triangles` en `hodge_reference.py`, y probá con un grafo donde agregues 2-simplices que "rellenen" alguno de los ciclos — deberías ver que la dimensión de la componente armónica baja respecto del caso sin triángulos.
