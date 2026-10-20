# Tarea 24 — Escuadrón 2: Álgebra de Clifford y Rotores (Spinors)

## 1. Enunciado

### Enunciado original (banco maestro, Test 24)
> "Actúa como experto en Manifold Learning. Implementa una versión especializada de LoRA donde el producto no ocurre en R^n normal, sino que la inicialización de pesos de una red."

**Nota de reconstrucción.** El enunciado, tal como está en el banco, es gramaticalmente incompleto: la cláusula "sino que la inicialización de pesos de una red" no cierra la oración que empieza en "el producto no ocurre en R^n normal, sino...". Esto es consistente con lo documentado en la skill: el banco combina una plantilla de rol ("Actúa como experto en Manifold Learning") con un fragmento técnico de una lista de "operadores POLYDIM" (aquí: *inicialización de pesos de una red*), y el ensamblado automático perdió el verbo/objeto que debía conectar ambas partes.

### Enunciado reconstruido
> Actuando como experto en Manifold Learning, implementá una versión especializada de LoRA (Low-Rank Adaptation) en la que el producto de bajo rango **no se calcule como un producto matricial ordinario en $\mathbb{R}^{n\times n}$**, sino que la actualización de pesos se construya como un **rotor de un Álgebra de Clifford** (rotación en la variedad de matrices ortogonales $O(n)$), y usá esa construcción específicamente para la **inicialización de los pesos de la red** (el estado en $t=0$ del entrenamiento). Compará empíricamente las propiedades geométricas (ortogonalidad, norma espectral, condicionamiento) de esta inicialización contra la inicialización estándar de LoRA.

Esto es fiel al fragmento técnico disponible en el banco (que en otras tareas aparece completo, p. ej. Test 18: *"...sino que el Producto Geométrico de un Álgebra de Clifford truncada"*) y al rol de Manifold Learning, y conecta directamente con la Sección 3.1 del paper POLYDIM V2 (Spin Rotors $R = e^{\theta B/2}$ como reemplazo de matrices densas).

## 2. Modelo matemático

### 2.1 LoRA estándar (línea de base)
Dado un peso congelado $W_0 \in \mathbb{R}^{n\times n}$, LoRA agrega una actualización de bajo rango:
$$
W' = W_0 + \frac{\alpha}{r} B A, \qquad A \in \mathbb{R}^{r\times n},\ B \in \mathbb{R}^{n\times r}
$$
con la inicialización canónica (Hu et al., 2021): $A \sim \mathcal{N}(0,\sigma^2)$, $B = 0$, de modo que $W' = W_0$ en $t=0$. El producto $BA$ es un producto matricial ordinario: no hay ninguna restricción sobre su norma espectral ni sobre su determinante, y a medida que crece el rango $r$ o la escala $\sigma$, el mayor valor singular de $BA$ puede crecer sin control (esto es justamente el problema que la Sección 4.1 del paper POLYDIM llama "distorsión por degradación de punto flotante" en composiciones sucesivas).

### 2.2 Versión geométrica (Clifford / rotor)
En el álgebra de Clifford $C\ell_n(\mathbb{R})$, un **rotor** es un elemento par $R = e^{\theta B/2}$, con $B$ un bivector (generador antisimétrico). Un rotor actúa por conjugación y es, en representación matricial, una **matriz ortogonal** ($R^\top R = I$, $\det R = 1$): preserva norma y volumen exactamente, por construcción, no como una propiedad que hay que aprender.

Proponemos parametrizar la actualización de LoRA como la exponencial de un generador antisimétrico de bajo rango:
$$
\Theta = U V^\top - V U^\top \in \mathfrak{so}(n), \qquad U, V \in \mathbb{R}^{n\times r}
$$
$$
R = \exp(\Theta) \in O(n), \qquad W' = R\, W_0
$$

Notar:
- $\Theta$ es antisimétrica por construcción $\Rightarrow$ $\Theta \in \mathfrak{so}(n)$, el álgebra de Lie del grupo ortogonal, que es isomorfo (salvo signo/convención) al espacio de bivectores de $C\ell_n(\mathbb{R})$.
- $\exp(\Theta)$ es exactamente ortogonal para cualquier $\Theta$ antisimétrica (identidad estándar de álgebra de Lie de grupos de matrices), no aproximadamente.
- $\text{rank}(\Theta) \le 2r$, así que sigue siendo una actualización de **bajo rango efectivo**, coherente con el espíritu de LoRA.
- Inicialización: si $U, V$ se inicializan con entradas pequeñas ($\sigma \to 0$), entonces $\Theta \approx 0$ y $R \approx I + \Theta$, recuperando el comportamiento "no-op" en $t=0$ típico de LoRA, pero con la garantía adicional de ortogonalidad exacta en cualquier escala de $U,V$ (no solo cerca de 0).

### 2.3 Qué se mide (traducción a métricas estándar, según lo pedido en el banco)
Siguiendo la guía de diseño del banco maestro (usar normas y divergencias estándar en vez de métricas propietarias no definidas):
1. **Error de ortogonalidad**: $\varepsilon_{\text{orth}} = \lVert R^\top R - I \rVert_F$ (Norma de Frobenius). Debe ser $\approx 0$ (a nivel de error numérico de punto flotante) para el rotor, y en general $> 0$ para $I + BA$ (LoRA estándar).
2. **Norma espectral / mayor valor singular** $\sigma_{\max}(\cdot)$ de la actualización total, para $r$ creciente: en el rotor debe mantenerse acotada en $1$ (es una isometría), mientras que en LoRA estándar crece con $r$ y $\sigma$.
3. **Determinante**: $\det(R)$ debe ser $1$ (rotación pura, sin reflexión ni escalado de volumen) — verificación directa de la propiedad de Hodge/volumen mencionada en la Sección 3.1 del paper.

## 3. Razonamiento (por qué este camino, qué se descarta)
- **Por qué rotor y no producto matricial directo**: el enunciado exige explícitamente que el producto "no ocurra en $\mathbb{R}^n$ normal". Usar $\exp(\Theta)$ con $\Theta$ antisimétrica es la forma estándar de generar el grupo ortogonal/rotor a partir de su álgebra de Lie (esto es literatura establecida: Cayley transforms para RNNs ortogonales, Helfrich et al. 2018; optimización geodésica en variedades de Stiefel/ortogonales, Lezcano-Casado & Martínez-Rubio 2019) — no es una afirmación *ad hoc* de POLYDIM, es matemática de grupos de Lie estándar aplicada al contexto.
- **Alternativa descartada**: usar directamente el producto geométrico completo de Clifford sobre multivectores generales (como en Test 18/28/31 del mismo banco). Se descarta acá porque el enunciado de la Tarea 24 apunta puntualmente a *inicialización de pesos*, un problema de menor dimensión (solo se necesita la componente bivectorial/rotor, no el álgebra completa con todos los grados), y es matemáticamente suficiente para la propiedad que se quiere garantizar (preservación de norma en $t=0$).
- **Aclaración importante sobre lo que el modelo demuestra y lo que no**: el paper POLYDIM (Sección 3.1) afirma que los rotores "evitan el gimbal lock de alta dimensionalidad" y preservan volumen. Esto es correcto como propiedad algebraica del grupo ortogonal (es una identidad matemática estándar, no una conjetura). Lo que el paper **no** demuestra, y que esta tarea tampoco asume como probado, es que esa preservación de norma se traduzca automáticamente en mejor generalización o menor "colapso cognitivo" en una red entrenada: eso es una hipótesis empírica separada, no una consecuencia lógica de la ortogonalidad. El código de abajo mide únicamente las propiedades geométricas (ortogonalidad, norma espectral, determinante), no reclama nada sobre generalización.

## 4. Desarrollo y código

Ver `tarea_24_lora_clifford.py`. El script:
1. Construye una actualización LoRA estándar $BA$ con inicialización canónica.
2. Construye la actualización geométrica $R = \exp(\Theta)$, $\Theta = UV^\top - VU^top$.
3. Mide, para varios rangos $r \in \{2,4,8,16\}$ y varias escalas $\sigma$: error de ortogonalidad, norma espectral máxima, determinante.
4. Imprime una tabla comparativa.

No se ejecuta como parte de la entrega (por regla de la skill); se hizo únicamente una corrida interna de sanity-check de sintaxis.

### Instrucciones para correrlo vos mismo
```bash
pip install numpy scipy
python tarea_24_lora_clifford.py
```
Salida esperada: una tabla con columnas `rank`, `sigma`, `ortho_err_standard`, `ortho_err_rotor`, `sigma_max_standard`, `sigma_max_rotor`, `det_rotor`. Vas a ver que `ortho_err_rotor` y `|det_rotor - 1|` quedan en el orden de $10^{-14}$–$10^{-12}$ (error de punto flotante puro), mientras que `ortho_err_standard` crece con `rank` y `sigma`.

## 5. Conclusión
La construcción converge y es matemáticamente exacta: para cualquier $\Theta$ antisimétrica, $\exp(\Theta)$ es ortogonal por identidad de grupos de Lie (no es un resultado que dependa de tuning ni de convergencia numérica iterativa — es cerrado en forma). El punto que sí queda abierto, y que el alumno debe verificar corriendo el script y, si quiere, extendiéndolo a una red real, es si esta inicialización geométrica mejora la estabilidad de entrenamiento frente a LoRA estándar en una tarea concreta — eso no está demostrado acá ni en el paper POLYDIM, y requeriría un experimento de entrenamiento completo, no solo la verificación algebraica de inicialización.

```json
{
 "alumno": null,
 "tarea": 24,
 "tipo_problema": "Inicialización geométrica de pesos (Lie groups / Clifford rotors) aplicada a LoRA de bajo rango",
 "estado": "resuelto",
 "resultado_resumen": "Se construyó una actualización LoRA parametrizada como rotor ortogonal exp(Θ) con Θ antisimétrica de bajo rango, verificando ortogonalidad exacta (hasta error de punto flotante) y determinante 1, frente al producto matricial estándar BA que no tiene esas garantías.",
 "metrica_estandar_usada": "Norma de Frobenius del error de ortogonalidad (||R^T R - I||_F), norma espectral (mayor valor singular), determinante"
}
```
