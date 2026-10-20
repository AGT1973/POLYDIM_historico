# Tarea 46 — POLYDIM (Escuadrón 3: Descomposición de Hodge y Grafos Esparsos)

## 1. Enunciado

**Original (banco maestro, Test 46):**

> "Actúa como arquitecto de Sistemas Distribuidos. Crea un arquetipo de una máquina virtual mínima en Rust/WASM que el mayor valor singular para asegurar convergencia topológica en FIXPOINT."

**Nota de reconstrucción.** El fragmento está gramaticalmente roto: la cláusula introducida por "que" no tiene verbo que la una a la acción principal ("Crea un arquetipo... que **[verbo faltante]** el mayor valor singular..."). Esto es consistente con lo esperado del banco (plantillas rol + fragmento técnico combinadas automáticamente, algunas quedan incompletas).

Reconstrucción propuesta, manteniendo el rol y el fragmento técnico reconocible:

> "Actúa como arquitecto de Sistemas Distribuidos. Crea un arquetipo de una máquina virtual mínima en Rust (compilable a WASM) que **implemente el operador `FIXPOINT` verificando y acotando el mayor valor singular del Jacobiano de la función de transición**, de modo de garantizar la convergencia topológica del punto fijo."

El fragmento técnico "mayor valor singular ... convergencia topológica en FIXPOINT" aparece repetido en varias tareas del banco (ver Tests 25, 39, 59, 63, 76, 92, 127, 142, 176) y coincide con la sección 9.4 del paper POLYDIM V2 (Amendment POPL 2027): el **Differentiable Banach Theorem**, que exige que el mayor valor singular del Jacobiano de `FIXPOINT` cumpla $\|J_T(x)\|_2 \le \lambda < 1$.

## 2. Modelo matemático

Sea $T: \mathbb{R}^n \to \mathbb{R}^n$ el operador de transición usado en el bucle `FIXPOINT`, definido como:

$$T(x) = \tanh(Wx + b), \qquad W \in \mathbb{R}^{n\times n},\; b \in \mathbb{R}^n$$

**Teorema del Punto Fijo de Banach (contracción).** Si existe $\lambda \in [0,1)$ tal que

$$\|T(x) - T(y)\|_2 \le \lambda \|x-y\|_2 \quad \forall x,y \in \mathbb{R}^n,$$

entonces $T$ tiene un único punto fijo $x^\* = T(x^\*)$, y la iteración $x_{k+1} = T(x_k)$ converge a $x^\*$ con tasa geométrica:

$$\|x_k - x^\*\|_2 \le \lambda^k \|x_0 - x^\*\|_2.$$

**Condición suficiente vía Jacobiano.** Como $\tanh$ es $1$-Lipschitz (su derivada $1-\tanh^2(\cdot) \in (0,1]$), y la composición de una función $1$-Lipschitz con una transformación afín de norma espectral $\sigma_{\max}(W)$ es $\sigma_{\max}(W)$-Lipschitz, se cumple:

$$\text{Lip}(T) \le \|J_T(x)\|_2 \le \sigma_{\max}(W) \quad \forall x.$$

Por lo tanto, **basta con acotar $\sigma_{\max}(W) \le \lambda < 1$** (norma espectral de $W$ = mayor valor singular) para garantizar la contracción y, vía Banach, la convergencia del bucle `FIXPOINT`. Este es el contenido exacto del "Differentiable Banach Theorem" del Apéndice del paper.

**Cómo se calcula $\sigma_{\max}(W)$ sin SVD completo.** Se usa *power iteration* sobre $W^\top W$:

$$u_{t+1} = \frac{W^\top (W u_t)}{\|W^\top (W u_t)\|_2}, \qquad \sigma_{\max}(W) \approx \|W u_\infty\|_2.$$

Esta es la misma técnica que usa *Spectral Normalization* (Miyato et al., 2018) para estabilizar GANs, y es la que se implementa en el código.

**Normalización espectral (enforcement).** Si $\sigma_{\max}(W) > \lambda$, se reescala:

$$W \leftarrow \frac{\lambda}{\sigma_{\max}(W)} \, W \quad \Rightarrow \quad \sigma_{\max}(W_{\text{nuevo}}) = \lambda.$$

## 3. Razonamiento (alternativas consideradas y descartadas)

- **SVD completo en cada paso** ($O(n^3)$): descartado por costo. Solo se necesita el mayor valor singular, no la descomposición completa — power iteration lo obtiene en $O(n^2)$ por iteración y converge geométricamente si el segundo valor singular no es igual al primero.
- **Cota de Lipschitz "exacta" para $T$ no lineal en general**: intratable para no linealidades arbitrarias. Se resuelve el problema componiendo una parte afín (cuya norma se controla exactamente) con una no linealidad conocida y globalmente $1$-Lipschitz ($\tanh$), de forma que la cota se hereda sin aproximaciones adicionales.
- **Forzar la contracción por recorte del vector de estado (clipping) en vez de la matriz**: descartado porque no da una garantía uniforme sobre todo $\mathbb{R}^n$; la normalización espectral de $W$ sí, porque acota el Jacobiano en todo punto.
- **Estimar $\sigma_{\max}$ con un único vector aleatorio sin iterar**: descartado por alta varianza; se usan 50–100 iteraciones de power iteration, suficientes para converger con precisión de máquina en dimensiones pequeñas/medias (ver verificación empírica frente a SVD exacto en el script Python).

## 4. Desarrollo y código

### 4.1 Rust (`fixpoint_vm.rs`) — arquetipo de VM mínima

Implementa, sin dependencias externas (para poder compilar tanto nativo como a `wasm32-unknown-unknown`):

- Álgebra matricial mínima (`Mat`: producto matriz-vector, transpuesta-vector).
- `spectral_norm_power_iteration`: estimación de $\sigma_{\max}(W)$ vía power iteration determinista (generador congruencial propio, sin `rand`).
- `enforce_spectral_bound`: aplica la normalización espectral descripta arriba y devuelve $(\sigma_{\text{antes}}, \sigma_{\text{después}})$.
- `fixpoint_op`: la instrucción `FIXPOINT` de la VM — corre $x_{k+1} = \tanh(Wx_k+b)$ hasta que $\|x_{k+1}-x_k\|_2 < \text{tol}$ o se agote `max_iter`, registrando el error en cada paso.

**Compilar y correr (nativo):**
```bash
rustc -O fixpoint_vm.rs -o fixpoint_vm
./fixpoint_vm
```

**Compilar a WASM** (requiere agregar el target una sola vez):
```bash
rustup target add wasm32-unknown-unknown
rustc --target wasm32-unknown-unknown -O fixpoint_vm.rs -o fixpoint_vm.wasm
```

### 4.2 Python (`fixpoint_check.py`) — verificación empírica independiente

Reimplementa el mismo modelo con `numpy` para que el alumno pueda:
1. Comparar la estimación por power iteration contra el SVD exacto de `numpy.linalg.svd` (error relativo esperado $\sim 10^{-16}$, límite de precisión de punto flotante).
2. Confirmar que tras la normalización espectral, $\sigma_{\max}(W) = \lambda$ exactamente (o queda igual si ya estaba por debajo).
3. Ajustar una recta a $\log(\text{error}_k)$ vs. $k$ para estimar la tasa de decaimiento empírica y verificar que es $\le \lambda$ (consistente con la cota de Banach).

**Correr:**
```bash
pip install numpy --break-system-packages
python3 fixpoint_check.py
```

## 5. Conclusión

- El modelo matemático estándar detrás de la instrucción `FIXPOINT` es, literalmente, el **Teorema del Punto Fijo de Banach** aplicado a una red con no linealidad $1$-Lipschitz y parte lineal acotada en norma espectral. No hace falta ningún aparato categórico adicional para este resultado puntual: es análisis funcional estándar (Banach, 1922) combinado con una técnica de ingeniería bien conocida (spectral normalization, 2018).
- **Métrica estándar usada:** norma espectral $\sigma_{\max}(W)$ (equivalente a la norma de operador inducida por $\ell_2$) y norma $L_2$ del residuo $\|x_{k+1}-x_k\|_2$ como criterio de parada/convergencia.
- **Resultado empírico (sanity-check interno, no reemplaza la corrida propia del alumno):** en ambas implementaciones (Rust y Python), forzar $\sigma_{\max}(W) \le \lambda = 0.9$ produjo convergencia en menos de 100 iteraciones, con tasa de decaimiento del error siempre $\le \lambda$ — consistente con la teoría. Los valores numéricos exactos dependen de la semilla/`W` inicial: **correlos vos mismo** con `cargo`/`rustc` y `python3` para ver tus propios números y, si querés, cambiá `lambda_target`, `n` o `W` para explorar qué pasa cuando $\lambda \ge 1$ (dejá de estar garantizada la convergencia — probalo).
- **Caveat honesto:** la cota $\sigma_{\max}(W) \le \lambda$ garantiza que $T$ es una contracción *global* en $\mathbb{R}^n$ con la métrica euclídea estándar. Si en una versión futura `FIXPOINT` operase sobre un dominio restringido (p. ej. una variedad o un subespacio con otra métrica, como sugiere el resto del paper POLYDIM con Clifford/Hodge), la equivalencia entre "norma espectral acotada" y "contracción" tendría que revisarse para esa métrica — eso no está demostrado acá, queda como pregunta abierta si se quiere extender el resultado.

## 6. Resumen estructurado

```json
{
  "tarea": 46,
  "tipo_problema": "Punto fijo / control de Lipschitz vía norma espectral (análisis numérico + optimización)",
  "estado": "resuelto",
  "resultado_resumen": "VM mínima en Rust (portable a WASM) que implementa FIXPOINT con normalización espectral de W (power iteration) para garantizar la contracción de Banach (lambda<1) y por lo tanto la convergencia geométrica del punto fijo. Verificado con script Python independiente contra SVD exacto.",
  "metrica_estandar_usada": "Norma espectral (sigma_max, operador L2) del Jacobiano; norma L2 del residuo entre iteraciones sucesivas"
}
```
