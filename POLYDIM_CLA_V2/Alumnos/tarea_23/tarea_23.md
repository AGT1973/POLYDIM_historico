# Tarea 23 — Woodbury para inversión de covarianzas diagonal + bajo rango

## 1. Enunciado

### 1.1 Original (banco maestro, Escuadrón 2 — "Álgebra de Clifford y Rotores")

> *"23 (Test 23): Actúa como DevOps y SRE de IA. Crea un arquetipo de una máquina virtual mínima en Rust/WASM que la identidad matricial de Woodbury para invertir matrices de covarianza de proyecciones de bajo rango."*

### 1.2 Nota de reconstrucción

El enunciado, tal como está en el banco, queda gramaticalmente roto: falta el verbo que conecta "una máquina virtual mínima en Rust/WASM" con "la identidad matricial de Woodbury...". Esto es consistente con lo que dice el propio banco maestro: los 200 ítems se generan combinando una plantilla de **rol** ("Actúa como DevOps y SRE de IA...") con un **fragmento técnico** ("...la identidad matricial de Woodbury para invertir matrices de covarianza de proyecciones de bajo rango"), y varias combinaciones pierden el verbo en el empalme.

Reconstrucción usada en esta entrega, manteniendo el rol y el fragmento técnico intactos:

> **Actuando como un DevOps/SRE de infraestructura de IA, construir un arquetipo de máquina virtual mínima (en Rust, con objetivo de compilación WASM) cuyo opcode central *implemente* la identidad matricial de Woodbury para invertir matrices de covarianza del tipo "diagonal + bajo rango" (proyecciones de bajo rango), documentando el costo computacional y los casos en que la inversión falla.**

No se asume ninguna conclusión no demostrada del enunciado original — el fragmento es una instrucción de construcción, no una afirmación a probar, así que no aplica la salvedad del paso 4 sobre premisas no establecidas.

## 2. Contexto en el paper POLYDIM V2

El fragmento conecta directamente con la Sección 5.1 del paper (*Information Geometry*):

> *"Convergence is calculated by optimizing the inverse of the Fisher Tensor ($F^{-1}$) via the Woodbury identity and KFAC in $\mathcal{O}(d \cdot r)$, solving the computational bottleneck of the Natural Gradient."*

Esto es exactamente el patrón **KFAC / Woodbury** usado en aproximaciones de gradiente natural: la matriz de Fisher (o una covarianza de activaciones/proyecciones) se aproxima como una parte diagonal más una corrección de bajo rango, y se invierte sin pagar el costo $O(d^3)$ de una inversión densa.

## 3. Modelo matemático

Sea $\Sigma \in \mathbb{R}^{d\times d}$ una matriz de covarianza con estructura **diagonal + bajo rango** (DPLR, *diagonal plus low rank*):

$$\Sigma = D + U U^ttop, \qquad D = \mathrm{diag}(d_1,\dots,d_d),\ d_i>0,\quad U \in \mathbb{R}^{d\times r},\ r \ll d$$

Esta es la forma estándar en la que aparecen las covarianzas de "proyecciones de bajo rango": $D$ captura varianza propia/ruido por coordenada y $U U^\top$ captura la estructura correlacionada de rango $r$ inducida por la proyección.

**Identidad de Woodbury** (Woodbury, 1950; también llamada fórmula de Sherman–Morrison–Woodbury):

$$(A + UCV)^{-1} = A^{-1} - A^{-1}U\left(C^{-1} + VA^{-1}U\right)^{-1}VA^{-1}$$

Con $A=D$, $C=I_r$, $V=U^\top$:

$$\Sigma^{-1} = D^{-1} - D^{-1}U\left(I_r + U^\top D^{-1} U\right)^{-1}U^\top D^{-1}$$

**Costo computacional:**
- Inversión densa de $\Sigma$: $O(d^3)$.
- Vía Woodbury: construir $D^{-1}U$ es $O(d\cdot r)$; el producto $U^top D^{-1} U$ es $O(d\cdot r^2)$; invertir el bloque $r\times r$ es $O(r^3)$; reconstruir $\Sigma^{-1}$ es $O(d\cdot r^2)$.
- Total: $O(d\cdot r^2 + r^3)$, que para $r \ll d$ se domina por el término $O(d\cdot r^2)$ — consistente en orden con el $O(d\cdot r)$ que menciona el paper (la diferencia $r$ vs. $r^2$ depende de si se cuenta el costo por columna o el total; ambos son lineales en $d$, que es la ganancia estructural real frente a $O(d^3)$).

**Condición de aplicabilidad (importante, y parte del rigor pedido):** la fórmula tal como está planteada exige que $D$ sea invertible por sí sola ($d_i \neq 0\ \forall i$). Si $\Sigma$ global es invertible pero $D$ no lo es, esta ruta de cómputo específica **no es aplicable** — no es que Woodbury sea "más débil", es que este split particular en $(D, U)$ deja de servir. Esto se documenta empíricamente en la Sección 5 de este entregable en vez de ocultarse.

## 4. Razonamiento y alternativas descartadas

- **Por qué no invertir $\Sigma$ directamente:** es la alternativa más simple, pero es $O(d^3)$ y no explota la estructura de bajo rango, exactamente el "cuello de botella" que el paper dice que Woodbury+KFAC resuelve.
- **Por qué Woodbury y no Sherman-Morrison iterado columna por columna:** Sherman-Morrison es el caso $r=1$; para $r>1$ conviene el bloque Woodbury completo, porque invertir el bloque $r\times r$ una sola vez es más estable numéricamente que aplicar $r$ actualizaciones de rango 1 encadenadas (que acumulan error de redondeo).
- **Por qué implementarlo como una "VM mínima" de opcodes y no como una función suelta:** el enunciado reconstruido lo pide explícitamente ("arquetipo de máquina virtual mínima en Rust/WASM"), y esto además refleja el patrón general del paper (Sección 9, "Smoking Gun Prototype"): las garantías topológicas/algebraicas se ejecutan como pasos discretos de una máquina de estados sobre un grafo de tensores, no como código imperativo libre.
- **Por qué sin crates externos (std puro) en Rust:** así el archivo compila con `rustc` solo, sin depender de `cargo` ni de descargar `nalgebra` desde crates.io, lo que hace la entrega reproducible sin fricciones de red/toolchain para cualquier compañero.

## 5. Desarrollo, código y resultados de sanity-check

Se entregan dos archivos de código, ambos funcionales (no pseudocódigo):

1. **`woodbury_vm.rs`** — la máquina virtual mínima pedida en Rust. Define:
   - Un tipo `Matrix` con las operaciones densas necesarias (`mul`, `transpose`, `add`, `sub`, `frobenius_norm`, `invert_dense` por Gauss-Jordan con pivoteo parcial).
   - `woodbury_inv(d_diag, U)`: implementa la fórmula de la Sección 3.
   - Una VM de opcodes (`LoadDiag`, `LoadU`, `WoodburyInv`, `Verify(eps)`, `Halt`) que ejecuta un programa como una secuencia de instrucciones sobre un stack, en vez de llamar funciones directamente — esto es lo que la hace "máquina virtual" y no solo una librería.
   - Un caso de éxito (D bien condicionada) y un caso de falla documentada (una entrada de D casi cero), sin ocultar el segundo.
   - Compila nativo con: `rustc woodbury_vm.rs -o woodbury_vm`
   - Para el target real WASM pedido en el enunciado: `rustc --target wasm32-unknown-unknown --crate-type=cdylib woodbury_vm.rs -o woodbury_vm.wasm` (requiere `rustup target add wasm32-unknown-unknown`; el `main()` con `println!` no corre igual en ese target, así que para WASM puro conviene exponer `woodbury_inv` como función `#[no_mangle] extern "C"` — se deja como paso siguiente para quien quiera empaquetarlo como módulo WASM real, ya que el enunciado pide el arquetipo de VM, no necesariamente el binario `.wasm` final).

   **Sanity-check interno realizado (solo para verificar que compila y no tiene errores obvios; no reemplaza la corrida del alumno):**
   ```
   === Caso 1: D bien condicionada ===
   WOODBURY_INV -> ok
   VERIFY: ||Sigma*Sigma^-1 - I||_F = 4.873e-16 (eps=1.0e-8) -> PASA
   Diferencia Woodbury vs inversión densa naive: 4.642e-16

   === Caso 2: D con una entrada ~0 (falla numérica esperada) ===
   WOODBURY_INV -> FALLO (D singular o mal condicionada)
   ```

2. **`woodbury_reference.py`** — referencia en NumPy, con tres partes: (a) verificación de que Woodbury coincide con la inversión densa, (b) benchmark de tiempos Woodbury vs. denso a medida que crece $d$ (con $r$ fijo), y (c) reproducción del caso de falla de la Sección 3.

## 6. Instrucciones para que vos lo corras

**Rust:**
```bash
rustc woodbury_vm.rs -o woodbury_vm
./woodbury_vm
```
No requiere ningún crate externo.

**Python:**
```bash
pip install numpy --break-system-packages  # si no lo tenés instalado
python3 woodbury_reference.py
```

Verificá con tus propios ojos:
- Que el residuo `||Sigma*Sigma^-1 - I||` en Rust y la diferencia `||Woodbury - naive||` en Python sean del orden de `1e-14`–`1e-16` (error de redondeo de punto flotante, no un error real).
- Que el benchmark en Python muestre el *speedup* de Woodbury creciendo a medida que `d` aumenta (con `r` fijo) — esa curva es la evidencia empírica de la ganancia $O(d\cdot r^2)$ vs. $O(d^3)$.
- Que el caso 2 (D con una entrada casi cero) falle explícitamente en ambos scripts, y pensá vos mismo si te convence la conclusión de la Sección 3 sobre por qué falla ahí específicamente y no en $\Sigma$ en general.

## 7. Conclusión

La identidad de Woodbury resuelve, para el caso diagonal+bajo-rango, la inversión de $\Sigma$ en $O(d\cdot r^2 + r^3)$ en vez de $O(d^3)$, reduciendo el bloque caro a una matriz $r\times r$. Esto es matemáticamente sólido y está bien establecido en la literatura de gradiente natural/K-FAC citada por el propio paper POLYDIM en la Sección 5.1. La implementación como VM de opcodes es una elección de diseño razonable para el enunciado pedido, pero **no es la única manera de implementarlo**, y el costo real en $d\cdot r$ vs. $d\cdot r^2$ depende de cómo se cuenten las operaciones — quedó documentado en la Sección 3 para que puedas juzgarlo vos mismo en vez de tomar la cifra del paper como dada.

```json
{
  "tarea": 23,
  "tipo_problema": "álgebra_lineal_numérica / inversión_de_covarianzas_bajo_rango",
  "estado": "resuelto",
  "resultado_resumen": "Woodbury implementado como VM de opcodes en Rust + referencia NumPy; residuo ~1e-15 en caso bien condicionado, falla documentada cuando D no es invertible.",
  "metrica_estandar_usada": "Norma de Frobenius del residuo ||Sigma*Sigma^-1 - I||_F"
}
```
