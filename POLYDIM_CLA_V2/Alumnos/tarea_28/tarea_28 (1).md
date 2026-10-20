# Tarea 28 — POLYDIM V2 (Álgebra de Clifford / AST)

## 1. Enunciado original (banco maestro)

> "28 (Test 28): Actúa como especialista en OCaml/Haskell. Diseña una pasada de
> optimización que intercepte un AST topológico y el Producto Geométrico de un
> Álgebra de Clifford truncada."

**Nota de reconstrucción.** Este enunciado sale roto del banco (se genera por
combinación de plantillas rol + fragmento, y a varios fragmentos les falta el
verbo). Acá falta el verbo que conecta "intercepte un AST" con "el Producto
Geométrico de un Álgebra de Clifford truncada" — la plantilla original
seguramente decía algo como "...que intercepte un AST topológico y
**optimice/simplifique** el Producto Geométrico...". Además, "Álgebra de
Clifford truncada" es en sí mismo un término ambiguo: el enunciado da por
sentado que existe *una* forma natural de truncar el álgebra, sin decir cuál.
Esa es justamente la parte que hay que investigar en vez de asumir.

**Versión reconstruida y investigable:**

> Diseñar e implementar, en OCaml (o Haskell), una pasada de optimización de
> compilador sobre un AST que representa expresiones del Álgebra de Clifford
> $C\ell_{3,0}(\mathbb{R})$. La pasada debe (a) hacer *constant folding* del
> operador de Producto Geométrico cuando sus operandos son constantes, y (b)
> decidir, usando un análisis estático de grado, cuándo un truncamiento
> explícito del multivector resultante es **redundante** (se puede eliminar
> sin pérdida) y cuándo **no se puede probar** que lo sea — documentando en
> ese segundo caso que el truncamiento puede estar descartando información,
> en vez de asumir que "truncar" siempre es seguro.

## 2. Modelo matemático

$C\ell_{3,0}(\mathbb{R})$ es el álgebra de Clifford real de dimensión $2^3=8$,
generada por $e_1,e_2,e_3$ con $e_i^2=1$ y $e_ie_j=-e_je_i$ ($i\neq j$). Cada
blade base se representa como un subconjunto de $\{1,2,3\}$, codificado como
una máscara de bits $a\in\{0,\dots,7\}$; el grado de un blade es
$\mathrm{grade}(a)=\mathrm{popcount}(a)$.

**Producto geométrico de dos blades base.** Para métrica euclidiana
($e_i^2=+1$), el blade resultante del producto es $a\oplus b$ (XOR de las
máscaras) y su signo es $(-1)^{c}$ con

$$c=\sum_{k\ge 1} \mathrm{popcount}\big((a\gg k)\wedge b\big)$$

(algoritmo de reordenamiento canónico, estándar en álgebra geométrica
computacional — Dorst, Fontijne, Mann, *Geometric Algebra for Computer
Science*, cap. 21). El producto geométrico de dos multivectores densos
$A,B\in\mathbb{R}^8$ es bilineal sobre esta operación entre blades:

$$ (A\cdot B)_{a\oplus b} \mathrel{+}= \mathrm{sign}(a,b)\,A_a B_b $$

**La pregunta central del enunciado — ¿qué significa "truncar"?**
Hay (al menos) dos truncamientos candidatos y **no son intercambiables**:

- **Truncamiento por grado $\le 2$** (descartar solo el pseudoescalar,
  índice 7, grado 3): el subespacio resultante, de dimensión 7, **no es
  cerrado** bajo el producto geométrico. Contraejemplo: $e_{12}\cdot e_3 =
  e_{123}$, que cae fuera del subespacio.
- **Subálgebra par** $\{1, e_{12}, e_{13}, e_{23}\}$ (grados $0$ y $2$,
  dimensión 4): **sí es cerrada**. Justificación: si $a,b$ tienen
  $\mathrm{popcount}$ par, entonces $\mathrm{popcount}(a\oplus b)$ también es
  par, porque $\mathrm{popcount}(a\oplus b)=\mathrm{popcount}(a)+
  \mathrm{popcount}(b)-2\,\mathrm{popcount}(a\wedge b)$, y el término restado
  es par. Esta subálgebra es exactamente el álgebra de los rotores que el
  paper POLYDIM usa para el operador `MIX` ($R=e^{\theta B/2}$), isomorfa a
  los cuaterniones.

Esto es información que el enunciado original no provee, y que cambia
completamente qué optimización es correcta hacer. Por eso el enunciado
"truncada" a secas se investiga como pregunta abierta en vez de resolverse
dando por sentado un truncamiento en particular.

## 3. Razonamiento

Se descartó implementar la pasada de optimización asumiendo que "truncar a
grado $\le 2$" es seguro, porque el contraejemplo de arriba lo refuta
directamente: eliminar el pseudoescalar sin verificar el álgebra pierde
información en cuanto aparece un producto de dos bivectores con soporte
distinto. En cambio, la pasada se diseñó con un análisis estático de grado
(`grade_tag`) que solo permite eliminar un truncamiento cuando puede
**demostrarse** (no asumirse) que el resultado ya vive en un subespacio
cerrado — en este caso, la subálgebra par. Cuando no puede demostrarlo, el
compilador conserva el truncamiento y emite un diagnóstico en vez de fallar
en silencio o de "confiar" en que no hay pérdida.

Esto es, en espíritu, el mismo patrón que usan los compiladores reales para
*constant folding* condicionado a análisis estático (p. ej. eliminación de
checks de rango cuando el rango ya está probado en tiempo de compilación).

## 4. Desarrollo y código

- **`ast_pass.ml`** (OCaml): implementa el producto geométrico completo sobre
  multivectores densos (algoritmo bit-blade), el tipo de AST (`Const`, `Var`,
  `Add`, `GProd`, `TruncEven`), el análisis estático de grado (`tag_of`) y la
  pasada de optimización (`fold`) que hace *constant folding* del producto
  geométrico y elide `TruncEven` solo cuando puede probarlo seguro. Incluye
  al final una demostración mínima con dos casos: uno donde el truncamiento
  se elimina (par × par) y uno donde se conserva y avisa por `stderr`
  (bivector × vector, que fuga al pseudoescalar).
- **`verificacion_geometric_product.py`**: implementación independiente
  (en Python) del mismo producto geométrico, usada para verificar
  empíricamente, por fuera de OCaml, los tres hechos matemáticos en los que
  se apoya la pasada: (1) asociatividad del producto, (2) el contraejemplo
  de no-clausura de "grado $\le 2$", y (3) la clausura empírica (2000
  muestras aleatorias) de la subálgebra par.

**Nota sobre ejecución en este entorno:** el script de Python sí se corrió
acá como sanity-check interno (sin errores, las tres verificaciones pasan).
El archivo OCaml **no se compiló en este entorno** porque no hay toolchain
de OCaml instalado en el sandbox — fue revisado cuidadosamente a mano, pero
la comprobación real la tenés que hacer vos, como corresponde en esta etapa
del curso.

## 5. Conclusión

- El producto geométrico completo de $C\ell_{3,0}(\mathbb{R})$ es asociativo
  y computable en $O(64)$ por par de multivectores densos (verificado).
- El truncamiento "grado $\le 2$" propuesto implícitamente por el enunciado
  **no es cerrado** bajo el producto geométrico — es una premisa falsa si se
  la toma literalmente, y quedó refutada con un contraejemplo explícito, no
  asumida como cierta.
- La subálgebra par sí es cerrada, y es la elección matemáticamente correcta
  si el objetivo es tener un truncamiento sobre el cual una pasada de
  optimización pueda eliminar checks/truncamientos redundantes sin perder
  información — que es exactamente lo que hace `fold` en `ast_pass.ml`.

## 6. Instrucciones para correr

**Python (verificación empírica):**
```bash
python3 verificacion_geometric_product.py
```
No requiere dependencias externas (usa solo la librería estándar).

**OCaml (la pasada de optimización en sí):**
```bash
ocamlfind ocamlopt -package str ast_pass.ml -o ast_pass 2>/dev/null || ocaml ast_pass.ml
./ast_pass   # si compilaste con ocamlopt
```
Si no tenés OCaml instalado: `opam install ocaml` o corré el archivo
interpretado con `ocaml ast_pass.ml` directamente (no necesita librerías
externas, es todo `Stdlib`). Fijate especialmente que el segundo caso de la
demo (bivector × vector) imprima el warning por `stderr` y **no** elimine el
`TruncEven` — si eso no pasa, hay un bug para encontrar y corregir vos mismo.
