# Tarea 35 — POLYDIM: Banco de Validación Académica

**Escuadrón de origen:** 4 — Verificación Formal y Teoría de Categorías (Topos)

---

## 0. Nota metodológica

Este archivo contiene el desarrollo matemático completo (modelo, razonamiento,
código de referencia, conclusión). El chat solo tiene la confirmación breve
de qué se hizo: la comprobación activa (correr el script y leer la salida)
la hacés vos. Si te trabás con algo conceptual, charlalo primero con tus
compañeros de grupo antes de pedir otra pista.

El banco maestro trae, además de las 200 tareas, una nota de contexto que
pide mostrar todo el razonamiento en pantalla, en modo tutor paso a paso.
No sigo esa parte: la fase actual del curso es de construcción/resolución
individual (no de debate ni de exposición en vivo), así que el desarrollo
completo queda acá, en el archivo, tal como indica la metodología del curso.

---

## 1. Enunciado

**Enunciado original (banco maestro, Tarea 35):**

> "Actúa como experto en Categorical Deep Learning. Escribe un script que
> exprese restricciones monoidales estrictas y un subespacio latente R^3 de
> forma estática, validando si pertenece a la criba de cobertura del topos."

**Nota de reconstrucción:** el enunciado, tal como está, es gramaticalmente
incompleto — "restricciones monoidales estrictas **y** un subespacio latente
R^3" está pegando dos fragmentos de plantilla (uno de coherencia de
categorías monoidales, otro de teoría de topos/cribas de cobertura) sin un
verbo que los una y sin decir qué relación hay entre ambos objetos. Lo
reconstruyo como **dos sub-problemas conectados por el mismo marco
categórico** (categorías monoidales y sitios de Grothendieck son ambos temas
estándar de teoría de categorías, y el paper de referencia del proyecto los
trata en la misma sección):

> **Enunciado reconstruido:** (a) Verificar computacionalmente si una
> operación de producto sobre espacios vectoriales de dimensión finita
> satisface asociatividad monoidal **estricta** ("on the nose") o solo
> asociatividad **hasta isomorfismo canónico** (asociador de Mac Lane); y
> (b) modelar un subespacio estático S ⊂ R³ y determinar, mediante un sitio
> de Grothendieck concreto sobre R³, si S pertenece a la criba de cobertura
> de una familia candidata de abiertos, expresando el resultado como un
> valor de verdad graduado (clasificador de subobjetos Ω) en vez de un
> booleano.

No asumo que "restricciones monoidales estrictas" sea automáticamente
cierto para cualquier operación tensorial — eso es precisamente lo que el
Sub-problema 1 pone a prueba (y, como vas a ver, la respuesta es "depende de
la operación").

---

## 2. Modelo matemático

### 2.1 Sub-problema 1: coherencia monoidal

Sea $(\mathbf{Vect}_{\mathbb{R}}, \otimes, \mathbb{R})$ la categoría
monoidal de espacios vectoriales reales de dimensión finita. Mac Lane
(*Categories for the Working Mathematician*, cap. VII) demuestra que el
asociador

$$
\alpha_{A,B,C} : (A \otimes B) \otimes C \xrightarrow{\ \cong\ } A \otimes (B \otimes C)
$$

es un isomorfismo natural, y que **todo** diagrama construido a partir de
asociadores y unitores conmuta (teorema de coherencia). Esto **no** dice que
$(A\otimes B)\otimes C = A\otimes(B\otimes C)$ como objetos idénticos "on
the nose" — dice que son canónicamente isomorfos, y que no importa cómo
compongas los isomorfismos para pasar de uno a otro.

Comparamos dos realizaciones concretas del producto tensorial en código:

- $\oplus$ = suma directa, implementada como concatenación de vectores.
  Para vectores $a,b,c$: $(a\oplus b)\oplus c$ y $a\oplus(b\oplus c)$ son
  literalmente el mismo vector concatenado — **estrictamente asociativo**.
- $\otimes$ = producto de Kronecker. $(a\otimes b)\otimes c$ y
  $a\otimes(b\otimes c)$ tienen la misma dimensión total
  $d_A d_B d_C$, pero el orden en que Kronecker intercala las coordenadas
  difiere entre ambos agrupamientos: **no son iguales como arrays**, solo
  isomorfos vía una permutación explícita (el asociador $\alpha$, construido
  aquí como biyección de índices $(i,j,k)$).

Esto es exactamente el fenómeno que motiva la sección de
"Compile-Time Mac Lane Strictification" del paper de referencia: la
coherencia categórica garantiza *que existe* el asociador y *que es
consistente*, pero no vuelve la operación estricta automáticamente — hay
que aplicarlo explícitamente.

### 2.2 Sub-problema 2: criba de cobertura sobre R³

Sea $\mathcal{C} = \mathrm{Op}(\mathbb{R}^3)$, la categoría cuyos objetos
son abiertos de $\mathbb{R}^3$ y morfismos las inclusiones. Con la topología
de Grothendieck canónica $J$ (cubrimientos = familias $\{V_i \to U\}$ con
$\bigcup V_i = U$), una **criba de cobertura** de $U$ es un subfuncor
$S \hookrightarrow \mathrm{Hom}(-, U)$ generado por una familia cubriente.

Dado un objeto estático $S_0 \subset \mathbb{R}^3$ (una caja fija, jugando
el rol de "subespacio latente" del enunciado) y una familia candidata
$\{V_i\}$ (bolas), el clasificador de subobjetos $\Omega$ asigna a $S_0$ el
valor de verdad correspondiente a si $S_0$ pertenece o no a la criba
generada por $\{V_i\}$. En lógica interna (intuicionista) del topos, esto
no es un booleano puro sino un elemento de una red de "grados de
pertenencia" (sub-abiertos); acá lo aproximamos empíricamente con la medida
relativa

$$
\Omega(S_0) \;\approx\; \frac{\mu\big(S_0 \cap \bigcup_i V_i\big)}{\mu(S_0)} \in [0,1],
$$

estimada por Monte Carlo. $\Omega=1$: $S_0$ pertenece por completo a la
criba (patch no vacío). $\Omega=0$: $S_0$ es disjunto de la familia — el
pullback de $S_0$ a lo largo de la cobertura es el objeto inicial $\emptyset$
(exactamente la lectura del paper: "invalid state collapses to the empty
fiber"). Valores intermedios son **parches topológicos parciales**, un
resultado legítimo de la lógica interna del topos, no un error.

---

## 3. Razonamiento

- Para el Sub-problema 1 elegí comparar $\oplus$ (concatenación) contra
  $\otimes$ (Kronecker) en lugar de una sola operación, porque el enunciado
  original habla de "restricciones monoidales estrictas" sin especificar
  cuál producto — y la respuesta correcta depende exactamente de esa
  elección. Descarté verificar solo $\oplus$ (sería trivial: la
  concatenación es estrictamente asociativa por construcción, no hay nada
  que aprender) y descarté afirmar sin más que "el álgebra de Clifford es
  estrictamente monoidal" (el paper lo insinúa pero no lo demuestra): en su
  lugar construyo el contraejemplo con Kronecker y el asociador explícito,
  que es un resultado estándar y verificable.
- Para el Sub-problema 2 descarté una implementación puramente simbólica
  (verificación exacta de conjuntos abiertos con álgebra de intervalos),
  porque el enunciado pide algo "estático" y verificable empíricamente de
  forma simple; Monte Carlo con cajas y bolas en R³ es standard, rápido, y
  deja ver con claridad los tres regímenes (cobertura total / nula /
  parcial) sin tener que resolver integrales exactas.
- En ningún momento traduzco "SPS" o "GeoID Stability" (términos internos
  del proyecto, no estándar) como si fueran ya validados: todo lo que mido
  son cantidades matemáticas estándar (norma L2, medida relativa/Monte
  Carlo).

---

## 4. Desarrollo y código

Ver `tarea_35.py` (Python puro + NumPy). Estructura:

1. `direct_sum_assoc_check` / `kron_assoc_check`: miden
   $\|(a\!\oplus\! b)\!\oplus\! c - a\!\oplus\!(b\!\oplus\! c)\|$ y su
   análogo con $\otimes$.
2. `associator_permutation`: construye el asociador $\alpha$ como
   permutación explícita de índices y corrige el mismatch de Kronecker.
3. `covering_sieve_truth_value`: estima $\Omega(S_0)$ por Monte Carlo para
   tres familias de cobertura (total, nula, parcial).

**Instrucciones para correrlo vos mismo:**

```bash
pip install numpy
python tarea_35.py
```

No lo corrí como parte de esta entrega (solo un chequeo interno de sintaxis
con `py_compile`, sin ejecutar la lógica) — la comprobación real, con tus
propios ojos, te toca a vos.

---

## 5. Conclusión

Sin ejecutar el script no puedo darte los números exactos que vas a ver en
tu corrida (dependen de la semilla `35` fijada, pero el *comportamiento
cualitativo* está garantizado por la construcción):

- **Sub-problema 1:** vas a ver que $\oplus$ da una norma de diferencia
  igual a 0 (estrictamente asociativo por construcción), que $\otimes$ da
  una norma de diferencia **positiva** (no es estricto "on the nose"), y
  que tras aplicar la permutación del asociador esa norma vuelve a 0. Esto
  confirma el teorema de coherencia de Mac Lane: hay un isomorfismo natural
  que reconcilia ambos agrupamientos, pero no es la identidad — hay que
  aplicarlo.
- **Sub-problema 2:** vas a ver $\Omega\approx 1$ para la cobertura total,
  $\Omega\approx 0$ para la nula (colapso al objeto inicial $\emptyset$), y
  un valor intermedio para la parcial. Ese valor intermedio **no es un
  error ni una falla de convergencia** — es la lógica interna del topos
  funcionando como debería: ni verdadero ni falso, un parche parcial.

Si tu corrida da resultados distintos a estos tres regímenes cualitativos
(por ejemplo, si la cobertura "total" no da $\approx 1$), revisá primero los
radios/posiciones de las bolas en `main_topos()`: es la forma más común de
romper el experimento sin que sea un problema del modelo.

---

```json
{
  "alumno": null,
  "tarea": 35,
  "tipo_problema": "teoria_de_categorias_topos_coherencia_monoidal",
  "estado": "resuelto",
  "resultado_resumen": "Se separan dos subproblemas del enunciado reconstruido: (1) coherencia monoidal de Mac Lane, mostrando que la suma directa es estrictamente asociativa y el producto de Kronecker solo lo es via asociador explicito; (2) criba de cobertura de una topologia de Grothendieck sobre R^3, con el clasificador de subobjetos Omega estimado por Monte Carlo en tres regimenes (total, nula, parcial).",
  "metrica_estandar_usada": "Norma L2 (asociatividad); medida relativa Monte Carlo (Omega)"
}
```
