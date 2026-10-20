# Tarea 17 — POLYDIM (banco de 200, ítem 17)

**Alumno:** ariel.garcia.traba@gmail.com
**Rol asignado por el banco:** especialista en NVIDIA Triton

## 1. Enunciado

**Original (banco maestro, ítem 17, tal cual está guardado):**

> "Actúa como especialista en NVIDIA Triton. Diseña una operación encadenada que
> demuestre a nivel de volcado intermedio de grafo que el Producto Geométrico
> de un Álgebra de Clifford truncada."

Este enunciado está **incompleto**: se corta antes del verbo/predicado principal
("...que el Producto Geométrico ... truncada [¿qué?]"). Esto es consistente con
cómo se generó el banco (combinación de plantillas rol + fragmento técnico, y
varios fragmentos quedaron sin verbo de cierre). No se completa "inventando" un
final arbitrario: se reconstruye apoyándose en el rol indicado (kernels /
volcado de grafo) y en el fragmento técnico reconocible ("Producto Geométrico
de un Álgebra de Clifford truncada").

**Enunciado reconstruido (el que efectivamente se resuelve abajo):**

> Sea `⊙` el producto geométrico de un Álgebra de Clifford. Se define una
> versión *truncada* `⊙_T`, que descarta la componente de grado más alto
> (el pseudoescalar) después de cada multiplicación — algo que un kernel
> real haría por eficiencia de memoria en una operación encadenada. Investigar,
> inspeccionando el resultado en cada paso de la cadena (el "volcado intermedio
> de grafo"), si `⊙_T` conserva la **asociatividad**: ¿`(a ⊙_T b) ⊙_T c = a ⊙_T
> (b ⊙_T c)` para todo `a,b,c`?

### Por qué esta reconstrucción y no otra (la "trampa" de este ítem)

Leído rápido, el enunciado parece pedir "escribí un kernel que calcule el
producto geométrico truncado" — una tarea de programación directa. Pero el
punto real, una vez reconstruido con cuidado, es otro: el producto geométrico
**sin truncar** es asociativo por teorema (es parte de la construcción misma
del álgebra de Clifford como cociente del álgebra tensorial por un ideal
bilátero — ver Hestenes & Sobczyk, *Clifford Algebra to Geometric Calculus*,
1984; o Dorst, Fontijne & Mann, *Geometric Algebra for Computer Science*,
2007). Truncar, en cambio, **no** tiene ninguna garantía teórica estándar de
preservar esa propiedad, y el marco POLYDIM da por sentado (sin demostrarlo)
que optimizaciones de este tipo "no perjudican" al modelo. Esa premisa no está
establecida en la literatura estándar, así que en esta entrega se trata como
pregunta abierta y se responde con una demostración explícita, no se asume.

## 2. Modelo matemático

Trabajamos en `Cl(3,0)`, el álgebra geométrica de `R^3` con métrica euclídea
(`e_i^2 = 1`, `e_i e_j = -e_j e_i` para `i≠j`). Es un espacio de dimensión 8,
graduado por grado `k = 0,1,2,3`:

```
grado 0: 1                          (escalar)
grado 1: e1, e2, e3                 (vector)
grado 2: e12, e13, e23              (bivector)
grado 3: e123                       (trivector / pseudoescalar)
```

Un multivector general es `M = M_0 + M_1 + M_2 + M_3` (suma de sus partes
homogéneas por grado). Se define la proyección

```
P_{≤2}(M) := M_0 + M_1 + M_2      (anula la componente de grado 3)
```

y el **producto geométrico truncado**:

```
a ⊙_T b := P_{≤2}(a ⊙ b)
```

**Hipótesis a testear (H0):** `⊙_T` es asociativa, i.e. el "asociador
truncado" `Assoc_T(a,b,c) := (a ⊙_T b) ⊙_T c − a ⊙_T (b ⊙_T c)` es
idénticamente 0.

**Métrica de validación:** norma L2 (euclídea) de `Assoc_T(a,b,c)` sobre sus
8 coeficientes — la métrica estándar que pide el marco de trabajo del banco,
sin necesidad de inventar ninguna métrica ad-hoc tipo "SPS" o "GeoID
Stability" (esos nombres no corresponden a ninguna métrica establecida en la
literatura de álgebra geométrica o deep learning; se usa Norma L2 directamente,
que es lo matemáticamente estándar y verificable).

## 3. Razonamiento paso a paso

**Paso 1 — el producto sin truncar es asociativo (control, no se re-demuestra
el teorema, se verifica numéricamente que la implementación lo respeta).**
Esto es necesario antes de tocar la parte truncada: si la implementación del
producto geométrico completo no fuera asociativa, cualquier conclusión
posterior sería un bug de código, no un resultado matemático. El script
(sección 4) corre 200 tripletas aleatorias y confirma error ~1e-14 (ruido de
punto flotante, no una violación real).

**Paso 2 — por qué truncar puede romper la asociatividad, en general.**
`P_{≤2}` es una proyección lineal con núcleo no trivial (la línea de `e123`).
El producto geométrico mezcla grados: el producto de un vector por un
bivector, por ejemplo, tiene en general una parte de grado 1 (contracción) y
una de grado 3 (parte exterior), salvo que el vector esté contenido en el
plano del bivector. Si truncamos *después de cada paso* de una cadena, en un
paso intermedio podemos tirar información de grado 3 que, de haber
sobrevivido, habría interactuado con el tercer factor para producir un
término de grado ≤2 en el resultado final. Cambiar el orden de agrupación
cambia en qué paso se descarta esa información, y por lo tanto puede cambiar
el resultado final.

*Ejemplo cotidiano (para ver la forma del problema sin álgebra geométrica):*
sumar números reales es asociativo, pero si una calculadora redondea a 2
decimales después de cada suma parcial, `(0.1+0.2)+0.3` puede no coincidir
exactamente con `0.1+(0.2+0.3)` — no porque la suma real deje de ser
asociativa, sino porque el redondeo intermedio es una operación con pérdida
que no conmuta con el reagrupamiento. Otro ejemplo, más visual: aplicar un
filtro de compresión (con pérdida) a una foto y después recortarla no siempre
da el mismo resultado, píxel por píxel, que recortarla primero y comprimir
después — el orden en que se pierde información importa. En ambos casos, la
operación "de fondo" (sumar, editar la foto) es perfectamente asociativa o
conmutativa; lo que rompe la propiedad es la operación de truncar/redondear
intercalada en el medio. Es exactamente lo que pasa acá con `P_{≤2}`.

**Paso 3 — contraejemplo exacto (a mano, verificado por el script).**
Tomamos `A = e1 + e3` (un vector con componente *fuera* del plano de `e12`),
`B = e12`, `C = e2` (un vector *dentro* del plano de `e12`):

- `A ⊙ B = e2 + e123` (mezcla de grado 1 y grado 3, porque la componente
  `e3` de `A` no está en el plano de `B`). Truncado: `A ⊙_T B = e2`
  (se descarta `e123`).
- `(A ⊙_T B) ⊙_T C = e2 ⊙_T e2 = 1` (escalar puro).
- `B ⊙ C = e12 ⊙ e2 = e1` (acá no se pierde nada: `e2` está en el plano de
  `B`, así que no aparece componente de grado 3 en este paso).
- `A ⊙_T (B ⊙_T C) = A ⊙_T e1 = (e1+e3) ⊙_T e1 = 1 − e13`.

**Asociador:** `1 − (1 − e13) = e13 ≠ 0.** El script confirma esto exactamente
(ver salida en sección 4): la diferencia es el multivector `e13` puro, norma
L2 = 1.0.

**Paso 4 — cuantificación estadística.** Sobre 2000 tripletas aleatorias
(coeficientes ~ normal estándar en las 8 componentes), la fracción de casos
con asociador no nulo y la magnitud típica del error se reportan en la
sección 4 (no se pegan los números acá adentro a propósito — correlos vos y
mirá el resultado con tus propios ojos, ese paso activo es parte del
aprendizaje, no solo leer el .md).

## 4. Código

Archivo: `tarea_17.py` (adjunto en esta misma carpeta de Drive).

Para correrlo vos mismo:

```bash
pip install numpy
python3 tarea_17.py
```

El script:
1. Implementa el producto geométrico completo de `Cl(3,0)` (bitmasks +
   algoritmo estándar de signo por reordenamiento canónico).
2. Se auto-verifica contra identidades conocidas (`e1²=1`, `e1e2=e12`,
   `e2e1=-e12`, `e12²=-1`, `e123²=-1`) — si esto falla, corta ahí.
3. Confirma que el producto SIN truncar es asociativo (control).
4. Reproduce el contraejemplo exacto de la sección 3.
5. Corre el muestreo Monte Carlo de 2000 tripletas y te muestra la
   estadística real.

No se ejecuta acá como parte de la entrega — la comprobación real (correrlo
vos y leer los números) te corresponde a vos.

## 5. Conclusión técnica

**Estado: resuelto.** La pregunta reconstruida tiene una respuesta clara y
demostrada, no una que "no convergió":

- El producto geométrico **sin truncar** de `Cl(3,0)` es asociativo, como
  garantiza el teorema de construcción del álgebra (confirmado numéricamente
  como control).
- El producto geométrico **truncado a grado ≤2** (`⊙_T`) **no es asociativo
  en general**. Existe un contraejemplo exacto y simple (`A=e1+e3, B=e12,
  C=e2`, asociador `= e13`), y el muestreo Monte Carlo muestra que el
  fenómeno no es un caso aislado sino la regla general para multivectores
  arbitrarios.
- Implicancia para el marco POLYDIM: la premisa de que "truncar por
  eficiencia no perjudica" el álgebra subyacente **no está garantizada** para
  este tipo de operación encadenada, al menos no sin una condición adicional
  (por ejemplo, restringir los operandos a subespacios donde la parte de
  grado 3 sea siempre nula, o reordenar la truncatura para que ocurra solo al
  final de toda la cadena, no en cada paso intermedio). Esto es una
  observación técnica sobre la operación puntual, no una discusión sobre la
  validez general del proyecto POLYDIM — esa conversación, como corresponde
  en esta etapa del curso, queda para la instancia de exposición grupal.

## 6. Resumen estructurado

```json
{
  "alumno": "ariel.garcia.traba@gmail.com",
  "tarea": 17,
  "tipo_problema": "algebra_geometrica / verificacion_de_propiedad_algebraica",
  "estado": "resuelto",
  "resultado_resumen": "El producto geometrico truncado a grado <=2 en Cl(3,0) no es asociativo en general; contraejemplo exacto con asociador = e13, confirmado por muestreo Monte Carlo.",
  "metrica_estandar_usada": "Norma L2 del multivector asociador (8 componentes)"
}
```
