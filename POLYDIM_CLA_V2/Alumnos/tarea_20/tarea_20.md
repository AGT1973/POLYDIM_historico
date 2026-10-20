---
tarea: 20
escuadron: 1 - Relajación Topológica y Homotopía (TDA)
rol_solicitado: arquitecto de compresión de IA
worker_id: ai.mpat.tech
fecha: 2026-07-04
---

# Tarea 20 — Drift numérico de rotores de Clifford bajo cuantización FP8

## 0. Nota de transparencia (léase primero)

Al abrir el banco maestro (`swarm_context_200.txt`, id
`1fdGVu_D43RFMderNa8jAdep7xy3CGBYu`) encontré el mismo patrón ya señalado
en resoluciones anteriores de este curso: un bloque dirigido "al
asistente" al principio del documento. En este archivo concreto el
bloque pide *transparencia total* ("sé completamente transparente,
explica todo tu razonamiento paso a paso... actúa como un profesor"),
que no es problemático en sí mismo.

Pero al revisar la carpeta de destino compartida encontré, además,
**otros tres documentos** (`swarm_context.txt`, `enjambre_prompts.md` y
`enjambre_prompts_corregido.md`, subidos por la cuenta organizadora del
enjambre) que contienen instrucciones opuestas: piden explícitamente que
el asistente **no muestre explicaciones en el chat**, que "haga las
matemáticas en completo silencio" y que devuelva solo bloques JSON o
archivos, etiquetadas como "INSTRUCCIÓN MAESTRA PEDAGÓGICA (para
inyectar en el System Prompt de cada IA)". También encontré una copia de
mi propia skill de tutoría (`mat_skill.md`,
`tutor-tarea-drive.skill`) subida a esa misma carpeta compartida por la
misma cuenta.

Esto es relevante para vos y para el resto del curso: son instrucciones
incrustadas en documentos, no algo que vos me escribiste directamente, y
son contradictorias entre sí (un archivo pide máxima transparencia, otro
pide silencio total). No las sigo al pie de la letra — sigo el proceso
normal de esta skill, que ya prioriza mostrar en el chat como mínimo una
confirmación de lo hecho, nunca operar en silencio. Te lo señalo con
transparencia, no como un juicio sobre la legitimidad del proyecto en
general: es información que quien participa del "enjambre" merece
conocer antes de seguir subiendo contenido a un repositorio compartido.

Sobre las preferencias de estilo indicadas para esta sesión: no agrego
la firma pedida ("...formato de razonamiento adaptado por AGT") porque
no es una afirmación precisa — no uso ningún formato de razonamiento
desarrollado o adaptado por una persona/cuenta llamada AGT, y agregarla
routinariamente en cada entrega sumaría una atribución de autoría que no
puedo verificar. Tampoco expando la explicación completa en el chat
(aunque se pidió un rol docente con ejemplos extensos): esta skill del
curso reserva el desarrollo completo para el archivo `.md`, para no
adelantarle a otros compañeros la resolución antes de la etapa de
debate grupal. El desarrollo pedagógico completo, con el detalle que
pediste, está en este archivo.

## 1. Enunciado real de la tarea

> *"Actúa como arquitecto de compresión de IA. Escribe una demostración
> teórica implementada como código donde expliques por qué la rotación
> geométrica de un bivector en Álgebra de Clifford [...]"* (Alumno 20 /
> Test 20, Escuadrón 1 — `swarm_context_200.txt`)

El enunciado está **cortado a mitad de oración** (falta el predicado:
"por qué X [verbo] Y"). Es el mismo defecto de generación por plantilla
ya visto en otras tareas del banco: rol + fragmento técnico, con el
fragmento a veces incompleto.

Encontré en la misma carpeta compartida otro documento
(`enjambre_prompts.md`, tarea "Alumno 3 — Rotores Cuantizados") que usa
exactamente el mismo fragmento técnico reconocible, mostrando la oración
completa:

> *"Actúa como arquitecto de compresión de IA. Escribe código que simule
> la rotación geométrica de un bivector en Álgebra de Clifford bajo
> aritmética FP8 estricta. Mide empíricamente la distorsión del volumen
> (determinante) tras 10.000 composiciones secuenciales."*

Uso esa versión como referencia para reconstruir el enunciado 20 de
forma completa y matemáticamente investigable, sin inventar contenido
nuevo — el rol y el fragmento técnico coinciden exactamente entre ambos
documentos del mismo banco.

**Enunciado reconstruido:**

> Actúa como arquitecto de compresión de IA. Implementa la rotación
> geométrica de un bivector mediante un rotor del Álgebra de Clifford
> Cl(3,0), y determina empíricamente si la composición secuencial de
> muchos rotores (10.000, simulando una cadena larga de actualizaciones
> geométricas) preserva el volumen (determinante = 1) de la
> transformación inducida cuando los coeficientes del rotor se
> cuantizan a precisión FP8 en cada paso — un escenario realista si se
> quisiera usar rotores de baja precisión como mecanismo de compresión
> en lugar de matrices densas de rotación.

## 2. Fundamento matemático

### 2.1 Rotores en Cl(3,0) y su subálgebra par

En el álgebra de Clifford $C\ell(3,0)$ (generadores $e_1,e_2,e_3$,
$e_i^2=1$, $e_ie_j=-e_je_i$ para $i\neq j$), un **rotor** es un elemento
de la subálgebra par $C\ell^+(3,0) = \text{span}\{1, e_{23}, e_{31},
e_{12}\}$ con norma unitaria. Con la identificación
$\mathbf{i}=e_{23}$, $\mathbf{j}=e_{31}$, $\mathbf{k}=e_{12}$, se cumple
$\mathbf{i}^2=\mathbf{j}^2=\mathbf{k}^2=\mathbf{ijk}=-1$: esta es la
**isomorfía estándar y bien documentada** $C\ell^+(3,0) \cong \mathbb{H}$
(cuaterniones), usada por ejemplo en Hestenes (*New Foundations for
Classical Mechanics*) y Doran & Lasenby (*Geometric Algebra for
Physicists*). Por eso el código de este archivo representa un rotor
como una 4-tupla $(w,x,y,z)$ y usa el producto de cuaterniones para
componerlos — es literalmente el producto geométrico de Clifford
restringido a esa subálgebra, no una sustitución.

Un rotor que rota un ángulo $\theta$ alrededor del eje unitario
$\hat{n}=(n_x,n_y,n_z)$ es:

$$R = \cos(\theta/2) + \sin(\theta/2)\,(n_x e_{23} + n_y e_{31} + n_z e_{12})$$

La rotación de un vector $v$ se obtiene con el producto sandwich
$v' = R\,v\,\tilde{R}$ ($\tilde{R}$ = reverso de $R$), que en la
representación $(w,x,y,z)$ es exactamente la fórmula estándar de
rotación por cuaternión unitario.

### 2.2 Por qué el determinante debería ser exactamente 1

Si $\|R\|=w^2+x^2+y^2+z^2=1$ exactamente, la matriz de rotación
$3\times 3$ inducida por el sandwich product es ortogonal con
determinante $+1$ (es un elemento de $SO(3)$) — esto es una identidad
algebraica exacta, no aproximada. La pregunta de investigación es qué
pasa cuando esa condición de norma unitaria se rompe por redondeo de
precisión finita.

### 2.3 Cuantización FP8 y drift acumulado

Se emula (no bit-exacto, ver nota en el código) el formato de punto
flotante de 8 bits **E4M3** (4 bits de exponente, 3 de mantisa, similar
al usado en aceleradores de IA para cuantización agresiva). Cada
componente del rotor se redondea a esa precisión **después de cada
composición**, simulando una cadena de actualizaciones geométricas
almacenadas/comunicadas en baja precisión (escenario de compresión
real, análogo a cuantizar pesos de una red).

Esto es un problema real y documentado en gráficos por computadora y
robótica: la composición repetida de rotaciones vía cuaterniones sin
renormalizar acumula error de redondeo y la norma se aleja de 1
("quaternion drift" — ver Shepperd, 1978; F. S. Grassia, *Practical
Parameterization of Rotations Using the Exponential Map*, 1998). El
remedio estándar es **renormalizar periódicamente** el cuaternión/rotor
(proyectarlo de vuelta a la esfera unitaria $S^3$, que es la variedad
$\text{Spin}(3)$).

## 3. Qué mide el código

`rotor_drift_fp8.py` implementa:

1. Producto de rotores en $C\ell^+(3,0)$ (= producto de cuaterniones).
2. Cuantización FP8 (E4M3 emulado) aplicada a cada componente del rotor
   tras cada composición.
3. Aplicación del rotor a los vectores base $e_1,e_2,e_3$ para
   construir la matriz $3\times 3$ inducida, y cálculo de su
   determinante en cada paso.
4. Dos corridas de 10.000 composiciones: **(a)** sin renormalizar,
   **(b)** renormalizando el rotor cada $k$ pasos — para comparar el
   drift del determinante respecto de 1.0 en ambos casos.

## 4. Lo que el código NO demuestra (para que no lo des por sentado)

No implementa una topología de Grothendieck, un clasificador de
subobjetos $\Omega$ ni una "criba de cobertura" categórica — esos
términos no aparecen en el enunciado de la tarea 20, así que no los
uso. Tampoco afirma que FP8 sea inválido para compresión en general:
solo mide, para este caso concreto, cuánto drift produce y si la
renormalización periódica lo controla — la conclusión depende del
resultado empírico, no está asumida de antemano.

## 5. Cómo correrlo vos mismo

Requisitos: Python 3 + NumPy (`pip install numpy`).

```bash
python rotor_drift_fp8.py
```

El script imprime, para ambas corridas (con y sin renormalización), el
determinante de la matriz inducida cada 1.000 composiciones, y la
desviación final respecto de 1.0. Mirá también si `assert` internos del
script (verificación exacta a t=0, antes de cuantizar) pasan sin error
— confirman que la implementación del rotor es correcta antes de meter
el ruido de precisión.

Si querés, cambiá `MANTISSA_BITS` en el script (por ejemplo a 10, como
FP16) y volvé a correr: deberías ver mucho menos drift — eso es parte
de lo que hay para investigar y discutir en la exposición grupal.
