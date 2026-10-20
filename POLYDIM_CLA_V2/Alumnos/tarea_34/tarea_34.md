# Tarea 34 — POLYDIM V2 (Escuadrón 7: Restricciones O-Mínimales y Análisis Estático / SMT)

## 1. Enunciado original (banco maestro, Test 34)

> "34 (Test 34): Actúa como matemático numérico. Integra la librería Z3 (SMT solver)
> para demostrar cómo resolver restricciones sobre un subespacio latente R^3 de forma
> estática, validando si pertenece a la criba de cobertura del topos."

A diferencia de otros tests de este banco, este enunciado está gramaticalmente completo
(no hace falta reconstruirlo por falta de verbo); sí requiere formalizar con precisión qué
significa "criba de cobertura del topos" en términos verificables por Z3, porque el paper
la introduce solo de manera informal (Sección 2.1 / Apéndice V2.0, punto 1).

## 2. Modelo matemático

### 2.1 El sitio, la familia de cobertura y la criba

Se modela un fragmento concreto del sitio C de la Sección 2.1 del paper:

- **Objeto del sitio:** U = [-2,2]^3 ⊂ R^3 (un "subespacio semántico").
- **Familia candidata a cubrir U:** dos bolas
  U1 = B(c1, r1), c1=(-1,-1,-1), r1=1.5,
  U2 = B(c2, r2), c2=(1,1,1), r2=1.5,
  interpretadas como dos "parches" de validez local (p. ej. dos conjuntos de reglas de
  negocio compatibles cada uno con una franja distinta del espacio latente).
- **Axioma de cobertura de Grothendieck** que debe cumplir la familia para llamarse
  cobertura de U: U ⊆ U1 ∪ U2.
- **Criba de cobertura** (sieve) asociada: el subfuntor/subobjeto
  S = U1 ∪ U2 ⊆ U. Un punto x "pertenece a la criba" si x ∈ S.
- **Subobject Classifier Ω (Sección 2.1 y Apéndice, punto 1):** para x ∈ U,
  el valor de verdad de "x es válido" es la clasificación de x ∈ S vs. su
  pullback al objeto inicial ∅ (si x ∉ S).

### 2.2 El objeto de estudio: subespacios latentes ESTÁTICOS

El enunciado pide validar si un **subespacio** (no un punto aislado) pertenece a la
criba, "de forma estática" — es decir, decidir la pertenencia para **todo** punto del
subespacio de una vez, sin muestrear ni evaluar caso por caso. Esto es exactamente lo que
un SMT solver permite: en vez de preguntar "¿x ∈ S?" para un x concreto, se
pregunta, para un subespacio Q ⊆ U:

φ_out(Q) := ∃x∈Q. x ∉ S       φ_in(Q) := ∃x∈Q. x ∈ S

y se usa la tabla de decisión:

| φ_out | φ_in | Veredicto |
|---|---|---|
| UNSAT | SAT | Q ⊆ S — **válido global** |
| SAT | UNSAT | Q ∩ S = ∅ — **inválido global** (colapsa a ∅) |
| SAT | SAT | **mixto** — subobjeto propio Q ∩ S ⊊ Q |

El caso "mixto" es el punto matemáticamente interesante: **no existe un valor de verdad
clásico (0 o 1) que describa correctamente a Q completo**. Es precisamente la situación
que la Sección 2.2 del paper describe como ausencia de la ley del tercio excluido: la
respuesta correcta no es "sí" ni "no", sino el subobjeto abierto Q ∩ S mismo. Z3, como
solver clásico, no "razona intuicionísticamente" — lo que hace es decidir, con lógica
clásica de primer orden, las dos preguntas de satisfacibilidad por separado; es la
**combinación de ambos resultados clásicos** la que reconstruye correctamente el tercer
valor de verdad no-booleano del topos. Esta distinción se deja explícita para no sobre-
afirmar capacidades de Z3 que no tiene.

## 3. Razonamiento

- Se representan U1, U2 como bolas (restricciones cuadráticas) y U, Qi como cajas
  (restricciones lineales) porque son las formas más simples que permiten a Z3 trabajar en
  aritmética real no lineal (`QF_NRA`) sin ambigüedad, y porque modelan de forma natural
  "parches de validez" con geometría no trivial (a diferencia de un puro poliedro, una
  bola no es unión de semiespacios triviales, lo que hace el ejercicio menos artificial).
- Antes de clasificar subespacios, se verifica si {U1,U2} es siquiera una cobertura
  válida de U (axioma de Grothendieck). Este paso no estaba pedido literalmente pero es
  un prerrequisito lógico: si la familia no cubre U, "la criba de cobertura" a la que se
  refiere el enunciado tiene agujeros por construcción, y conviene mostrarlo explícitamente
  en vez de asumir tácitamente que la cobertura es total.
- Se eligen tres subespacios de prueba Q1,Q2,Q3 diseñados a propósito para caer en los
  tres regímenes de la tabla (uno claramente adentro, uno claramente afuera, uno a caballo
  del borde de U1), para que la demostración cubra los tres casos posibles y no solo el
  caso trivial.

## 4. Desarrollo y código

Ver `tarea_34.py` (funcional, usa `z3-solver`). Resumen:

1. Define U, U1, U2 como restricciones de Z3 sobre variables reales x,y,z.
2. `check_global_coverage()`: pregunta si existe x ∈ U con x ∉ U1 ∪ U2
   (axioma de cobertura).
3. `classify_subspace(Q)`: corre φ_out(Q) y φ_in(Q) y
   aplica la tabla de decisión.
4. Corre todo para Q1 (caja pegada al centro de U1), Q2 (esquina lejana de U) y
   Q3 (caja que cruza el borde esférico de U1).

### Corrida interna de sanity-check (no es la validación final — corré vos el script)

```
1) ¿La familia {U1, U2} cubre efectivamente el sitio U?
NO es un cubrimiento válido de U. Z3 encontró un punto de U fuera
de U1 ∪ U2: [y = 0, x = 0, z = 0]

2) Clasificacion de Q1, Q2, Q3:
[Q1 (caja cerca del centro de U1)] fuera: unsat   dentro: sat  -> VALIDO GLOBAL (Q ⊆ criba)
[Q2 (esquina lejana de U)]         fuera: sat     dentro: unsat -> INVALIDO GLOBAL (Q ∩ criba = ∅)
[Q3 (caja que cruza el borde de U1)] fuera: sat   dentro: sat  -> MIXTO (subobjeto propio)
```

El witness que da Z3 para el punto no cubierto es el origen (0,0,0): su distancia a
c1=(-1,-1,-1) y a c2=(1,1,1) es √3 ≈ 1.732 > 1.5 = r1=r2, así que
efectivamente cae fuera de ambas bolas — se puede verificar a mano sin el solver, lo cual
sirve como control de cordura del propio resultado de Z3.

## 5. Conclusión

**Estado: `resuelto`.**

Z3 permite decidir de forma estática — es decir, para un subespacio completo, no solo para
puntos individuales muestreados — si ese subespacio pertenece enteramente a una criba de
cobertura, queda enteramente afuera, o queda en una situación mixta que no admite un valor
de verdad clásico único. Los tres regímenes se observaron efectivamente en los tres
subespacios de prueba, y además se detectó (subproducto necesario del modelo) que la
familia de cobertura elegida **no** cubre todo el sitio U: hay un agujero real (el
origen, y por simetría toda una región alrededor de él), lo cual en el lenguaje del paper
corresponde a estados semánticos que colapsarían al objeto inicial ∅ en vez de
ser absorbidos por algún parche válido.

**Aclaración importante para no sobre-interpretar el resultado:** Z3 no está "haciendo
lógica intuicionista"; sigue siendo un solver de lógica clásica de primer orden. Lo que
reconstruye la semántica no-booleana del topos es la **combinación** de dos resultados
clásicos (SAT/UNSAT de φ_out y φ_in), no una sola llamada
al solver. Esto es consistente con la Sección 9.2 del paper ("Compile-Time Mac Lane
Strictification"): la coherencia categórica se garantiza de forma estática sobre el grafo
de restricciones, no dentro del propio motor de decisión, que sigue siendo clásico.

**Métrica/técnica estándar usada:** decisión de satisfacibilidad en aritmética real no
lineal (`QF_NRA` vía Z3); axioma de cobertura de Grothendieck como condición de
satisfacibilidad; clasificación en tres regímenes (⊆, ∩=∅, mixto) como sustituto
verificable del valor de verdad en Ω.

## 6. Instrucciones para correr el código vos mismo

```bash
pip install z3-solver --break-system-packages   # si no lo tenés
python3 tarea_34.py
```

Como ejercicio de verificación adicional: cambiá los centros/radios de U1,U2 para que
sí formen una cobertura válida de U (por ejemplo agrandando los radios), volvé a correr
`check_global_coverage()`, y confirmá que el resultado pasa a `unsat` (sin agujeros). También
podés mover Q3 un poco para que quede completamente adentro o completamente afuera, y
confirmar que el veredicto cambia de "mixto" a uno de los otros dos casos.

```json
{
 "tarea": 34,
 "tipo_problema": "verificación formal estática (SMT) de pertenencia de un subespacio a una criba de cobertura de un topos de Grothendieck",
 "estado": "resuelto",
 "resultado_resumen": "Z3 decide, para subespacios completos (no solo puntos), si caen enteramente dentro de la criba, enteramente afuera, o en un régimen mixto sin valor de verdad clásico único; se verificó además que la familia de cobertura elegida no cubre todo el sitio U, exhibiendo un agujero real.",
 "metrica_estandar_usada": "Satisfacibilidad en aritmética real no lineal (QF_NRA); axioma de cobertura de Grothendieck; clasificación en tres regímenes (⊆, ∩=∅, mixto)"
}
```
