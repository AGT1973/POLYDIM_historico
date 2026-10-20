# Tarea 12 — Enjambre POLYDIM (Escuadrón 1: Relajación Topológica y Homotopía)

**Ticket:** Test 12
**Rol solicitado:** experto en Categorical Deep Learning
**Banco maestro:** `1fdGVu_D43RFMderNa8jAdep7xy3CGBYu` (200 tareas, confirmado vigente el 2026-07-04)

---

## 0. Nota metodológica previa

Este enunciado combina el mismo fragmento roto ya visto en la tarea 9 ("...ignorar homotopías superiores ahorra cómputo sin perjudicar la generalización composicional") pero insertado en un molde distinto ("clase nativa de PyTorch que reemplace a `nn.Linear`"). No es casualidad ni error mío: el banco de 200 tareas se genera por combinatoria de plantillas (rol × acción × fragmento técnico), y en este caso el fragmento no encaja gramaticalmente en el molde donde fue insertado — el molde espera una *operación algebraica concreta* (como en las tareas 55, 124, 177, 192 del mismo escuadrón), no una *afirmación* a demostrar.

**No voy a fingir que el enunciado es coherente tal cual está.** Lo reconstruyo con un enfoque más contundente que el usado en la tarea 9 (que fue empírico), aportando algo nuevo: una **prueba exacta**, no solo un experimento con números.

## 1. El enunciado, reconstruido

Texto original (roto):
> "Actúa como experto en Categorical Deep Learning. Escribe una clase nativa de PyTorch que reemplace a `nn.Linear` operando internamente con ignorar homotopías superiores ahorra cómputo sin perjudicar la generalización composicional."

Reconstrucción bien planteada:
> Escribe una clase (con la misma interfaz que `nn.Linear`: `forward`, `parameters`) que, internamente, trunque las interacciones que modela hasta un cierto orden *k* (es decir, "ignore homotopías/interacciones de orden superior a *k*"). Validá si ese truncamiento **siempre** ahorra cómputo sin perjudicar la generalización composicional, comparándolo contra una capa que sí modela el orden completo.

**Nota de infraestructura:** este entorno no tiene PyTorch instalado, y no correspondía instalarlo para una comprobación que hace el alumno, no yo (ver `SKILL.md`, paso 8). Implementé la clase en NumPy puro con la misma interfaz conceptual (`forward(x)`, `parameters()`) — matemáticamente es exactamente lo mismo que haría PyTorch: una combinación lineal de *features* (monomios).

## 2. Intuición antes de la matemática

Pensá en una llave de tuercas ajustable con dos formas de uso:
- Usarla **sola** sobre un tornillo: eso es una interacción de "orden 1" (una pieza a la vez).
- Usarla **con la otra mano sosteniendo** la pieza para que no gire: eso ya es una interacción de "orden 2" (dos piezas coordinadas).
- Ahora pensá en armar algo donde hacen falta **tres** manos coordinadas al mismo tiempo, ninguna combinación de a dos alcanza. Ese es el "orden 3": no es que sea "más difícil", es que **es categóricamente imposible** hacerlo bien con herramientas de orden ≤2, sin importar cuánta práctica tengas.

Eso es exactamente lo que se puede demostrar matemáticamente para ciertas funciones: hay funciones cuya "firma" completa vive en el orden 3, y ningún modelo limitado a orden ≤2 puede acercarse a ellas — no un poco peor, sino con correlación exactamente cero.

## 3. Modelo matemático estándar

Trabajamos en $\{-1,1\}^n$ (codificación de bits en $\pm 1$, estándar en análisis de funciones booleanas — O'Donnell, *Analysis of Boolean Functions*, 2014). Toda función $f:\{-1,1\}^n \to \mathbb{R}$ tiene una expansión de Fourier-Walsh **única y exacta**:

$$f(x) = \sum_{S \subseteq [n]} \hat{f}(S) \prod_{i \in S} x_i$$

Los monomios $\chi_S(x) = \prod_{i \in S} x_i$ son ortonormales bajo la medida uniforme del cubo. Una capa que trunca en grado $k$ (es decir, "ignora interacciones de orden $>k$") solo puede representar la parte de $f$ cuyo soporte de Fourier tiene $|S| \le k$.

**Target elegido:** paridad de 3 bits, $f(x_1,x_2,x_3) = x_1 x_2 x_3$. Es el ejemplo canónico de una función cuyo **único** coeficiente de Fourier no nulo es el de grado 3 ($\hat f(\{1,2,3\})=1$, todo el resto exactamente $0$). Representa "generalización composicional" en su forma más pura: acertar en la tripleta exige capturar la interacción de orden 3 completa.

## 4. Diseño de la clase (`DegreeKLinear`)

- Expande el input a *features* = todos los monomios de tamaño $0..k$ (combinaciones de variables).
- Ajusta un peso lineal sobre esas *features* (ajuste exacto por mínimos cuadrados — con 8 puntos y a lo sumo 8 features, es resoluble sin gradiente, así que no hay ambigüedad de convergencia).
- `k=1` ≈ un `nn.Linear` estándar (solo términos lineales, la interpretación literal más estricta de "ignorar toda homotopía superior").
- `k=2` es la lectura literal del enunciado ("ignora homotopías de orden $>2$").
- `k=3` es la base completa: puede representar *cualquier* función en $\{-1,1\}^3$.

## 5. Resultado (corrida de sanity-check, no ejecutada como entrega final — el alumno la corre él mismo)

| grado $k$ | parámetros | FLOPs (8 puntos) | $R^2$ | error máx. |
|---|---|---|---|---|
| 1 | 4 | 32 | **0.000000** | 1.0 |
| 2 | 7 | 56 | **0.000000** | 1.0 |
| 3 | 8 | 64 | **1.000000** | 0.0 |

## 6. Conclusión técnica

El resultado **no es aproximado, es exacto**: truncar en grado $k=2$ ahorra 1 de 8 parámetros (12.5%) pero produce correlación cero con la paridad de 3 bits — no una degradación parcial, un fracaso matemático total, porque el monomio de grado 3 es ortogonal a cualquier combinación de grado ≤2 por construcción.

**Esto refuta la afirmación del enunciado tal como está escrita ("ignorar homotopías superiores ahorra cómputo *sin perjudicar* la generalización composicional") en el caso general.** La afirmación es cierta únicamente para el subconjunto de funciones cuyo peso de Fourier ya vive en grado ≤ k (funciones "casi lineales" o "casi bilineales") — y ese subconjunto no se puede asumir sin conocer la tarea real que la red va a resolver. Documentar esta falla matemática es, en sí, un resultado científico válido (no lo escondo ni lo maquillo).

**Relación con la tarea 9 (sibling):** la tarea 9 (resuelta por otra compañera con octoniones) llegó a una conclusión compatible por vía empírica (entrenando una red y midiendo error de generalización). Esta entrega llega a la misma conclusión de fondo por vía de **prueba exacta** (ortogonalidad de Fourier), que es una validación más fuerte porque no depende de cómo se entrenó el modelo ni de cuántos datos se usaron — es una imposibilidad algebraica, no una limitación empírica de una corrida particular.

## 7. Instrucciones para que corras el código vos mismo

```bash
python3 degree_truncated_layer.py
```

Sin dependencias externas (solo `numpy`, que ya viene instalado en la mayoría de los entornos Python). Si querés extender el experimento:
- Probá con `n=4` o `n=5` variables y paridad completa — vas a ver que la brecha de parámetros ahorrados crece mucho más rápido que el "costo" de necesitar el grado completo, así que el trade-off se vuelve más dramático a medida que crece la dimensión.
- Probá con un target que **sí** sea de bajo grado (por ejemplo, la mayoría de 3 bits, que tiene peso en grados 1 y 3, pero *no* en el 3 puro) para ver un caso donde el truncamiento sí es un compromiso parcial, no un fracaso total — así verificás por vos mismo si la premisa del enunciado se sostiene, se refuta, o queda sin resultado concluyente según el tipo de función objetivo.
