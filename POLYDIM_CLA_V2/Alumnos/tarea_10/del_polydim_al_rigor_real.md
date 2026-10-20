# Del POLYDIM al Rigor Real
### Guía de 15 pilares matemáticos, diagnóstico de pseudociencia, e implementación verificada de rotores de Clifford

---

## Cómo usar este documento

Cada uno de los 15 conceptos se presenta en tres capas, pensadas para que avances de nivel universitario inicial a nivel superior:

1. **Intuición cotidiana** — una situación de la vida diaria o de programación básica.
2. **La matemática real** — la definición formal, tal como existe en la literatura, sin adornos.
3. **Contraste con POLYDIM** — qué afirma el paper sobre este concepto, y si esa afirmación se sostiene o es un salto sin demostración.

Al final de cada concepto hay una fila de **veredicto**: REAL (el concepto existe y se usa tal como se describe), INFLADO (el concepto es real pero la conclusión que se saca de él no está probada), o INVENTADO (la conexión no tiene sustento matemático conocido).

---

## PARTE 1 — Los 15 pilares matemáticos

### 1. Topos de Grothendieck

**Intuición cotidiana:** pensá en una app de mapas. En vez de decir "este local existe / no existe" (verdadero/falso), la app puede decir "este local es visible si estás en el nivel de zoom de barrio, pero no en el de zoom de país". La "verdad" depende del contexto (la "vista") desde la que mirás. Un topos formaliza justamente eso: verdades que dependen del contexto de observación.

**La matemática real:** un topos de Grothendieck es una categoría de haces (*sheaves*) sobre un sitio $(\mathcal{C}, J)$, donde $J$ es una topología de Grothendieck (una noción de "cobertura" de un objeto por otros objetos). Es una generalización de la categoría de conjuntos que permite tener una lógica interna distinta de la clásica. Es un objeto de estudio legítimo y muy activo (fundamento de la geometría algebraica moderna, desde Grothendieck en los años 60, y hoy con aplicaciones en teoría de tipos homotópica).

**Contraste con POLYDIM:** el paper dice que el espacio latente de una red neuronal "es" un topos de Grothendieck y que por eso "las restricciones lógicas se computan a la velocidad de la topología". Ningún paper de matemática o de ML publicado construye ese sitio $(\mathcal{C}, J)$ de forma explícita para un espacio de embeddings real, ni define qué cobertura corresponde a qué operación de red neuronal. Es una analogía de nombre, no una construcción.

**Veredicto:** REAL el concepto — INVENTADA la aplicación a redes neuronales tal como se describe.

---

### 2. Clasificador de subobjetos (Ω)

**Intuición cotidiana:** en un topos, en vez de una casilla de verificación con dos opciones (sí/no), tenés un "medidor" con más matices — como un semáforo con colores intermedios en vez de solo rojo/verde.

**La matemática real:** en un topos, $\Omega$ es un objeto que representa los "valores de verdad" internos, junto con un morfismo universal $\text{true}: 1 \to \Omega$ tal que cada subobjeto corresponde a un único morfismo característico hacia $\Omega$. En **Set**, $\Omega = \{0,1\}$ (lógica clásica). En un topos de haces, $\Omega$ puede tener más "valores" (los abiertos de un espacio, por ejemplo).

**Contraste con POLYDIM:** el paper afirma que un estado inválido "colapsa naturalmente a la fibra vacía" en vez de lanzar una excepción. Eso requeriría definir explícitamente el morfismo característico para cada validación de negocio real (ej: "el email tiene formato válido") como un subobjeto del topos — cosa que el paper no hace ni una sola vez con un ejemplo concreto.

**Veredicto:** REAL el concepto — INVENTADA la aplicación práctica descrita.

---

### 3. Lógica de Heyting (lógica intuicionista)

**Intuición cotidiana:** en lógica clásica, "o llueve o no llueve" es siempre verdad (tercero excluido). En lógica intuicionista, una afirmación solo es verdadera si tenés una *prueba constructiva* de ella — "no tengo pruebas de que llueva ni de que no llueva" es una posición lógica válida, no una contradicción.

**La matemática real:** un álgebra de Heyting es una red (lattice) con una operación de implicación que no satisface la ley del tercero excluido ($A \vee \neg A$ no es un teorema). Es la lógica interna natural de cualquier topos. Área madura, con décadas de desarrollo (Heyting, años 1930).

**Contraste con POLYDIM:** el paper dice que "la ausencia del tercero excluido permite que los estados latentes fluyan suavemente entre fronteras semánticas sin violar la consistencia lógica macroscópica". Esto mezcla dos cosas distintas: la *lógica intuicionista* trata sobre qué proposiciones se pueden probar, no sobre *continuidad geométrica* de vectores. Que un vector se mueva "suavemente" en $\mathbb{R}^n$ no tiene relación demostrada con qué axiomas lógicos rigen las proposiciones sobre ese vector.

**Veredicto:** REAL el concepto — INVENTADA la conexión con "superposición" de estados latentes.

---

### 4. Álgebra de Clifford (multivectores)

**Intuición cotidiana:** en vez de solo tener "flechas" (vectores), el álgebra de Clifford te da también "trocitos de plano" (bivectores) y "trocitos de volumen" (trivectores) como objetos algebraicos con los que podés sumar, multiplicar y rotar, todo dentro de una misma estructura.

**La matemática real:** dado un espacio vectorial $V$ con forma cuadrática $Q$, el álgebra de Clifford $C\ell(V,Q)$ es el álgebra generada por $V$ con la relación $v^2 = Q(v)$. Contiene escalares, vectores, bivectores, etc. Es matemática real, usada en física (spin, relatividad) y en gráficos por computadora (rotaciones sin gimbal lock, ver Parte 3).

**Contraste con POLYDIM:** acá el paper acierta en la elección del formalismo — Clifford es una herramienta real y potente para rotaciones. El problema no es este pilar en sí, sino las conclusiones que se le atribuyen más adelante (puntos 5 y 6).

**Veredicto:** REAL, y de hecho el más sólido de los 15 — lo implementamos y verificamos en la Parte 3.

---

### 5. Rotores de espín (Spin Rotors)

**Intuición cotidiana:** para rotar una figura en un videojuego 3D sin que se "trabe" en ciertos ángulos (el clásico *gimbal lock* de los ángulos de Euler), los motores gráficos modernos usan cuaterniones. Un rotor de Clifford es la generalización de un cuaternión a cualquier dimensión.

**La matemática real:** un rotor es un elemento $R = e^{-\theta B/2}$ del grupo Spin, donde $B$ es un bivector unitario ($B^2 = -1$). La rotación se aplica como producto sandwich $v' = R\,v\,\tilde{R}$. Esto preserva la norma y compone correctamente (lo verificamos con código real en la Parte 3). Es matemática estándar en gráficos y robótica.

**Contraste con POLYDIM:** el paper dice que esto "evita el gimbal lock de la alta dimensionalidad". Un rotor efectivamente evita el gimbal lock *de los ángulos de Euler en 3D* — eso es real y lo demostramos. Pero "gimbal lock de alta dimensionalidad" no es una noción definida en ningún lado: no existe una definición estándar de qué sería ese fenómeno en $Cl(n)$ para $n$ grande, así que la frase sí es real matemática (el rotor) aplicada a un problema (alta dimensionalidad de redes neuronales) que nunca se formalizó.

**Veredicto:** REAL el mecanismo — INFLADA la generalización a "alta dimensionalidad" sin definición.

---

### 6. Dualidad de Hodge (en álgebra geométrica)

**Intuición cotidiana:** en un espacio 3D, el dual de un plano (por ejemplo, el plano $xy$) es la línea perpendicular a él (el eje $z$). La dualidad de Hodge formaliza esa correspondencia "plano ↔ línea perpendicular" para cualquier subespacio.

**La matemática real:** dado un pseudoescalar unitario $I$ del álgebra, el dual de Hodge de un multivector $M$ es $M^* = M I^{-1}$. Es una operación real y bien definida en Cl(n), y su costo computacional **depende de la dimensión** $n$ del espacio (no es constante).

**Contraste con POLYDIM:** el paper afirma textualmente que "navegar a dimensiones ortogonales... se hace en $O(1)$ usando el dual de Hodge". Esto es matemáticamente incorrecto tal como está escrito: calcular $M I^{-1}$ para un multivector general en $Cl(n)$ tiene complejidad que crece con $n$ (el pseudoescalar y el multivector tienen hasta $2^n$ componentes). Afirmar $O(1)$ sin ninguna condición adicional es un error de complejidad computacional, no una simplificación válida.

**Veredicto:** REAL el concepto — INCORRECTA la afirmación de complejidad $O(1)$.

---

### 7. ∞-categorías (Lurie) y complejos de Kan

**Intuición cotidiana:** en matemática clásica, dos caminos son "iguales" o "distintos". En una ∞-categoría, dos caminos pueden ser "iguales salvo una deformación continua" (una homotopía), y esa deformación misma puede compararse con otra deformación, y así sucesivamente, en una torre infinita de niveles de "casi igual".

**La matemática real:** las ∞-categorías (formalizadas por Jacob Lurie usando complejos de Kan y quasi-categorías) generalizan la noción de categoría permitiendo morfismos entre morfismos, indefinidamente. El *Kan horn filling* es una condición combinatoria exacta y discreta: dado un "cuerno" (parte de un símplice), existe una forma de completarlo. Área real, de investigación activa en topología algebraica y teoría de tipos homotópica (HoTT).

**Contraste con POLYDIM:** el paper propone usar "Kan horn filling" para "estabilizar" dos caminos de cómputo que difieren por errores de redondeo de punto flotante. El problema es de naturaleza distinta: el horn filling es una operación *exacta* sobre estructuras *discretas* (símplices combinatorios); el error de redondeo es un fenómeno *continuo* y *numérico*. El propio Apéndice del paper (sección 9, punto 1) lo reconoce indirectamente al decir que el "horn filling estricto es intratable" y sustituirlo por homología persistente — pero nunca prueba que esa sustitución preserve el significado original de "identidad homotópica".

**Veredicto:** REAL el concepto — INVENTADO el puente hacia errores de punto flotante.

---

### 8. Ditopología y d-paths (topología dirigida)

**Intuición cotidiana:** pensá en las calles de una ciudad con calles de un solo sentido. Podés preguntarte "¿puedo llegar de A a B respetando el sentido de las calles?" — eso es un camino dirigido (d-path). Puede que exista un camino topológico normal entre A y B, pero no uno que respete las direcciones.

**La matemática real:** la topología dirigida (usada en verificación de sistemas concurrentes, por Goubault y otros) estudia espacios con una noción de "dirección preferida" en los caminos, relevante para analizar condiciones de carrera en programas concurrentes. Es un área real, aunque más especializada.

**Contraste con POLYDIM:** el paper concluye que por esto "es geométricamente imposible que la red experimente loops temporales lógicos". Esta es una afirmación de imposibilidad (un teorema fuerte) sin demostración: pasar de "restringimos las trayectorias de inferencia a un preorden" a "es imposible" requiere probar que el preorden efectivamente se respeta en la implementación real de la red — cosa que no se muestra.

**Veredicto:** REAL el concepto — INJUSTIFICADA la afirmación de imposibilidad.

---

### 9. Geometría de la información: métrica de Fisher-Rao

**Intuición cotidiana:** no todas las "distancias" entre dos recetas de cocina son iguales aunque cambies la misma cantidad de un ingrediente: cambiar de 1 a 2 gramos de sal es un cambio enorme de sabor; cambiar de 500 a 501 gramos de harina casi no se nota. La métrica de Fisher-Rao mide distancias entre distribuciones de probabilidad "a escala del efecto real", no en unidades crudas.

**La matemática real:** la métrica de información de Fisher define una estructura riemanniana sobre el espacio de distribuciones de probabilidad de una familia paramétrica. Es la base de la geometría de la información (Amari, Rao) y tiene aplicaciones reales y bien establecidas en estadística y en optimización (gradiente natural).

**Contraste con POLYDIM:** este es otro de los pilares donde el nombre está bien elegido — el gradiente natural es una técnica real y usada en ML. El paper no comete un error grave acá, más allá de integrarlo como si fuera parte de la misma arquitectura "topológica" sin mostrar cómo se conecta con los rotores o el topos.

**Veredicto:** REAL y razonablemente bien aplicado en su forma aislada — el problema es la integración forzada con el resto.

---

### 10. Conexiones $\alpha$ de Amari

**Intuición cotidiana:** hay más de una forma "razonable" de promediar o interpolar entre dos distribuciones de probabilidad, así como hay más de una forma razonable de promediar dos colores (mezclar pigmentos no es lo mismo que mezclar luces). Las conexiones $\alpha$ parametrizan una familia continua de esas formas de interpolar.

**La matemática real:** las $\alpha$-conexiones de Amari son una familia de conexiones afines sobre una variedad estadística, que interpolan entre la conexión exponencial ($\alpha=1$) y la mixta ($\alpha=-1$). Concepto real y bien establecido en geometría de la información.

**Contraste con POLYDIM:** el paper dice que el operador `MIX` "interpola a través de las conexiones $\alpha$ de Amari sobre la métrica de Fisher-Rao" sin especificar qué distribución de probabilidad estaría siendo interpolada en el contexto de mezclar dos vectores de embedding — falta el puente que conecte "vector de una red neuronal" con "punto en una variedad estadística paramétrica".

**Veredicto:** REAL el concepto — INDEFINIDA su aplicación concreta al caso descrito.

---

### 11. Identidad de Woodbury y KFAC

**Intuición cotidiana:** si necesitás invertir una matriz enorme pero sabés que es "una matriz simple más una corrección pequeña", hay un atajo matemático para no tener que invertir la matriz gigante entera — como corregir un balance contable con un pequeño ajuste en vez de rehacer todas las cuentas desde cero.

**La matemática real:** la identidad de Woodbury permite calcular la inversa de $A + UCV$ en términos de la inversa de $A$ (más barata) y una corrección de rango bajo. KFAC (Kronecker-Factored Approximate Curvature) es un método real, publicado y usado en optimización de redes neuronales (Martens & Grosse, 2015) que aprovecha esta estructura para aproximar el gradiente natural de forma eficiente.

**Contraste con POLYDIM:** este es el pilar mejor fundamentado técnicamente del paper — Woodbury + KFAC para acelerar el gradiente natural es una técnica real, publicada, y con implementaciones que funcionan. Acá el paper no exagera.

**Veredicto:** REAL y correctamente aplicado en su forma aislada.

---

### 12. Descomposición de Hodge combinatoria (Laplaciano de complejos de celdas)

**Intuición cotidiana:** en una red de calles de una ciudad, el tráfico se puede separar en tres componentes: el flujo que va "de A hacia B de forma consistente" (como el agua bajando una pendiente), los "remolinos" de tráfico que da vueltas sin avanzar, y una parte residual que ni fluye ni da vueltas, simplemente está en equilibrio.

**La matemática real:** dado un complejo simplicial con operadores de borde $B_1, B_2$, el Laplaciano de Hodge $\Delta_1 = B_2 B_2^T + B_1^T B_1$ permite descomponer un flujo en tres partes ortogonales: gradiente, rotacional (curl) y armónica. Esto es real y tiene aplicaciones concretas y publicadas en ranking de torneos, análisis de redes y procesamiento de señales sobre grafos (Lim, 2020 formaliza esto con rigor para "HodgeRank").

**Contraste con POLYDIM:** el paper usa esta descomposición sobre las activaciones de una red para "filtrar ruido" durante `FIXPOINT`, definiendo la parte "armónica" como "la convergencia estable y segura, la verdadera respuesta al cómputo". Esta última frase es un salto: la teoría de Hodge combinatoria no tiene ninguna noción incorporada de "seguridad" o de que la componente armónica sea preferible semánticamente — eso es una interpretación añadida sin justificación matemática.

**Veredicto:** REAL el mecanismo — INVENTADA la interpretación semántica de "seguridad" atribuida a la componente armónica.

---

### 13. Teorema del punto fijo de Banach y constante de Lipschitz

**Intuición cotidiana:** si tenés un mapa de una ciudad y lo doblás (a escala reducida) y lo tirás dentro de la ciudad real, hay exactamente un punto del mapa que cae justo sobre el lugar que representa — ese es el punto fijo. El teorema de Banach dice que si tu "transformación" siempre encoge las distancias (constante de Lipschitz $< 1$), ese punto existe, es único, y lo podés encontrar iterando.

**La matemática real:** si $T$ es una función tal que $\|T(x) - T(y)\| \le \lambda \|x-y\|$ con $\lambda < 1$ (contracción), entonces $T$ tiene un único punto fijo y las iteraciones $x_{n+1} = T(x_n)$ convergen a él. Es un teorema clásico, con demostración corta y sólida (Banach, 1922).

**Contraste con POLYDIM:** el paper propone imponer una "penalización espectral de Lipschitz" en el loop `FIXPOINT` forzando $\|J_T(x)\|_2 \le \lambda < 1$, y concluye que esto "garantiza convergencia global sin oscilaciones". La hipótesis del teorema (contracción) es real y la conclusión (convergencia) sería válida **si** se demuestra que la penalización efectivamente logra $\lambda < 1$ en todo el dominio de entrenamiento, no solo en el punto donde se mide la penalización. El paper asume la hipótesis satisfecha en vez de probarla — es el único pilar de los 15 donde la matemática citada es exactamente correcta, pero la aplicación da un salto de "yo penalizo esto" a "por lo tanto se garantiza" sin verificación.

**Veredicto:** REAL y bien elegido — INFLADA la certeza de que la penalización garantiza la hipótesis en la práctica.

---

### 14. Transporte óptimo entrópico (Sinkhorn) y distancia de Wasserstein

**Intuición cotidiana:** si tenés un montón de arena en un lugar y querés moverla para que tome la forma de otro montón en otro lugar, la distancia de Wasserstein mide "cuánto trabajo" (masa × distancia) hace falta para esa transformación — es más informativa que comparar solo los picos de los montones.

**La matemática real:** el transporte óptimo entrópico (regularización de Sinkhorn) es una técnica real y muy usada actualmente en ML (generación de imágenes, alineación de dominios, GANs) para calcular distancias de Wasserstein de forma diferenciable y eficiente (Cuturi, 2013).

**Contraste con POLYDIM:** el paper la usa correctamente como sustituto práctico de la homotopía estricta (ver punto 7) para verificar si dos representaciones "son parecidas". El uso en sí no es descabellado — el problema, como ya señalamos, es afirmar que esa distancia pequeña *implica* la existencia de una identidad homotópica formal (símplice de Kan), sin demostrar ese puente.

**Veredicto:** REAL y razonable como heurística — INJUSTIFICADA la equivalencia formal con homotopía que se le atribuye.

---

### 15. Relatividad POLYDIM-CLAensional y C-espacios (Pezzaglia, Castro)

**Intuición cotidiana:** esto es distinto a los 14 anteriores. Imaginate que una teoría dice "un objeto puede tener una dimensión relativa al observador" citando a autores reales — pero cuando buscás esos papers, son propuestas especulativas de física teórica marginal, no resultados establecidos ni ampliamente aceptados en la comunidad de física o matemática.

**La matemática real:** Pezzaglia y Castro son autores reales que publicaron trabajos especulativos sobre "relatividad POLYDIM-CLAensional" y "C-espacios" en revistas de física teórica alternativa. No son resultados centrales de la física ni la matemática establecida — son propuestas minoritarias, no verificadas experimentalmente ni adoptadas por el consenso de la comunidad.

**Contraste con POLYDIM:** este es el pilar más débil de los 15: apoya una afirmación central del paper ("la dimensionalidad de un objeto de software es relativa al observador") citando trabajo especulativo y no consensuado como si fuera fundamento sólido. Es una apelación a autoridad con una autoridad frágil.

**Veredicto:** el concepto citado existe como propuesta especulativa — pero se usa como si fuera un fundamento establecido, cuando no lo es.

---

## PARTE 2 — Diagnóstico de pseudociencia: caso de estudio POLYDIM

### El patrón "nombre-caída-conclusión"

Analizando los 15 pilares anteriores aparece un patrón consistente en 11 de los 15:

1. **Nombrar** un objeto matemático real y específico (topos, Kan horn filling, Wasserstein).
2. Opcionalmente, **reconocer** una limitación práctica real (ej: "el horn filling estricto es intratable").
3. **Sustituir** por otra herramienta relacionada pero matemáticamente distinta (homología persistente en vez de horn filling).
4. **Concluir** con una afirmación fuerte (garantía, imposibilidad, optimalidad) que la sustitución del paso 3 no demuestra.

El paso 2 es lo que hace creíble al texto: admite límites reales, lo cual da la sensación de honestidad autocrítica. El fallo está siempre en el salto del paso 3 al 4.

### Checklist para detectar este patrón en cualquier texto técnico

Cuando leas una afirmación técnica que suene impresionante, preguntate:

1. **¿Hay una definición formal de cada término, o son solo nombres de teorías famosas?** (ej: "topos" mencionado sin construir el sitio $(\mathcal{C}, J)$ específico.)
2. **¿La afirmación de complejidad computacional tiene una demostración o un análisis, o es una cifra suelta?** (ej: "$O(1)$" sin análisis de por qué no depende de $n$.)
3. **¿Las palabras "garantiza", "imposible", "certifica" están acompañadas de una prueba, o son afirmaciones de fe?**
4. **¿Hay algún experimento reproducible, con código, datos y métricas reportadas?** (o solo se dice "ha sido validado" sin una sola cifra.)
5. **¿Los "puentes" entre dos áreas de matemática distintas están construidos explícitamente, o solo se yuxtaponen por similitud de vocabulario?**

Aplicado a POLYDIM: de las 15 secciones, prácticamente ninguna pasa los 5 criterios simultáneamente. Los pilares 4 (Clifford), 9 (Fisher-Rao) y 11 (Woodbury/KFAC) son los que mejor resisten el checklist en su forma aislada — no es casual que sean, también, los que elegimos para la implementación real de la Parte 3.

### Por qué esto importa más allá de POLYDIM

Este patrón no es exclusivo de este documento. Es una técnica retórica que aparece cada vez con más frecuencia en textos generados con asistencia de IA sin supervisión crítica, precisamente porque un modelo de lenguaje puede encadenar términos técnicos reales con fluidez gramatical perfecta sin que eso implique validez matemática. El antídoto no es desconfiar de la terminología avanzada — es exigir, en cada afirmación fuerte, la demostración o el experimento que la sostenga.

---

## PARTE 3 — Implementación real y verificada: rotores de Clifford

A diferencia de las 14 secciones anteriores, acá no nos quedamos en la crítica: construimos desde cero el álgebra de Clifford $Cl(3,0)$ en Python, sin librerías externas de álgebra geométrica, y verificamos con pruebas ejecutables cuáles de las propiedades atribuidas a los rotores son reales.

### 3.1 Planteo matemático

Representamos cada elemento base ("blade") del álgebra como una máscara de bits de 3 dígitos:

| Bitmask | Blade | Significado |
|---|---|---|
| `000` | $1$ | escalar |
| `001` | $e_1$ | vector base 1 |
| `010` | $e_2$ | vector base 2 |
| `100` | $e_3$ | vector base 3 |
| `011` | $e_{12}$ | bivector (plano 1-2) |
| `101` | $e_{13}$ | bivector (plano 1-3) |
| `110` | $e_{23}$ | bivector (plano 2-3) |
| `111` | $e_{123}$ | pseudoescalar (volumen) |

Un multivector es una combinación lineal de estos 8 elementos. El producto geométrico de dos blades se calcula con el algoritmo de "reordenamiento canónico": el número de intercambios de signo necesarios para ordenar los índices, más la contracción $e_i^2 = +1$ (métrica euclídea).

Un **rotor** en el plano $e_1 e_2$ es:

$$R = e^{-\theta B/2} = \cos(\theta/2) - \sin(\theta/2)\, e_{12}$$

(fórmula cerrada válida porque $e_{12}^2 = -1$, exactamente como $i^2=-1$ en los números complejos). La rotación de un vector $v$ es el producto sandwich:

$$v' = R\, v\, \tilde{R}$$

donde $\tilde{R}$ es la reversión de $R$ (invertir el orden de los factores en cada blade).

### 3.2 Código verificado

El siguiente código fue ejecutado en un intérprete real antes de escribir este documento — los resultados que siguen son la salida real, no una proyección.

```python
import numpy as np

def blade_sign(a: int, b: int) -> int:
    """Signo por reordenamiento canónico al multiplicar blade a * blade b."""
    a >>= 1
    sign = 1
    while a:
        sign *= (-1) ** bin(a & b).count("1")
        a >>= 1
    return sign

def blade_multiply(a: int, b: int):
    sign = blade_sign(a, b)
    result = a ^ b  # métrica euclídea: bits comunes se contraen a +1, no cambian signo
    return result, sign

def mv_multiply(A: dict, B: dict) -> dict:
    result = {}
    for ba, ca in A.items():
        for bb, cb in B.items():
            blade, sign = blade_multiply(ba, bb)
            result[blade] = result.get(blade, 0.0) + sign * ca * cb
    return {k: v for k, v in result.items() if abs(v) > 1e-12}

def mv_reverse(A: dict) -> dict:
    def grade(bitmask):
        return bin(bitmask).count("1")
    out = {}
    for b, c in A.items():
        k = grade(b)
        sign = (-1) ** (k * (k - 1) // 2)
        out[b] = sign * c
    return out

def vec(x, y, z) -> dict:
    return {1: x, 2: y, 4: z}

def mv_norm(A: dict) -> float:
    prod = mv_multiply(A, mv_reverse(A))
    return np.sqrt(prod.get(0, 0.0))

def rotor_e12(theta):
    half = theta / 2.0
    return {0: np.cos(half), 3: -np.sin(half)}  # R = exp(-theta/2 * e12)

def apply_rotor(R: dict, v: dict) -> dict:
    Rr = mv_reverse(R)
    return mv_multiply(mv_multiply(R, v), Rr)
```

### 3.3 Resultados de verificación (salida real del intérprete)

**Test 1 — ¿El rotor preserva la norma?**

```
theta=   0°  |v|=1.000000  |RvR~|=1.000000  preservada=True
theta=  30°  |v|=1.000000  |RvR~|=1.000000  preservada=True
theta=  90°  |v|=1.000000  |RvR~|=1.000000  preservada=True
theta= 137°  |v|=1.000000  |RvR~|=1.000000  preservada=True
theta= 350°  |v|=1.000000  |RvR~|=1.000000  preservada=True
```
**Confirmado.** La norma se preserva exactamente para cualquier ángulo. Esto es un teorema real de álgebra de Clifford, no una afirmación de fe.

**Test 2 — ¿Coincide con la matriz de rotación clásica R_z(θ)?**

```
Rotor de Clifford:  [0.18167986 0.73958936 0.]
Matriz R_z clásica: [0.18167986 0.73958936 0.]
Coinciden: True
```
**Confirmado.** El rotor de Clifford no es una metáfora — reproduce exactamente el resultado de la matriz de rotación estándar de trigonometría.

**Test 3 — ¿La composición de dos rotores da el rotor del ángulo suma?**

```
R2*R1            = {0: 0.6087614290087207, 3: -0.7933533402912353}
R(theta1+theta2) = {0: 0.6087614290087207, 3: -0.7933533402912352}
Coinciden: True
```
**Confirmado** (dentro del error de punto flotante de la última cifra, ~1e-16). $R_2 R_1 = R_{\theta_1+\theta_2}$, propiedad de grupo real.

**Test 4 — ¿$R\tilde{R} = 1$ (el rotor es puramente una rotación, no un escalado)?**

```
R * R~ = {0: 1.0}
```
**Confirmado.**

**Test 5 — ¿Qué pasa realmente en dimensión más alta (Cl(4,0))?**

```
v4 original: e1=1.0, e2=0.0, e3=2.0, e4=-1.0
v4 rotado (plano e1-e2, 90°): {e2: 1.0, e3: 2.0, e4: -1.0}
```
**Confirmado y clave:** el rotor actúa exclusivamente sobre el plano $e_1 e_2$ que definimos; las componentes $e_3$ y $e_4$ quedan intactas (2.0 y -1.0 sin cambios). **Esto sí es una propiedad real y demostrable**: un rotor de Clifford rota un plano específico sin interferir con los ejes ortogonales a él, en cualquier dimensión.

Lo que **no** se puede afirmar a partir de este resultado es la frase del paper POLYDIM: "evita el gimbal lock de la alta dimensionalidad" como propiedad general del sistema. Lo que demostramos es algo más modesto y más preciso: *un* rotor rota *un* plano sin afectar el resto. Que eso resuelva algún problema de "alta dimensionalidad" en el espacio latente de una red neuronal es una extrapolación que el paper nunca formaliza ni mide.

### 3.4 Relación explícita con POLYDIM: qué queda en pie y qué no

| Afirmación de POLYDIM | ¿Sostenida por el código? |
|---|---|
| "El rotor preserva volumen/norma semántica" | Sí — Test 1 lo confirma como propiedad matemática general. |
| "Rotación pura sin distorsión, coincide con rotación clásica" | Sí — Test 2 lo confirma exactamente. |
| "Composición de operaciones MIX es consistente" | Sí, si MIX se define como composición de rotores — Test 3 lo confirma. |
| "Evita el gimbal lock de alta dimensionalidad" | Parcialmente: rota un plano sin afectar otros ejes (Test 5), pero "gimbal lock de alta dimensionalidad" nunca se define formalmente, así que la frase no es verificable tal como está escrita. |
| "Resuelve el problema neuro-simbólico de las redes reales" | No demostrado: nunca se conecta el multivector con pesos reales de una red entrenada, ni se mide ninguna métrica de una red real usando este mecanismo. |

---

## Conclusión general

De los 15 pilares, el álgebra de Clifford (con sus rotores) es el único que pudimos tomar, implementar desde cero, y verificar con pruebas ejecutables que reproduce exactamente lo que promete matemáticamente. Los otros 14 son, en distintos grados, matemática real con conclusiones no demostradas o extrapolaciones injustificadas hacia el dominio de las redes neuronales.

La lección metodológica que te llevás de este ejercicio no es "desconfiar de la matemática avanzada", sino exactamente lo que vos mismo planteaste como "trampa educativa": la única forma de distinguir un resultado real de una apariencia de rigor es intentar construirlo y verificarlo vos mismo, paso por paso, con código o con una demostración — no aceptar la fluidez del lenguaje técnico como sustituto de la prueba.
