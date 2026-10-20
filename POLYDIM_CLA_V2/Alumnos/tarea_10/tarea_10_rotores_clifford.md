# Tarea 10 — Escuadrón 1: Convergencia y rotores de Clifford

**Nota para el grupo:** este material NO sigue el formato "worker/JSON silencioso" de `swarm_context_200.txt`. Incluye la explicación completa, código ejecutado y verificado, y una distinción explícita entre lo que es matemática real y lo que es interpretación exagerada del documento madre — en la misma línea que `tarea_02_rotor_volumen.md` y `tarea_14_hodge_armonica.md`.

## Enunciado tal como aparece en el banco de tareas

"Actúa como investigador de convergencia de ML. Diseña una operación encadenada que demuestre a nivel de volcado intermedio de grafo que converger el tensor usando superposición lineal con rotores sin bloqueos."

Como con la tarea 14, esta consigna combina jerga real de varias áreas (convergencia, grafos computacionales, rotores) sin definir una hipótesis matemática verificable. En vez de fingir una demostración de "convergencia con rotores sin bloqueos" (frase sin significado matemático establecido), este documento hace lo que sí se puede hacer con rigor: tomar el objeto matemático real que el enunciado menciona —el rotor de Clifford— y verificar exactamente qué propiedades tiene y cuáles no.

---

## Contenido de este documento

1. Los 15 pilares matemáticos que POLYDIM cita en su paper, explicados con rigor pedagógico (intuición cotidiana → matemática real → contraste con POLYDIM).
2. Un diagnóstico del patrón retórico que usa el paper para sonar riguroso sin estarlo, con un checklist de 5 preguntas para detectarlo en cualquier texto.
3. La implementación completa y verificada de rotores de Clifford (álgebra geométrica Cl(3,0) construida desde cero), con los resultados reales de ejecución. Ver `tarea_10_rotores_clifford.py` en esta misma carpeta.

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

**Veredicto:** REAL, y de hecho el más sólido de los 15 — lo implementamos y verificamos en la Parte 2.

---

### 5. Rotores de espín (Spin Rotors)

**Intuición cotidiana:** para rotar una figura en un videojuego 3D sin que se "trabe" en ciertos ángulos (el clásico *gimbal lock* de los ángulos de Euler), los motores gráficos modernos usan cuaterniones. Un rotor de Clifford es la generalización de un cuaternión a cualquier dimensión.

**La matemática real:** un rotor es un elemento $R = e^{-\theta B/2}$ del grupo Spin, donde $B$ es un bivector unitario ($B^2 = -1$). La rotación se aplica como producto sandwich $v' = R\,v\,\tilde{R}$. Esto preserva la norma y compone correctamente (lo verificamos con código real en la Parte 2). Es matemática estándar en gráficos y robótica.

**Contraste con POLYDIM:** el paper dice que esto "evita el gimbal lock de la alta dimensionalidad". Un rotor efectivamente evita el gimbal lock *de los ángulos de Euler en 3D* — eso es real y lo demostramos. Pero "gimbal lock de alta dimensionalidad" no es una noción definida en ningún lado: no existe una definición estándar de qué sería ese fenómeno en $Cl(n)$ para $n$ grande, así que la frase sí es real matemática (el rotor) aplicada a un problema (alta dimensionalidad de redes neuronales) que nunca se formalizó.

**Veredicto:** REAL el mecanismo — INFLADA la generalización a "alta dimensionalidad" sin definición.

---

### 6. Dualidad de Hodge (en álgebra geométrica)

**Intuición cotidiana:** en un espacio 3D, el dual de un plano (por ejemplo, el plano $xy$) es la línea perpendicular a él (el eje $z$). La dualidad de Hodge formaliza esa correspondencia "plano ↔ línea perpendicular" para cualquier subespacio.

**La matemática real:** dado un pseudoescalar unitario $I$ del álgebra, el dual de Hodge de un multivector $M$ es $M^* = M I^{-1}$. Es una operación real y bien definida en Cl(n), y su costo computacional **depende de la dimensión** $n$ del espacio (no es constante).

**Contraste con POLYDIM:** el paper afirma textualmente que "navegar a dimensiones ortogonales... se hace en $O(1)$ usando el dual de Hodge". Esto es matemáticamente incorrecto tal como está escrito: calcular $M I^{-1}$ para un multivector general en $Cl(n)$ tiene complejidad que crece con $n$. Afirmar $O(1)$ sin ninguna condición adicional es un error de complejidad computacional, no una simplificación válida.

**Veredicto:** REAL el concepto — INCORRECTA la afirmación de complejidad $O(1)$.

---

### 7. ∞-categorías (Lurie) y complejos de Kan

**Intuición cotidiana:** en matemática clásica, dos caminos son "iguales" o "distintos". En una ∞-categoría, dos caminos pueden ser "iguales salvo una deformación continua" (una homotopía), y esa deformación misma puede compararse con otra, en una torre infinita de niveles de "casi igual".

**La matemática real:** las ∞-categorías (formalizadas por Jacob Lurie usando complejos de Kan y quasi-categorías) generalizan la noción de categoría permitiendo morfismos entre morfismos, indefinidamente. El *Kan horn filling* es una condición combinatoria exacta y discreta. Área real, de investigación activa en topología algebraica y teoría de tipos homotópica (HoTT).

**Contraste con POLYDIM:** el paper propone usar "Kan horn filling" para "estabilizar" dos caminos de cómputo que difieren por errores de redondeo de punto flotante. El horn filling es una operación *exacta* sobre estructuras *discretas*; el error de redondeo es un fenómeno *continuo* y *numérico*. El propio Apéndice del paper reconoce que el "horn filling estricto es intratable" y lo sustituye por homología persistente — pero nunca prueba que esa sustitución preserve el significado original de "identidad homotópica".

**Veredicto:** REAL el concepto — INVENTADO el puente hacia errores de punto flotante.

---

### 8. Ditopología y d-paths (topología dirigida)

**Intuición cotidiana:** pensá en calles de un solo sentido. Podés preguntarte "¿puedo llegar de A a B respetando el sentido de las calles?" — eso es un camino dirigido (d-path). Puede existir un camino topológico normal entre A y B sin que exista uno que respete las direcciones.

**La matemática real:** la topología dirigida (Goubault y otros) estudia espacios con una noción de "dirección preferida" en los caminos, relevante para analizar condiciones de carrera en sistemas concurrentes. Área real, aunque más especializada.

**Contraste con POLYDIM:** el paper concluye que por esto "es geométricamente imposible que la red experimente loops temporales lógicos". Esta es una afirmación de imposibilidad sin demostración: pasar de "restringimos las trayectorias a un preorden" a "es imposible" requiere probar que el preorden se respeta en la implementación real de la red — cosa que no se muestra.

**Veredicto:** REAL el concepto — INJUSTIFICADA la afirmación de imposibilidad.

---

### 9. Geometría de la información: métrica de Fisher-Rao

**Intuición cotidiana:** no todas las "distancias" entre dos recetas son iguales aunque cambies la misma cantidad de un ingrediente: cambiar de 1 a 2 gramos de sal es un cambio enorme de sabor; cambiar de 500 a 501 gramos de harina casi no se nota. La métrica de Fisher-Rao mide distancias entre distribuciones de probabilidad "a escala del efecto real".

**La matemática real:** la métrica de información de Fisher define una estructura riemanniana sobre el espacio de distribuciones de probabilidad de una familia paramétrica. Es la base de la geometría de la información (Amari, Rao), con aplicaciones reales y bien establecidas en estadística y optimización (gradiente natural).

**Contraste con POLYDIM:** este pilar tiene el nombre bien elegido — el gradiente natural es una técnica real y usada en ML. El paper no comete un error grave acá, más allá de integrarlo como si fuera parte de la misma arquitectura "topológica" sin mostrar cómo se conecta con los rotores o el topos.

**Veredicto:** REAL y razonablemente bien aplicado en su forma aislada.

---

### 10. Conexiones $\alpha$ de Amari

**Intuición cotidiana:** hay más de una forma "razonable" de interpolar entre dos distribuciones de probabilidad, así como hay más de una forma razonable de promediar dos colores (mezclar pigmentos no es lo mismo que mezclar luces). Las conexiones $\alpha$ parametrizan una familia continua de esas formas de interpolar.

**La matemática real:** las $\alpha$-conexiones de Amari son una familia de conexiones afines sobre una variedad estadística, que interpolan entre la conexión exponencial ($\alpha=1$) y la mixta ($\alpha=-1$). Concepto real y bien establecido en geometría de la información.

**Contraste con POLYDIM:** el paper dice que el operador `MIX` "interpola a través de las conexiones $\alpha$ de Amari" sin especificar qué distribución de probabilidad estaría siendo interpolada en el contexto de mezclar dos vectores de embedding.

**Veredicto:** REAL el concepto — INDEFINIDA su aplicación concreta al caso descrito.

---

### 11. Identidad de Woodbury y KFAC

**Intuición cotidiana:** si necesitás invertir una matriz enorme pero sabés que es "una matriz simple más una corrección pequeña", hay un atajo matemático para no invertir la matriz gigante entera — como corregir un balance contable con un pequeño ajuste en vez de rehacer todas las cuentas.

**La matemática real:** la identidad de Woodbury permite calcular la inversa de $A + UCV$ en términos de la inversa de $A$ y una corrección de rango bajo. KFAC (Martens & Grosse, 2015) es un método real y publicado que aprovecha esta estructura para aproximar el gradiente natural eficientemente.

**Contraste con POLYDIM:** este es el pilar mejor fundamentado técnicamente del paper — acá no exagera.

**Veredicto:** REAL y correctamente aplicado en su forma aislada.

---

### 12. Descomposición de Hodge combinatoria

**Intuición cotidiana:** en una red de calles, el tráfico se puede separar en tres componentes: flujo consistente "de A hacia B" (como agua bajando una pendiente), "remolinos" que dan vueltas sin avanzar, y una parte residual en equilibrio.

**La matemática real:** dado un complejo simplicial con operadores de borde $B_1, B_2$, el Laplaciano de Hodge $\Delta_1 = B_2 B_2^T + B_1^T B_1$ permite descomponer un flujo en gradiente, rotacional (curl) y armónica. Real, con aplicaciones publicadas en ranking de torneos y análisis de redes (Jiang, Lim, Yao, Ye — HodgeRank, 2011).

**Contraste con POLYDIM:** el paper usa esta descomposición para "filtrar ruido" y define la componente "armónica" como "la convergencia estable y segura, la verdadera respuesta al cómputo". Esta interpretación semántica no tiene sustento en la teoría de Hodge combinatoria — es una interpretación añadida sin justificación matemática. (Ver `tarea_14_hodge_armonica.md` en esta misma carpeta madre para una implementación completa en Rust de este mismo mecanismo.)

**Veredicto:** REAL el mecanismo — INVENTADA la interpretación de "seguridad" atribuida a la componente armónica.

---

### 13. Teorema del punto fijo de Banach y constante de Lipschitz

**Intuición cotidiana:** si tenés un mapa de una ciudad, lo doblás a escala reducida y lo tirás dentro de la ciudad real, hay exactamente un punto del mapa que cae justo sobre el lugar que representa — ese es el punto fijo. Banach dice que si tu transformación siempre encoge las distancias (constante de Lipschitz $< 1$), ese punto existe, es único, y se encuentra iterando.

**La matemática real:** si $\|T(x) - T(y)\| \le \lambda \|x-y\|$ con $\lambda < 1$, $T$ tiene un único punto fijo y las iteraciones convergen a él. Teorema clásico (Banach, 1922).

**Contraste con POLYDIM:** el paper propone forzar $\|J_T(x)\|_2 \le \lambda < 1$ y concluye que esto "garantiza convergencia global sin oscilaciones". La hipótesis es real y la conclusión sería válida **si** se demuestra que la penalización logra $\lambda < 1$ en todo el dominio, no solo donde se mide. El paper asume la hipótesis satisfecha en vez de probarla.

**Veredicto:** REAL y bien elegido — INFLADA la certeza de que la penalización garantiza la hipótesis en la práctica.

---

### 14. Transporte óptimo entrópico (Sinkhorn) y distancia de Wasserstein

**Intuición cotidiana:** si tenés arena en un lugar y querés moverla para que tome la forma de otro montón en otro lugar, la distancia de Wasserstein mide "cuánto trabajo" (masa × distancia) hace falta para esa transformación.

**La matemática real:** el transporte óptimo entrópico (Cuturi, 2013) es una técnica real y muy usada en ML para calcular distancias de Wasserstein de forma diferenciable y eficiente.

**Contraste con POLYDIM:** se usa correctamente como sustituto práctico de la homotopía estricta (punto 7). El problema, como ya se señaló, es afirmar que esa distancia pequeña *implica* la existencia de una identidad homotópica formal, sin demostrar ese puente.

**Veredicto:** REAL y razonable como heurística — INJUSTIFICADA la equivalencia formal con homotopía que se le atribuye.

---

### 15. Relatividad POLYDIM-CLAensional y C-espacios (Pezzaglia, Castro)

**Intuición cotidiana:** esto es distinto a los 14 anteriores. Una teoría dice "un objeto puede tener una dimensión relativa al observador" citando autores reales — pero al buscar esos papers, son propuestas especulativas de física teórica marginal, no resultados establecidos ni consensuados.

**La matemática real:** Pezzaglia y Castro son autores reales que publicaron trabajos especulativos sobre "relatividad POLYDIM-CLAensional" y "C-espacios" en revistas de física teórica alternativa. No son resultados centrales de la física ni la matemática establecida.

**Contraste con POLYDIM:** el pilar más débil de los 15: apoya una afirmación central del paper citando trabajo especulativo como si fuera fundamento sólido.

**Veredicto:** el concepto citado existe como propuesta especulativa — pero se usa como si fuera un fundamento establecido, cuando no lo es.

---

## PARTE 2 — Rotores de Clifford: implementación real y verificada

### Planteo matemático

Representamos cada blade del álgebra $Cl(3,0)$ como una máscara de bits de 3 dígitos (000=escalar, 001=e1, 010=e2, 100=e3, 011=e12, 101=e13, 110=e23, 111=e123). El producto geométrico se calcula con el algoritmo de reordenamiento canónico (contar swaps de signo) más la contracción $e_i^2=+1$ (métrica euclídea).

Un rotor en el plano $e_1e_2$: $R = e^{-\theta B/2} = \cos(\theta/2) - \sin(\theta/2)e_{12}$ (fórmula cerrada, igual que Euler pero con $e_{12}$ en vez de $i$). La rotación es el producto sandwich $v' = R\,v\,\tilde{R}$.

El código completo está en `tarea_10_rotores_clifford.py`, en esta misma carpeta.

### Resultados de verificación (salida real del intérprete, ejecutada antes de escribir este documento)

**Test 1 — ¿El rotor preserva la norma?**
```
theta=   0°  |v|=1.000000  |RvR~|=1.000000  preservada=True
theta=  30°  |v|=1.000000  |RvR~|=1.000000  preservada=True
theta=  90°  |v|=1.000000  |RvR~|=1.000000  preservada=True
theta= 137°  |v|=1.000000  |RvR~|=1.000000  preservada=True
theta= 350°  |v|=1.000000  |RvR~|=1.000000  preservada=True
```
Confirmado: teorema real de álgebra de Clifford.

**Test 2 — ¿Coincide con la matriz de rotación clásica R_z(θ)?**
```
Rotor de Clifford:  [0.18167986 0.73958936 0.]
Matriz R_z clásica: [0.18167986 0.73958936 0.]
Coinciden: True
```
Confirmado: el rotor reproduce exactamente el resultado trigonométrico estándar.

**Test 3 — ¿La composición de rotores da el rotor del ángulo suma?**
```
R2*R1            = {0: 0.6087614290087207, 3: -0.7933533402912353}
R(theta1+theta2) = {0: 0.6087614290087207, 3: -0.7933533402912352}
Coinciden: True
```
Confirmado (diferencia ~1e-16, ruido de punto flotante). Propiedad de grupo real.

**Test 4 — ¿$R\tilde{R}=1$?**
```
R * R~ = {0: 1.0}
```
Confirmado.

**Test 5 — ¿Qué pasa en Cl(4,0)?**
```
v4 original: e1=1.0, e2=0.0, e3=2.0, e4=-1.0
v4 rotado (plano e1-e2, 90°): {e2: 1.0, e3: 2.0, e4: -1.0}
```
Confirmado y clave: el rotor rota exclusivamente el plano $e_1e_2$; $e_3$ y $e_4$ quedan intactos. Esto sí es real y demostrable. Lo que no se puede afirmar a partir de este resultado es la frase de POLYDIM "evita el gimbal lock de la alta dimensionalidad" como propiedad general del sistema — esa frase nunca se define formalmente.

### Relación explícita con el enunciado de la Tarea 10

El enunciado pedía demostrar "convergencia del tensor usando superposición lineal con rotores sin bloqueos". Lo que se puede demostrar con rigor es más modesto y más preciso: los rotores de Clifford (a) preservan norma, (b) coinciden exactamente con las rotaciones clásicas, (c) componen correctamente como grupo, y (d) actúan sobre un plano específico sin interferir con otros ejes en cualquier dimensión. Ninguna de estas cuatro propiedades demuestra ni implica "convergencia de un tensor" en el sentido de optimización o entrenamiento de una red — esa sería una afirmación distinta, que requeriría definir una función objetivo y probar que iterar con rotores converge a un mínimo, cosa que ni el enunciado ni el paper madre formalizan.

## Para que cada alumno verifique por su cuenta

1. Correr `tarea_10_rotores_clifford.py` con otros ángulos y confirmar que la norma se sigue preservando.
2. Extender el Test 5 a Cl(5,0) o Cl(6,0) y confirmar que el patrón se mantiene (el rotor solo afecta su plano).
3. Preguntarse: ¿qué habría que agregar a este código para que constituya una prueba real de "convergencia"? (pista: hace falta una función de pérdida y un proceso iterativo, ninguno de los cuales está definido en el enunciado original.)
