# Teorema de Preservación Cognitiva (Versión Rigurosa)

**Propuesta de frente de investigación — POLYDIM-CLA / MPAT_V3_02**
**Fuente:** Auditoría externa independiente (Claude)

---

## 1. El Marco Formal: Espacios y Tareas

Un estado cognitivo en dimensión \(n\) es un punto \(x \in V_n = \mathbb{R}^n\).
Una transformación dimensional es \(\varphi: V_n \to V_m\) ("bajar" si \(m < n\), "subir" si \(m > n\)).
Una tarea (decisión, clasificación, unidad agéntica) es una función \(f: V_n \to Y\).

Decimos que **\(\varphi\) preserva \(f\)** si existe \(g: V_m \to Y\) tal que \(f = g \circ \varphi\).

---

## 2. Refutación de la Preservación Universal (Teorema A)

**Teorema A (Preservación total \(\iff\) Inyectividad):**
\(\varphi\) preserva *toda* tarea posible (cualquier \(Y\), cualquier \(f\)) si y solo si \(\varphi\) es inyectiva.

**Demostración:**
- (\(\impliedby\)) Si \(\varphi\) es inyectiva, ningún par \(x_1 \neq x_2\) comparte imagen, así que "\(f\) constante en las fibras de \(\varphi\)" se cumple vacíamente para cualquier \(f\); definir \(g(\varphi(x)) := f(x)\) no genera ambigüedad.
- (\(\implies\)) Si \(\varphi\) no es inyectiva, existen \(x_1 \neq x_2\) con \(\varphi(x_1) = \varphi(x_2) =: z\). Definamos \(f(x_1) := 0\), \(f(x_2) := 1\). Si existiera \(g\) con \(f = g \circ \varphi\), tendríamos \(g(z) = 0\) y \(g(z) = 1\) simultáneamente — imposible. Esa \(f\) no se preserva. \(\blacksquare\)

**Corolario A.1 (Bajar dimensiones SIEMPRE pierde información):**
Si \(\varphi\) es lineal y \(m < n\), por rango-nulidad \(\dim(\ker\varphi) \geq n-m > 0\), así que \(\varphi\) nunca es inyectiva. Por Teorema A: **ninguna reducción de dimensión preserva todas las tareas posibles.** No es un límite de ingeniería, es álgebra lineal estricta. (Válido también en dominios continuos por Invariancia del Dominio de Brouwer).

**Corolario A.2 (Subir dimensiones es gratis):**
Si \(m \geq n\), existe \(\varphi\) inyectiva (inclusión canónica \(x \mapsto (x,0,\dots,0)\)), y por Teorema A esa \(\varphi\) preserva todo. La asimetría original de POLYDIM (MODO_G/MODO_S) —"bajar es peligroso, subir no"— es un teorema, no una intuición.

---

## 3. La Versión Corregida (Teorema B)

Nadie necesita preservar todas las tareas imaginables, solo las relevantes.

**Teorema B (Preservación de Tareas Específicas):**
Dado un conjunto de tareas relevantes \(\mathcal{T} = \{f_1, \dots, f_k\}\), \(\varphi\) preserva cada \(f \in \mathcal{T}\) si y solo si \(\varphi\) no confunde ningún par de puntos que alguna \(f \in \mathcal{T}\) distinga: 
\[\varphi(x_1) = \varphi(x_2) \implies f(x_1) = f(x_2) \quad \forall f \in \mathcal{T}\]

La condición no pide inyectividad global, sino que **\(\ker\varphi\) quede contenido en la intersección de los conjuntos de nivel de cada \(f \in \mathcal{T}\)**. Este es un criterio de diseño verificable algebraicamente.

---

## 4. Estado del Arte (Lo que ya existe)

La literatura de **compressive classification** ya abordó versiones de este problema:
- Calderbank et al. (2009): Desempeño de SVM bajo reducción vía JL y compressed sensing.
- Davenport et al.: "Smashed filter" para clasificación en dominio comprimido.
- McWhirter et al. (SqueezeFit): Proyectores de baja dimensión que preservan separación prescrita de clases.

**Conclusión:** POLYDIM-CLA no puede reclamar la noción de "preservar una tarea tras reducir dimensión" como propia sin citar este bloque masivo de literatura.

---

## 5. El Verdadero Hueco de Investigación (Aporte Defendible)

Toda esa literatura resuelve UN proyector + UN clasificador. Lo que no está resuelto es la **versión compuesta**:
Una cadena de \(k\) transformaciones dimensionales heterogéneas \(\varphi_1, \dots, \varphi_k\) intercaladas con decisiones agénticas sucesivas (el escenario AI2AI real, donde cada agente vive en su propia dimensión \(V_{n_i}\)).

**El problema abierto:**
No es "¿se preserva \(f\) tras un cambio?", sino **"¿Qué cota de preservación sobrevive tras componer \(k\) cambios heterogéneos, cada uno con su propio \((n_i, m_i, \epsilon_i)\)?"**
Esto fundamentaría la **Cota de Preservación Compuesta para cadenas de decisión AI2AI** — un aporte inédito y altamente publicable.

---

## 6. Conexión Teórica (Puentes Múltiples)

### Puente hacia la Teoría de Haces (Punto C / Protocolos)
La condición del Teorema B ("\(f\) constante en las fibras de \(\varphi\)") es la propiedad universal de un coecualizador (versión local). 
La versión global (N agentes conectados por una red de \(\varphi_{ij}\)) es cohomología de haces pura: existe estructura cognitiva global consistente si y solo si se anula la obstrucción en \(H^1\) de la cubierta correspondiente.
El prehaz a "sheafificar" es:
\[ U \mapsto \{ f : V_{n_U} \to Y \mid f \text{ respeta todos los } \varphi_{ij} \text{ en } U \} \]

### Corolario Operacional para MODO_G/MODO_S
Si el humano/observador periférico vive en una vista comprimida \(\varphi_h: V_n \to V_h\) (\(h \ll n\)) del estado completo del sistema, el Teorema B da la condición exacta: \(\varphi_h\) puede comprimir todo lo que quiera, **siempre que no confunda dos estados que difieran en alguna \(f\) que el humano necesite distinguir** (ej. "¿el sistema está por tomar una acción irreversible?"). Esto deja de ser una metáfora y pasa a ser un parámetro de diseño algebraico.
