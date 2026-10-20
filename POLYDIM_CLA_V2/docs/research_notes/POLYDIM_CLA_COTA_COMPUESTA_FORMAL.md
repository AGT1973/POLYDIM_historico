# Cota de Preservación Compuesta para Cadenas de Decisión AI2AI (Teorema C)

**Propuesta de frente de investigación — POLYDIM-CLA / MPAT_V3_02**
**Fuente:** Derivación formal rigurosa (Auditoría externa 2026)

---

## 0. Traducción del enunciado original a objetos formales

Fijamos el diccionario entre la prosa original y los objetos matemáticos:

| Enunciado original | Objeto formal |
|---|---|
| MODO_S (alta dimensionalidad, especialista) | \(\mathbb{R}^{n_0}\) |
| MODO_G (baja dimensionalidad, generalista) | \(\mathbb{R}^{n_k}\), tras \(k\) transformaciones |
| "transformación de proyección dimensional" | \(\varphi_i:\mathbb{R}^{n_{i-1}}\to\mathbb{R}^{n_i}\), lineal, \((n_{i-1},n_i,\varepsilon_i)\) |
| "coherencia del modelo interno" | invariancia de \(\ker(\Phi_k)^{\perp}\) respecto a las direcciones de tarea |
| "densidad de información local" perdida | \(\dim\ker(\varphi_i) = n_{i-1}-n_i>0\) (Teorema A) |
| "integridad semántica de la tarea global" preservada | \(\ker(\varphi_i)\subseteq\bigcap_j\ker(df_j)\) (Teorema B) |
| "decisiones agénticas sucesivas" | composición \(\Phi_k=\varphi_k\circ\cdots\circ\varphi_1\) con lectura de tarea intercalada |

---

## 1. Preliminares (Recapitulación de Teoremas A y B)

**Definición 1 (tarea linealizada).** Sea \(f_j:\mathbb{R}^{n_0}\to\mathbb{R}\) una tarea de decisión, diferenciable. En la región de interés, admite la aproximación lineal \(f_j(x)\approx w_j^\top x\), con \(w_j=\nabla f_j\).

**Teorema A (cálculo de núcleo puro).** Sea \(\varphi:\mathbb{R}^{n}\to\mathbb{R}^{m}\) lineal, \(m<n\). Por el teorema de rango-nulidad, \(\dim\ker(\varphi)=n-\operatorname{rank}(\varphi)\ge n-m>0\). La preservación incondicional de toda la información (inyectividad) es imposible para toda reducción dimensional estricta.

**Definición 2 (compatibilidad exacta de tarea).** \(\varphi\) preserva \(f_j\) si \(x-x'\in\ker(\varphi) \Rightarrow f_j(x)=f_j(x')\). Para \(f_j\) linealizada, esto equivale a \(\ker(\varphi)\subseteq \ker(w_j\,\cdot) = w_j^{\perp}\).

**Teorema B.** \(\varphi\) preserva simultáneamente \(\{f_1,\dots,f_r\}\) si y solo si
\[ \ker(\varphi)\ \subseteq\ \bigcap_{j=1}^r w_j^{\perp} \;=\; \big(\operatorname{span}\{w_1,\dots,w_r\}\big)^{\perp}. \]

---

## 2. El caso de referencia k=1 (SOTA previo)

La literatura de **compressive classification** (Calderbank, Davenport et al.) cubre exactamente este caso, cuantificando \(\varepsilon_1\) para una \(\varphi_1\) (aleatoria tipo RIP/JL) y un clasificador lineal o 1-NN. El hueco en SOTA es qué sobrevive tras componer \(k\ge 2\) transformaciones heterogéneas intercaladas con decisiones (cadena AI2AI).

---

## 3. Teorema C: Cota de Preservación Compuesta

El ángulo euclídeo crudo entre la dirección de tarea y el \(\ker\) no telescopa. La cantidad correcta es el ángulo de Mahalanobis en la covarianza acumulada.

**Definición 3 (ángulo de fuga adaptado).** Sea \(x\sim \mathcal N(0,I_{n_0})\). Para una dirección de tarea \(w\) (\(\|w\|=1\)), definimos recursivamente:
\[ \Sigma_0:=I_{n_0},\qquad \Sigma_i:=\varphi_i\Sigma_{i-1}\varphi_i^\top,\qquad c_i:=\Phi_i w,\qquad h_i^{*}:=\Sigma_i^{+}c_i \]
El ángulo de fuga en la etapa \(i\), \(\vartheta_i\), es el ángulo de Mahalanobis (métrica \(\Sigma_{i-1}\)) entre \(h_{i-1}^{*}\) y \(\operatorname{row}(\varphi_i)\). Escribimos \(F_i:=\cos\vartheta_i\), \(\varepsilon_i:=\sin\vartheta_i\).

**Teorema C (Cota de Preservación Compuesta — identidad exacta gaussiana).** Sea \(R_i\) la mejor correlación lineal alcanzable entre \(w^\top x\) y cualquier lectura lineal de \(\Phi_i(x)\). Entonces \(R_0=1\) y, para todo \(k\ge 1\):
\[ R_k=\prod_{i=1}^{k}\cos\vartheta_i=\prod_{i=1}^k F_i, \qquad\text{equivalentemente}\qquad \varepsilon_{\text{total}}:=\sqrt{1-R_k^2}=\sqrt{1-\prod_{i=1}^{k}(1-\varepsilon_i^2)}. \]

**Corolario 1 (cotas utilizables).**
\[ \varepsilon_{\text{total}}\ \le\ 1-\prod_{i=1}^k(1-\varepsilon_i)\ \le\ \sum_{i=1}^k \varepsilon_i. \]

*Nota Teórico-Informacional:* La identidad se sustenta en el coeficiente de contracción de Dobrushin y la correlación máxima de Hirschfeld–Gebelein–Rényi (HGR), que coincide con la correlación canónica de Hotelling en el caso gaussiano.

---

## 4. Reformulación Cohomológica: Haz Celular sobre la Cadena

Sea \(G\) un grafo camino. Fibras \(F(v_i) = \mathbb{R}^{n_i}\), y mapas de restricción \(\varphi_i\).
- **Proposición 1 (Topología trivial).** \(H^0(F)\cong\mathbb{R}^{n_0}\) y \(H^1(F)=0\) (el camino no tiene ciclos, sustitución hacia atrás funciona).
- **Proposición 2 (Curl semántico emergente).** El invariante \(\varepsilon_{\text{total}}^2=1-R_k^2\) es exactamente la norma al cuadrado de la obstrucción a que la 0-cocadena generada por \(w\) se extienda como sección compatible. Es la magnitud exacta del defecto de cobordo restringido al subhaz generado por la dirección de tarea (Cohomología relativa \(H^1(F,F_w)\)).

**Extrapolación a grafos con ciclos (AI2AI en red):** 
Si la topología de interacción deja de ser un camino lineal, aparece un ciclo en \(G\). \(H^1(G;F)\) queda gobernado por el primer número de Betti y la monodromía alrededor del ciclo. Este curl topológico es independiente de la deriva dimensional del Teorema C, justificando la necesidad teórica de anclar POLYDIM en el Laplaciano de Hodge del haz.

---

## 5. Corolarios Finales
**Corolario 2 (Presupuesto de distorsión homogéneo).** Para \(k\) grande, \(\varepsilon \le 1-(1-\tau)^{1/k} \approx \tau/k\). Cada eslabón reduce el margen de error tolerable proporcionalmente a \(1/k\).
**Corolario 3 (Compresión SOTA).** Combina el régimen de Calderbank-Davenport con la composición de Teorema C, demostrando la extensión a redes multicapa heterogéneas.
