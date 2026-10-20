# Tarea 55 — POLYDIM (Escuadrón 3: Descomposición de Hodge y Grafos Esparsos)

## 1. Enunciado original (banco maestro)

> "55 (Test 55): Actúa como arquitecto de compresión de IA. Escribe una clase nativa de PyTorch que reemplace a `nn.Linear` operando internamente con el sistema de 'Anclaje de Codebook' logra recuperar la identidad original (GEO_ID)."

Gramaticalmente roto (falta conector/verbo entre "operando internamente con..." y "logra recuperar...") y además usa dos términos sin definir en el propio banco: **"Anclaje de Codebook"** y **"GEO_ID"**. Estos son placeholders del marco POLYDIM (junto con "SPS" y "GeoID Stability" mencionados en la introducción del banco), que el propio marco pide traducir a métricas matemáticas estándar verificables — no términos con significado técnico establecido en la literatura. Se los traduce explícitamente en la Sección 2.

## 2. Reconstrucción del enunciado y traducción de términos ambiguos

**Traducción de los placeholders:**
- *"Anclaje de Codebook"* → **cuantización vectorial (Vector Quantization, VQ)** de los pesos: cada vector de peso se reemplaza ("ancla") por el codeword más cercano de un diccionario/codebook aprendido, técnica estándar de compresión de redes (Han, Mao & Dally, *Deep Compression*, 2015; Gray, *Vector Quantization*, 1984).
- *"Identidad original (GEO_ID)"* → no hay una métrica estándar con ese nombre. Se traduce a la pregunta matemáticamente precisa: **¿es posible, a partir del codeword asignado, identificar/recuperar de qué vector de peso original provino?** Esto es un problema de **identificabilidad de cuantización**, medible con dos métricas estándar: error de reconstrucción L2 y precisión de recuperación por vecino más cercano (retrieval accuracy).

**Versión reformulada, investigable:**

> Implementar una clase de PyTorch, `CodebookLinear`, que reemplace a `nn.Linear` comprimiendo su matriz de pesos $W \in \mathbb{R}^{d_{out}\times d_{in}}$ mediante cuantización vectorial: particionar las filas de $W$ en vectores, aprender un codebook $C\in\mathbb{R}^{K\times d_{in}}$ vía k-means, y reemplazar cada fila por su codeword más cercano (el "anclaje"). Evaluar empíricamente si este anclaje preserva suficiente información como para **recuperar/identificar la fila original** a partir del codeword asignado (identificabilidad), y cuantificar la pérdida de fidelidad funcional de la capa comprimida frente a la original.

## 3. Modelo matemático estándar

### 3.1 Cuantización vectorial de los pesos

Sea $W \in \mathbb{R}^{d_{out}\times d_{in}}$ la matriz de pesos de la capa lineal original, con filas $w_1,\dots,w_{d_{out}} \in \mathbb{R}^{d_{in}}$ (cada una es el vector de pesos de una neurona de salida).

**Codebook:** se aprende $C = \{c_1,\dots,c_K\}\subset\mathbb{R}^{d_{in}}$ resolviendo k-means:

$$
C^\star = \arg\min_{C}\ \sum_{i=1}^{d_{out}} \min_{c\in C} \|w_i - c\|_2^2
$$

**Anclaje (asignación):** cada fila se ancla a su codeword más cercano:

$$
a_i = \arg\min_{k\in\{1,\dots,K\}} \|w_i - c_k\|_2, \qquad \hat{w}_i = c_{a_i}
$$

La matriz reconstruida (dequantizada) es $\hat{W}$, con filas $\hat{w}_i$. La compresión viene de almacenar solo $C$ ($K\times d_{in}$ valores) y los índices $a_i$ ($d_{out}$ enteros de $\lceil \log_2 K\rceil$ bits) en vez de $W$ completa ($d_{out}\times d_{in}$ valores en punto flotante).

**Forward pass de la capa comprimida:**

$$
y = \hat{W} x + b = C[a] \, x + b
$$

donde $C[a]$ indexa el codebook con los índices de anclaje — esto es exactamente lo que hace `CodebookLinear.forward` en el código (Sección 5).

### 3.2 Métricas estándar (traducción de "GEO_ID")

1. **Error de reconstrucción (fidelidad de los pesos), norma L2 relativa:**
$$
\varepsilon_{\text{recon}} = \frac{\|W - \hat{W}\|_F}{\|W\|_F}
$$

2. **Fidelidad funcional (lo que realmente importa para una red neuronal):** divergencia entre las salidas de la capa original y la comprimida sobre un conjunto de entradas de prueba $\{x_j\}$:
$$
\varepsilon_{\text{func}} = \frac{1}{N}\sum_{j=1}^N \frac{\|f_W(x_j) - f_{\hat{W}}(x_j)\|_2}{\|f_W(x_j)\|_2}
$$

3. **Identificabilidad (recuperación de identidad original), precisión de vecino más cercano:** dado el codeword $c_{a_i}$ asignado a la fila $i$, ¿el vecino más cercano de $c_{a_i}$ entre **todas** las filas originales $\{w_1,\dots,w_{d_{out}}\}$ es efectivamente $w_i$?
$$
\text{Acc}_{\text{ident}} = \frac{1}{d_{out}}\sum_{i=1}^{d_{out}} \mathbb{1}\left[\arg\min_{j} \|c_{a_i} - w_j\|_2 = i\right]
$$
Esta es la métrica que opera como traducción rigurosa de "recuperar la identidad original": mide si el anclaje es lo bastante fino como para servir de "huella digital" que apunta de vuelta a la neurona correcta, o si en cambio varias neuronas colapsan al mismo codeword y la identidad se pierde (caso esperable cuando $K \ll d_{out}$).

## 4. Razonamiento y premisa no establecida

El enunciado original **da por sentado** que el sistema de anclaje "logra recuperar la identidad original". Esto **no se asume como hecho**: es exactamente lo que hay que investigar empíricamente, y matemáticamente hay una tensión estructural conocida:

- La cuantización vectorial es, por construcción, un mapa **muchos-a-uno**: si $K < d_{out}$ (que es el caso de interés — si no, no hay compresión), el pigeonhole principle garantiza que múltiples filas $w_i$ comparten codeword. Para esas filas, **es matemáticamente imposible** recuperar la identidad original a partir únicamente del codeword: la información se perdió en la cuantización.
- Por lo tanto, se espera que $\text{Acc}_{\text{ident}} < 1$ en general, y que decrezca monótonamente a medida que la tasa de compresión $d_{out}/K$ aumenta. El código mide esto explícitamente barriendo distintos valores de $K$, en vez de reportar un único número que podría ocultar esta relación.
- Este es un ejemplo directo de la regla del curso: no se demuestra como cierta una afirmación que el enunciado da por sentada; se investiga como pregunta abierta y, si la premisa no se sostiene (como es de esperar aquí para $K$ chico), se documenta la falla — que en este caso no es una falla numérica sino una **cota de información** inherente al método (relacionada con la desigualdad de tasa-distorsión de Shannon para cuantización).

## 5. Desarrollo y código

Ver `tarea_55_codebook_linear.py`. Contiene:

1. **`CodebookLinear(nn.Module)`**: clase nativa de PyTorch que reemplaza a `nn.Linear`. Constructor recibe `in_features`, `out_features`, `num_codewords` (K). Método `from_linear(linear_layer, num_codewords)` para comprimir una capa `nn.Linear` ya entrenada (k-means sobre las filas de su `.weight`, vía `sklearn.cluster.KMeans` o un k-means propio si no está disponible). `forward(x)` reconstruye $\hat{W} = C[a]$ y aplica la transformación lineal.
2. **`evaluar_compresion(...)`**: calcula $\varepsilon_{\text{recon}}$, $\varepsilon_{\text{func}}$ (con entradas aleatorias de prueba) y $\text{Acc}_{\text{ident}}$.
3. **Barrido experimental**: prueba $K \in \{4, 8, 16, 32, 64\}$ para $d_{out}=128$ y reporta cómo cae $\text{Acc}_{\text{ident}}$ y sube $\varepsilon_{\text{recon}}$ a medida que $K$ se aleja de $d_{out}$ — verificando empíricamente la premisa de la Sección 4 en vez de asumirla.

**No se ejecuta como parte de la entrega** (solo sanity-check de sintaxis). Para correrlo vos mismo:

```bash
pip install torch numpy scikit-learn --break-system-packages
python3 tarea_55_codebook_linear.py
```

Salida esperada: una tabla con $K$, $\varepsilon_{\text{recon}}$, $\varepsilon_{\text{func}}$ y $\text{Acc}_{\text{ident}}$ para cada valor de $K$ probado, más una conclusión textual sobre en qué régimen de $K$ la "recuperación de identidad" es viable y en cuál no.

## 6. Conclusión

El núcleo matemático real detrás de "Anclaje de Codebook" y "GEO_ID" es cuantización vectorial de pesos (compresión de redes, Han et al. 2015) y el problema de identificabilidad que surge de todo mapa de cuantización muchos-a-uno. La afirmación del enunciado original ("logra recuperar la identidad original") **no es universalmente cierta**: depende críticamente de la razón de compresión $d_{out}/K$, y para compresiones agresivas (que son las únicas interesantes en la práctica) es matemáticamente imposible por el principio del palomar. El código deja esto verificable empíricamente corriendo el barrido de $K$ vos mismo.
