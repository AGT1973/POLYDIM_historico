# POLYDIM_DEST
# destino: polydim/research/
# nombre:  RESEARCH_SUBSPACE_EMERGENCE_V1.md
# autor:   ai.mpat.agt@gmail.com
# fecha:   2026-06-27
# tarea:   TASK_R01

---

# POLYDIM — Emergencia Automática de Subespacios
## Q1: ¿Cómo emergen los subespacios de un modelo real sin definirlos manualmente?

**Estado:** DISEÑO DEL EXPERIMENTO — pendiente ejecución con GPU
**Relevancia académica:** Q1 de Open Problems (Sec 8.3 del paper)
**Si el experimento confirma la hipótesis:** base empírica para afirmar que los 9 subespacios de POLYDIM son propiedades intrínsecas de los LLMs, no definiciones arbitrarias.

---

## SECCIÓN 1 — HIPÓTESIS

**H (Emergencia):** Los 9 subespacios nativos de POLYDIM (DIM_SQL, DIM_FLUTTER, DIM_PYTHON, etc.) no son definiciones arbitrarias del diseñador. Son regiones de alta densidad semántica que emergen naturalmente del espacio de activaciones de cualquier LLM de suficiente capacidad.

Formalmente:
```
Sea M un LLM con espacio de activaciones R^d (d ≥ 768).
Sea A = {a_1, ..., a_n} el conjunto de activaciones de M sobre un corpus C.
Sea K = k-means(A, k=9).

Hipótesis: ∃ una biyección φ: K → {DIM_PYTHON, DIM_SQL, ..., DIM_META}
           tal que φ(K_i) es semánticamente coherente con el nombre del subespacio.
```

**Por qué esperamos que sea verdad:**
La arquitectura transformer aprende representaciones distribuidas donde conceptos relacionados ocupan regiones similares del espacio latente. Un corpus suficientemente diverso debería activar regiones distintas para código Python, queries SQL, componentes UI, embeddings, etc.

---

## SECCIÓN 2 — DISEÑO EXPERIMENTAL

### 2.1 Corpus de activaciones

Generar un corpus balanceado con exactamente 100 ejemplos por categoría semántica:

```python
CORPUS = {
    "python_code":    100,  # fragmentos de código Python
    "sql_queries":    100,  # queries SELECT, INSERT, UPDATE
    "flutter_widgets":100,  # código Dart/Flutter
    "vector_ops":     100,  # operaciones de embedding, FAISS
    "graph_queries":  100,  # Cypher, GraphQL, NetworkX
    "time_series":    100,  # pandas timeseries, datetime ops
    "error_traces":   100,  # stack traces, exception messages
    "metadata":       100,  # JSON de metadatos, YAML configs
    "rust_code":      100,  # código Rust, ownership, lifetimes
    "mixed":          100,  # texto general, no categorizable
}
# Total: 1000 ejemplos, con etiquetas ground truth
```

### 2.2 Extracción de activaciones

```python
from transformers import GPT2Model, GPT2Tokenizer
import numpy as np

def extract_activations(texts: list[str], layer: int = -2) -> np.ndarray:
    """
    Extrae activaciones de la penúltima capa de GPT-2.
    layer=-2: última capa oculta (más semántica que la última, menos ruidosa)
    
    Returns: float[n_texts × hidden_size]
    """
    model = GPT2Model.from_pretrained('gpt2')  # d=768
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    activations = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt',
                          truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Promedio sobre tokens de la capa especificada
        hidden = outputs.hidden_states[layer]  # [1, seq_len, d]
        mean_activation = hidden[0].mean(0).numpy()  # [d]
        activations.append(mean_activation)
    
    return np.array(activations)  # [n × d]
```

### 2.3 Clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

def run_emergence_experiment(activations: np.ndarray, 
                             true_labels: np.ndarray,
                             k_range: range = range(2, 20)) -> dict:
    """
    Experimento completo de emergencia de subespacios.
    
    1. k-means para k=2..19 (encontrar k óptimo por silhouette)
    2. Para k=9: comparar clusters con etiquetas ground truth
    3. Si ARI > 0.5: hipótesis confirmada
    """
    # Paso 1: encontrar k óptimo
    silhouette_scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(activations)
        s = silhouette_score(activations, labels)
        silhouette_scores[k] = s
    
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    
    # Paso 2: clustering forzado a k=9
    km9 = KMeans(n_clusters=9, random_state=42, n_init=20)
    predicted_labels = km9.fit_predict(activations)
    
    # Paso 3: métricas de coincidencia con ground truth
    ari = adjusted_rand_score(true_labels, predicted_labels)
    
    # Paso 4: ¿qué categoría domina cada cluster?
    cluster_semantics = {}
    for cluster_id in range(9):
        mask = predicted_labels == cluster_id
        # Encontrar la categoría más frecuente en este cluster
        cats_in_cluster = true_labels[mask]
        dominant_cat = np.bincount(cats_in_cluster).argmax()
        purity = (cats_in_cluster == dominant_cat).mean()
        cluster_semantics[cluster_id] = {
            "dominant_category": dominant_cat,
            "purity": purity,
        }
    
    return {
        "optimal_k": optimal_k,
        "silhouette_at_k9": silhouette_scores.get(9, 0),
        "adjusted_rand_index": ari,
        "cluster_semantics": cluster_semantics,
        "hypothesis_confirmed": ari > 0.5,
    }
```

### 2.4 Análisis de correspondencia

Una vez obtenidos los 9 clusters, mapearlos a los subespacios POLYDIM:

```python
EXPECTED_MAPPING = {
    "python_code":     "DIM_PYTHON",
    "sql_queries":     "DIM_SQL",
    "flutter_widgets": "DIM_FLUTTER",
    "vector_ops":      "DIM_VECTOR",
    "graph_queries":   "DIM_GRAPH",
    "time_series":     "DIM_TIME",
    "error_traces":    "DIM_ERROR",
    "metadata":        "DIM_META",
    "rust_code":       "DIM_RUST",
}

def map_clusters_to_dims(cluster_semantics, category_names) -> dict:
    """
    Intenta encontrar la biyección φ: clusters → DIM_X.
    Si la categoría dominante de K_i coincide con DIM_X_expected,
    la correspondencia es confirmada.
    """
    mapping = {}
    for cluster_id, info in cluster_semantics.items():
        cat_name = category_names[info["dominant_category"]]
        expected_dim = EXPECTED_MAPPING.get(cat_name, "DIM_META")
        mapping[cluster_id] = {
            "cluster": cluster_id,
            "predicted_polydim_dim": expected_dim,
            "purity": info["purity"],
            "confirmed": info["purity"] > 0.6,
        }
    return mapping
```

---

## SECCIÓN 3 — RESULTADOS ESPERADOS Y CRITERIOS DE ÉXITO

### 3.1 Si ARI > 0.5 (hipótesis confirmada)

Conclusión: Los 9 subespacios de POLYDIM son regiones emergentes del espacio de activaciones de GPT-2. Los nombres (DIM_SQL, DIM_FLUTTER, etc.) son aproximaciones legibles de propiedades geométricas intrínsecas del modelo.

**Implicaciones para el paper:**
- La sección 3.2 puede reformularse: "Los 9 subespacios nativos fueron empíricamente identificados en el espacio de activaciones de GPT-2 y otros LLMs, y luego formalizados como aliases. Los aliases de strings del bootstrap son una aproximación simbólica (MODO_S) a estas regiones geométricas."
- Q1 de Open Problems se cierra parcialmente: los subespacios emergen con k-means k=9 sobre corpus balanceado.

### 3.2 Si ARI ∈ [0.3, 0.5] (hipótesis parcial)

Los subespacios existen pero con overlap. Posibles causas:
- GPT-2 es demasiado pequeño (d=768). Probar con GPT-2 large (d=1280) o Llama-2 (d=4096).
- El corpus no es suficientemente balanceado.
- Algunos subespacios (DIM_ERROR, DIM_META) se solapan naturalmente.

### 3.3 Si ARI < 0.3 (hipótesis no confirmada)

Los subespacios definidos manualmente en POLYDIM no corresponden a regiones emergentes. Esto no invalida el lenguaje — equivale a decir que los subespacios de POLYDIM son constructos útiles pero no naturales. El Bootstrap V0.3 sigue siendo válido como MODO_S simbólico.

---

## SECCIÓN 4 — VARIANTES DE CLUSTERING A PROBAR

| Método | Ventaja | Desventaja |
|---|---|---|
| k-means (k=9) | Simple, reproducible | Asume clusters esféricos |
| HDBSCAN | Detecta k automáticamente | Puede detectar < 9 o > 9 clusters |
| VQ-VAE (k=9) | Entrenado end-to-end | Requiere más datos y GPU |
| SOM (9×1 grid) | Preserva topología | Requiere tuning de hiper-parámetros |

**Recomendación:** Empezar con k-means k=9 (más rápido). Si ARI < 0.4, probar HDBSCAN para ver cuántos clusters emergen naturalmente.

---

## SECCIÓN 5 — HIPÓTESIS SECUNDARIA: LA DIMENSIÓN k

**H2 (k natural):** k-means con k libre (silhouette máximo) converge cerca de k=9 en LLMs de suficiente capacidad.

Para verificar: graficar silhouette vs k para k=2..20. Si el máximo está en k∈[7,12], H2 se confirma parcialmente.

```python
# Script de verificación rápida
import matplotlib.pyplot as plt

k_range = range(2, 20)
scores = [silhouette_score(activations, 
          KMeans(k, random_state=42).fit_predict(activations))
          for k in k_range]

plt.plot(k_range, scores)
plt.axvline(x=9, color='red', linestyle='--', label='k=9 (POLYDIM)')
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Analysis — GPT-2 Activations')
plt.legend()
plt.savefig('silhouette_analysis.png')
```

---

## SECCIÓN 6 — CONEXIÓN CON EL PAPER

Si el experimento confirma H, añadir a la Sección 7.4 (VSA) del paper:

> "The nine native subspaces of POLYDIM are not arbitrary design choices. Clustering experiments on GPT-2 activations using a balanced corpus of 1,000 inputs across nine semantic categories yield a k-means solution (k=9) with Adjusted Rand Index > 0.5 against ground-truth category labels, confirming that the nine dimensions correspond to naturally emerging semantic regions of transformer latent space."

---

*RESEARCH_SUBSPACE_EMERGENCE_V1.md · V1.0 · 2026-06-27 · TASK_R01*
*Estado: DISEÑO COMPLETO — pendiente ejecución con GPU y corpus*
*Impacto: cierre parcial de Q1 (Open Problems Sec 8.3) si ARI > 0.5*
