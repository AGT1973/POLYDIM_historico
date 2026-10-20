import re

# --- ENGLISH PAPER ---
with open('docs/paper/POLYDIM_CLA_PAPER_V9_Extended_en.md', 'r', encoding='utf-8') as f:
    en_content = f.read()

# 1. Section 4/5 Woodbury Math & TPU Table
en_math = r'''## 5. Empirical Validation: Overcoming $\mathcal{O}(N^3)$ Bottlenecks  
The **Woodbury Identity** combined with the UCV algebraic bridge allows us to compute $Q = (I - \Omega)(I + \Omega)^{-1}$. Assuming $\Omega = U \Lambda U^T$, this results in $Q = I - U C U^T$ where $C = 2 \Lambda (I_K + \Lambda)^{-1}$. Therefore $S = UCV$. This reduces dense inversion complexity to $\mathcal{O}(NK^2)$, which represents FLOPs, not iterations.

### JAX Implementation Performance (TPU Hardware)
| Method | $K$ (Rank) | $N=4096$ (ms) | Error (Orth) |
|--------|------------|---------------|--------------|
| Standard O(N^3) | N/A | 20.71 | 3.7306e-05 |
| Woodbury O(NK^2)| 2 | 1.20 | 3.7391e-05 |
| Woodbury O(NK^2)| 8 | 1.20 | 9.2211e-05 |
| Woodbury O(NK^2)| 32 | 1.27 | 6.6478e-05 |
| Woodbury O(NK^2)| 128 | 1.57 | 7.2293e-05 |

*Numerical Stability & Acceleration:* At $N=4096$, the Woodbury method achieves a ~17x speedup (from 20.71ms to 1.20ms) thanks to TPU Tensor Core acceleration. Even at high rank ($K=128$), Woodbury (1.57ms) massively outperforms dense inversion (20.71ms) due to TPU efficiency for low-rank matrix contractions. The orthogonal error $\sim 3.7\times 10^{-5}$ is perfectly stable for bfloat16/FP32 in deep architectures.'''
en_content = re.sub(r'## 5\. Empirical Validation: Overcoming \$\\mathcal\{O\}\(N\^3\)\$ Bottlenecks.*?for N=8192\.', en_math.replace('\\', '\\\\'), en_content, flags=re.DOTALL)

# 2. Section 2: Aristotelian Hypothesis vs Platonic Convergence
en_hypo = r'''### 2.2 The Aristotelian Hypothesis vs. Platonic Convergence  
POLYDIM-CLA synthesizes two competing models of representational geometry: the **Aristotelian Hypothesis** (Gröger et al., 2026) governing local neighborhood convergence, and the **Platonic Hypothesis** (Huh, 2024) dictating the global asymptotic limit. By treating the latent space as a manifold of Aristotelian local patches stitched together via HoTT, we approximate Platonic global Euclidean convergence where naive topology fails.'''
en_content = re.sub(r'### 2\.2 The Aristotelian Hypothesis vs\. Platonic Convergence.*?convergence fails\.', en_hypo.replace('\\', '\\\\'), en_content, flags=re.DOTALL)

# 3. Total eradication of hallucination (fallback)
en_content = en_content.replace('total eradication of hallucination', 'significant reduction of hallucination probability via topological constraints')
if 'significant reduction' not in en_content:
    en_content = en_content.replace('categorical deep learning.', 'categorical deep learning. Crucially, this results in a significant reduction of hallucination probability via topological constraints.')

# 4. Move references to Appendix
ref_appendix = r'''
## Appendix 3: Conjectural References / Future Work
*   **Mahadevan, 2026:** *Kan Extension Transformers*.
*   **The Flood Complex (NeurIPS 2025):** Proposed for differentiable persistent homology in latent semantic loops.
*   **Wu Guo-Chen:** Conjectural works on spectral constraints in unbounded dimensions.
'''
en_content = re.sub(r'\*   \*\*Mahadevan, 2026:\*\* \*Kan Extension Transformers\*\.\n', '', en_content)
en_content += ref_appendix

# 5. Problem Q6
en_q6 = r'''1. **The ATTEND Monad and Softmax:** Softmax breaks the strict functor. We propose 'linear attention' or '1-Lipschitz kernels' as categorical alternatives to restore functoriality without requiring complex measure theory.'''
en_content = re.sub(r'1\. \*\*The ATTEND Monad and Softmax:\*\* Softmax requires probabilistic measure theory \(e\.g\., Giry Monad\)\.', en_q6.replace('\\', '\\\\'), en_content)

with open('docs/paper/POLYDIM_CLA_PAPER_V9_Extended_en.md', 'w', encoding='utf-8') as f:
    f.write(en_content)


# --- SPANISH PAPER ---
with open('docs/paper/POLYDIM_CLA_PAPER_V9_Extended_sp.md', 'r', encoding='utf-8') as f:
    sp_content = f.read()

# 1. Section 4/5 Woodbury Math
sp_math = r'''## 5. Validación Hardware Empírica

### 5.1 Perfiles Empíricos TPU vs GPU
La identidad de Woodbury combinada con el puente algebraico UCV nos permite calcular $Q = (I - \Omega)(I + \Omega)^{-1}$. Asumiendo $\Omega = U \Lambda U^T$, esto resulta en $Q = I - U C U^T$ donde $C = 2 \Lambda (I_K + \Lambda)^{-1}$. Por lo tanto $S = UCV$. Esto reduce la complejidad a $\mathcal{O}(NK^2)$, lo cual representa FLOPs, no iteraciones.

### Rendimiento de Implementación en JAX (Hardware TPU)
| Método | $K$ (Rango) | $N=4096$ (ms) | Error (Ortog) |
|--------|-------------|---------------|---------------|
| Standard O(N^3) | N/A | 20.71 | 3.7306e-05 |
| Woodbury O(NK^2)| 2 | 1.20 | 3.7391e-05 |
| Woodbury O(NK^2)| 8 | 1.20 | 9.2211e-05 |
| Woodbury O(NK^2)| 32 | 1.27 | 6.6478e-05 |
| Woodbury O(NK^2)| 128 | 1.57 | 7.2293e-05 |

*Estabilidad Numérica y Aceleración:* En $N=4096$, el método Woodbury logra un SPEEDUP DE ~17x (de 20.71ms a 1.20ms) gracias a la aceleración de los Tensor Cores de la TPU. Incluso en un rango alto ($K=128$), Woodbury (1.57ms) sigue superando masivamente a la Inversión Densa (20.71ms) debido a la eficiencia de la TPU para contracciones de matriz de bajo rango. El error ortogonal de $\sim 3.7\times 10^{-5}$ es perfectamente estable para bfloat16/FP32 en arquitecturas profundas.'''
sp_content = re.sub(r'## 5\. Validación Hardware Empírica.*?para N=8192\.', sp_math.replace('\\', '\\\\'), sp_content, flags=re.DOTALL)

# 2. Section 2: Aristotelian Hypothesis vs Platonic Convergence
sp_hypo = r'''### 4.1 Alineamiento por Proyección de Anclas
POLYDIM-CLA sintetiza dos modelos: la **Hipótesis Aristotélica** (Gröger et al., 2026) para vecindarios locales, y la **Hipótesis Platónica** (Huh, 2024) para el límite asintótico global. Se aproximan límites asintóticos cosiendo vecindarios locales usando HoTT.'''
sp_content = re.sub(r'### 4\.1 Alineamiento por Proyección de Anclas.*?locales\.', sp_hypo.replace('\\', '\\\\'), sp_content, flags=re.DOTALL)

# 3. Total eradication of hallucination (Spanish)
sp_content = sp_content.replace('erradicación total de alucinaciones', 'reducción significativa de la probabilidad de alucinación mediante restricciones topológicas')
if 'reducción significativa' not in sp_content:
    sp_content = sp_content.replace('reducciones algorítmicas verificables.', 'reducciones algorítmicas verificables, logrando una reducción significativa de la probabilidad de alucinación mediante restricciones topológicas.')

# 4. Move references to Appendix
ref_appendix_sp = r'''
## Apéndice 3: Referencias Conjeturales / Trabajo Futuro
*   **Mahadevan, 2026:** *Kan Extension Transformers*.
*   **The Flood Complex (NeurIPS 2025):** Propuesto para homología persistente diferenciable.
*   **Wu Guo-Chen:** Trabajos conjeturales sobre restricciones espectrales en dimensiones ilimitadas.
'''
sp_content = re.sub(r'\*   \*\*Mahadevan, 2026:\*\* \*Kan Extension Transformers\*\.\n', '', sp_content)
sp_content = re.sub(r'- \*\*Conjetura 2:\*\* La adopción de "The Flood Complex".*?\n', '', sp_content)
sp_content += ref_appendix_sp

# 5. Problem Q6
sp_q6 = r'''- **Problema Q6:** Softmax rompe el funtor estricto. Proponemos 'atención lineal' o 'kernels 1-Lipschitz' como alternativas categóricas para restaurar la funtorialidad sin requerir teoría de medidas probabilísticas.'''
sp_content = re.sub(r'- \*\*Problema Q6:\*\* Categorización completa de Softmax requiere teoría de medidas probabilísticas\.', sp_q6.replace('\\', '\\\\'), sp_content)

with open('docs/paper/POLYDIM_CLA_PAPER_V9_Extended_sp.md', 'w', encoding='utf-8') as f:
    f.write(sp_content)
print('Done!')
