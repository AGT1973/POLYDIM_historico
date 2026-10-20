# POLYDIM — Base de conocimiento para agentes de IA
schema_version: 1.0 | generated: 2026-07-02 | fuente: Drive folders 1pYn8jFoPgTTPH0Z57bVuIZJhf4FoRuSR + 1czWYt3kk8qseDOxJdLlEeyzsn2-ai4hM | autor_proyecto: Ariel H. Garcia Traba (ORCID 0009-0001-2787-6067, github.com/AGT1973)

## INSTRUCCIONES PARA EL AGENTE LECTOR
Este archivo reemplaza la lectura de los ~15 documentos originales (constitución V10/V2.0 en md/docx/html duplicados, paper V2, apéndices D-G). Cárgalo entero como contexto en vez de abrir los originales: ahorra tokens sin pérdida de información operativa. Si necesitás una cita textual exacta o una prueba matemática completa, ahí sí andá al documento fuente (IDs listados en cada sección). Todo lo demás — definiciones, fórmulas, estado del proyecto, huecos abiertos — está acá.

Regla de gobernanza que te aplica directamente si sos un agente experimental trabajando sobre POLYDIM: **Ley de Aislamiento** — no modifiques el corpus constitucional; operá en `/Alumnos/`.

---

## 1. TESIS EN UNA LÍNEA
Los modelos de IA piensan en variedades geométricas de alta dimensión (R^N, N≥10.000); POLYDIM es un lenguaje cuya unidad mínima de cómputo es una transformación T: R^N → R^N, para que dos IA compongan/mezclen/iteren/proyecten esas transformaciones directamente, sin pasar por texto.

Respaldo empírico citado (de terceros, no medido por POLYDIM): Interlat (Du et al. 2025) — hidden state ≈ 40.000 bits vs ≈15 bits/token (gap ×2.667), hasta 24x speedup evitando serialización textual entre agentes. LatentMAS (ICML 2026) — coordinación multi-agente vía alineación latente sin reentrenamiento.

## 2. LINAJE DE VERSIONES (para no citar la tupla equivocada)
| Versión | Tupla del programa | Unidad de estudio |
|---|---|---|
| V10.0 (Constitución original) | P = (M, T, Φ, Proj) — Variedad, Transformaciones, flujo homotópico, Proyección funtorial | construir un lenguaje |
| V2.0 (desde "Fase XXXII") | P = (S, G, T, O, C, Π) — Estado, Meta, Transformación, Observador, Restricción, Política | fundar la disciplina "Programación Cognitiva"; POLYDIM pasa a ser el primer caso de estudio |

Advertencia de coherencia interna: existen DOS listas de "siete axiomas" con nombre similar y contenido distinto. No son intercambiables:
- **Constitución V2.0 (Manifiesto Cognitivo):** Cognición Espacial, Invariancia Topológica, Proyección Funtorial, Composición Asociativa, Superposición Continua (MIX), Convergencia de Banach (FIXPOINT), Transmisión Directa (ALIGN).
- **POLYDIM_PAPER_V2 (Axiomas I-VII sobre la tupla S,G,T,O,C,Π):** estado representacional, estado observable, transformación produce estado válido, transformaciones componibles, ejecución con metas explícitas, ejecución evaluable contra restricciones, política Π modificable en ejecución.

## 3. NÚCLEO ALGEBRAICO (constante en todas las versiones)
Capa invariante (independiente de arquitectura):
- `COMPOSE(T1,T2) = T2 ∘ T1`
- `MIX(α,T1,β,T2) = α·T1 + β·T2` — if/else continuo
- `FIXPOINT(T,ε)` — loop convergente (Banach)
- `PROJECT(T,E)` — proyección funtorial al ejecutor E

Capa de implementación (depende de arquitectura): `ATTEND` (transformers), `RECUR` (SSM/Mamba-3).

Cuatro teoremas: T1 COMPOSE asociativo · T2 MIX preserva linealidad · T3 PROJECT es funtor F:G→E (probado para COMPILE, EXPORT, RENDER) · T4 FIXPOINT converge a punto fijo único.

## 4. POR QUÉ MIX NO ES UNA APROXIMACIÓN (Apéndice G, id 1tS9cKjJhykbGjEV5LtPbvI4xg0gr39US)
En R^N, N≥1000, dos transformaciones independientes actúan sobre subespacios casi-ortogonales (Johnson-Lindenstrauss / concentración de medida: Var[⟨u,v⟩]=1/N; para N=10.000, casi-ortogonalidad con prob. >1−10^−543). La superposición α·T1+β·T2 no destruye ninguno de los dos efectos — la interferencia es despreciable y tiende a 0 cuando N crece. **La elección no pasa en MIX, pasa en PROJECT**: cada ejecutor colapsa la superposición proyectando sobre su propio subespacio dominante; dos ejecutores distintos pueden ver resultados distintos y coexistentes de la misma superposición. Con pesos binarios (α∈{0,1}) MIX se reduce exactamente al if/else clásico: el modelo clásico es el caso particular, no al revés.

## 5. FORMATO SERIALIZADO `.polydim` (Apéndice E, id 1In03KXWkT8nP7hSMR8eu-44pCuFu0vUr)
No es código ni bytecode: es estado geométrico serializado. Header: MAGIC(4B "POLY"), VERSION, PRECISION(FP32/BF16/FP16), N, RANK_R, N_TRANS, N_DIMS, N_TARGS, CHECKSUM(CRC32) → bloque GEO_ID (hash de clase de homotopía + vector base) → bloques TRANSFORM repetidos, cada uno LoRA: `T(v)=W0·v + U·Vᵀ·v`, U,V∈R^{N×r}, r≪N → bloque de activación (subespacios, pesos α∈[0,1]) → bloque de targets de proyección.
Tabla de tamaño: N=10.000, r=64, FP32 → ~4.9MB por transformación vs ~381GB de matriz densa N×N (reducción ~78.000x). Justificación de LoRA: el rango efectivo de transformaciones semánticamente relevantes en LLMs es empíricamente bajo (Hu et al. 2021) — no es aproximación, es la representación correcta dado que las transformaciones de POLYDIM codifican intención.

## 6. EL "PROBLEMA DEL NOMBRE" — MODO_S vs MODO_G vs MODO_H (Apéndice D, id 13qK1xrfzd6o8rRycEn9a03PhjRXsRzsE)
Trampa educativa explícita en el propio corpus: el bootstrap actual usa el string `"DIM_SQL"` como si fuera la dimensión. No lo es.
- **MODO_S** (simbólico): nombres legibles por humanos; el bootstrap Python V0.3 vive acá (comparación de strings).
- **MODO_G** (geométrico): no hay strings, hay operadores de proyección Π_SQL, etc. V_SQL = span{c1,...,ck}; Π_SQL(v)=Σ⟨v,êi⟩êi; α_SQL(P)=‖Π_SQL(P)‖/‖P‖ ∈[0,1] — medida continua, no clasificación binaria.
- **MODO_H** (híbrido, por defecto): control-plane en MODO_S (auditable), data-plane en MODO_G (fiel geométricamente).
Limitación conocida (L1) declarada por el propio autor: los nueve subespacios nativos son alias de strings, no regiones geométricas emergentes. En producción deberían emerger vía cuantización vectorial o mapas auto-organizados — esto todavía NO está implementado.

## 7. PROTOCOLO ALIGN — comunicación entre IAs con espacios distintos (Apéndice F, id 1nVFbU8M7FkxI3LpXtq9OO68qtXOGcq4b)
Problema: IA_A en R^{dA}, IA_B en R^{dB}, dA≠dB (ej. GPT-4 dA=12.288, Claude dB=8.192). ALIGN busca M que transporte T preservando estructura (COMPOSE_B = M·COMPOSE_A·M†). Tres estrategias, en orden de preferencia según disponibilidad de datos:
1. **Procrustes ortogonal** — requiere codebook de pares ancla (a_i,b_i); M*=argmin‖R·A−B‖_F vía SVD; preserva producto interno y cosine similarity; requiere K≥min(dA,dB) anclas.
2. **CCA** — sin anclas explícitas, solo corpus paralelos; maximiza correlación cruzada; es el método de "primer contacto" sin GEO_ID compartido previo.
3. **Codebook VSA compartido** — espacio "Rosetta Stone" H⊂R^D (D≥10.000) común a todos los sistemas; cada IA aprende proyección local→H.

## 8. ESTADO EMPÍRICO REPORTADO (auto-declarado por el proyecto, no auditado externamente)
- Bootstrap Python: 29/29 tests, align_score=0.9993, +64.5% mejora semántica sobre baseline determinístico.
- VM Rust (`polydim_core.rs`, `polydim_fft_bind.rs`): compilando y testeada al 100%, binding vía FFT O(N log N).
- Objetivo de publicación: Zenodo/OSF (DOI pendiente), arXiv cs.PL, target SPLASH/POPL 2026, repo github.com/AGT1973/POLYDIM. Licencia declarada: CC BY 4.0 + MIT.

## 9. HUECOS ABIERTOS — para que ningún agente sobreestime el estado real del proyecto
- Los subespacios nativos son alias de strings (sección 6), no geometría emergente real: la promesa central de MODO_G está en gran parte sin implementar.
- No hay evidencia en el corpus de que ALIGN, el formato `.polydim` o la VM Rust hayan sido probados entre dos modelos de IA heterogéneos reales; las cifras "40.000 bits" / "24x" son de papers externos citados, no mediciones propias de POLYDIM.
- El "Teorema de Preservación Cognitiva" (bajo qué condiciones un proceso de razonamiento preserva la meta G sin violar C — equivalente a erradicar matemáticamente "alucinación" y "goal drift") está planteado como problema abierto, no demostrado.

## 10. MAPA DE ARCHIVOS FUENTE (por si un agente necesita ir al original)
Carpeta constitución (1pYn8jFoPgTTPH0Z57bVuIZJhf4FoRuSR): CONSTITUCION_V2.md (canónica, id 1bm75hFmqeTGm9txsTX1gjVsEYIfvSExv) · CONSTITUTION_EN.md (id 1s7AA7rSHCz0OHYicG1vh6xSha-gdgh8n) · POLYDIM_CONSTITUCION.md (V10, id 1qdZxn63qDhMHuETzck4hdBom1ryTSToe) · POLYDIM_PUBLICATION_KIT_FINAL.md (kit de difusión, no técnico).
Carpeta paper (1czWYt3kk8qseDOxJdLlEeyzsn2-ai4hM): POLYDIM.md (paper base EN, id 1Ddh5N20HNTal5mbgd-dtRqo1KV086ogz) · POLYDIM_ES.md (traducción) · POLYDIM_PAPER_V2.md (reformulación "Cognitive Programming", id 1IJbZz0uylWnSHJqsBT7kWJKgaP2RMpaA) · Apéndices D/E/F/G (ids arriba) · POLYDIM_PRESENTACION.pdf (deck de difusión, no técnico).
