# Operadores Algebraicos de Proyección Dimensional con Contrato Numérico Explícito para Fusión de Estado Latente entre Agentes LLM Heterogéneos

**Propuesta de frente de investigación — POLYDIM / MPAT_V3_02**
**Estado:** borrador para discusión (v2 - Incluye Teoría de Haces Celulares)
**Relación con el árbol de trabajo:** extiende Tarea 8 (POLYDIM Escuadrón 1 — TDA/Relajación Topológica) y RES.179/RES.228 (identidad de agente, AgentCard)

---

## 1. Resumen ejecutivo

Existen hoy cuatro cuerpos de investigación maduros que nunca se cruzaron entre sí:

1. **Geometría de la reducción dimensional con garantías demostrables** (lema de Johnson-Lindenstrauss y variantes, restricted isometry property / compressed sensing).
2. **Comunicación latente entre LLMs heterogéneos** (fusión de KV-caches, adaptadores entrenados: Cache-to-Cache, Latent Cache Flow, alineación densa vía análisis de compressed-sensing).
3. **Determinismo numérico en cómputo distribuido** (no-asociatividad de punto flotante, deriva por orden de acumulación, ruido estructurado no-gaussiano en GPU).
4. **Teoría de haces celulares (cellular sheaf theory) aplicada a fusión de datos y consenso multi-agente heterogéneo** (sheaf Laplacians, sheaf neural networks, coordinación multi-agente vía ADMM sobre haces).

Cada uno es SOTA en su dominio. Ninguno cita a los otros tres. La intersección —proyectar/fusionar estado latente entre modelos con *dimensiones y contratos numéricos distintos*, con una *cota de distorsión demostrable* y una *estructura formal para medir desacuerdo/deriva*, en vez de un adaptador entrenado ad-hoc— es un hueco de investigación abierto y, tras esta búsqueda, parece tener una respuesta candidata natural: el aparato de haces celulares está construido exactamente para el caso en que distintos nodos llevan espacios vectoriales de distinta dimensión. Este documento fija ese hueco, lo conecta con el trabajo ya hecho en Tarea 8, y propone un experimento mínimo para ocuparlo.

---

## 2. Estado del arte por frente

### 2.1 Frente A — Geometría de la reducción dimensional (garantías formales)

El lema de Johnson–Lindenstrauss establece que un conjunto de puntos en alta dimensión puede proyectarse a una dimensión menor preservando aproximadamente las distancias por pares, con cota de distorsión ε demostrable. 

Puntos clave:
- Las garantías clásicas (JL, RIP) son para proyecciones lineales aleatorias, no aprendidas. 
- Métodos rápidos de proyección (Fast JL Transform) sacrifican la garantía de preservar la energía residual ortogonal al subespacio principal, crítico para estabilizar tareas posteriores.
- Ningún trabajo menciona KV-caches de transformers como objeto a proyectar.

### 2.2 Frente B — Comunicación latente entre LLMs heterogéneos (ingeniería empírica)

- **Cache-to-Cache (C2C):** Fusión de KV-caches entre modelos para lograr comunicación semántica directa. Límite: escalar el número de LLMs con costo O(N) es un problema abierto.
- **Latent Cache Flow (LCF):** Comprime adaptadores transmitiendo solo información nueva.
- **Alineación densa:** Dualidad transferencia dispersa/densa vía análisis de compressed-sensing.

Ningún método ofrece una función algebraica con propiedades demostrables independiente del par de arquitecturas; todos son adaptadores neuronales entrenados.

### 2.3 Frente C — Determinismo numérico y no-asociatividad de punto flotante 

- La no-asociatividad de suma en punto flotante genera variación entre corridas.
- DASH fuerza orden de acumulación para determinismo bit-exacto.
- Ruido estructurado no-gaussiano en GPU.

Toda esta línea estudia deriva numérica dentro de un solo pipeline de entrenamiento/inferencia, no cuando dos modelos distintos fusionan su estado latente.

### 2.4 Frente D — Teoría de haces celulares (cellular sheaf theory) para fusión heterogénea

Un haz celular asigna a cada nodo un espacio vectorial propio (*stalk*), y conexiones mediante *mapas de restricción* (funciones lineales). **Los stalks de nodos distintos pueden tener dimensiones distintas entre sí**. Es el objeto matemático que formaliza "cada agente vive en su propia dimensión, y hace falta una función explícita para traducir entre dimensiones".

- **Sheaf Neural Networks** (Hansen y Gebhart, 2020)
- **Neural Sheaf Diffusion** (Bodnar et al., 2022): Evita el "aplastamiento" (sobre-suavizado) de información.
- **Laplaciano de haz (sheaf Laplacian):** Medidor de "desacuerdo" entre stalks.
- **Sheaf-ADMM (2026):** Coordinación multi-agente donde agentes heterogéneos resuelven subproblemas locales.

El punto crítico: ningún trabajo conecta haces celulares con KV-caches de LLMs. Toda la literatura de comunicación latente heterogénea (Frente B) ignora el aparato formal (Laplaciano) para tratar nodos con espacios de distinta dimensión.

---

## 3. La intersección vacía: A ∩ B ∩ C ∩ D

Ningún paper de los relevados menciona que:

1. **Fusionar KV-caches con contratos numéricos distintos** introduce una deriva idéntica a la que el Frente C caracterizó, aplicada a modelos heterogéneos. (Tesis de Tarea 8).
2. **Los adaptadores de C2C/LCF no tienen cota de distorsión tipo JL/RIP (Frente A).**
3. **El problema de escalado O(N)** es síntoma de la ausencia de proyección canónica intermedia (Frente A).
4. **El Laplaciano de haz ya es un medidor de deriva/desacuerdo entre representaciones heterogéneas (Frente D).** Diferencia: mide desacuerdo estructural, mientras Tarea 8 mide deriva numérica. Conjugar ambas cosas (desacuerdo estructural + deriva numérica) es una formalización más completa.

---

## 4. Extrapolación: forma del operador dimensional POLYDIM

**Modelar el enjambre de agentes como un haz celular**, donde cada agente es una celda con su stalk (KV-cache), y cada canal es un mapa de restricción declarado explícitamente.

Debería declarar:
- **Espacio de origen y destino (stalks)** (d₁, d₂).
- **Mapa de restricción** explícito entre stalks.
- **Contrato numérico** como parámetro explícito adjunto al mapa.
- **Cota de distorsión objetivo (ε)** en el espíritu JL/RIP.
- **Descomposición del Laplaciano de haz** en desacuerdo estructural y deriva numérica espuria.
- **Composabilidad en enjambre:** Cota total de distorsión no descontrolada.

---

## 5. Conexión con el árbol de trabajo existente

- **Tarea 8:** Frente C subido a fusión entre modelos.
- **RES.179 / RES.228:** Contrato numérico como metadata.
- **Escuadrón 1:** Diseño O(N) proyectando a un espacio canónico (topología de haz).
- **Deriva de manifold + Laplaciano:** Separa deriva numérica de desacuerdo estructural genuino.

---

## 6. Experimento mínimo propuesto

**Objetivo:** Determinar si la incompatibilidad de contrato numérico entre dos modelos es una fuente de deriva **medible y separable**.

1. Tomar par de modelos open-weight (ej. Qwen2.5-0.5B / Qwen3-0.6B).
2. Ejecutar fusión de KV-cache bajo condición controlada (mismo hardware, acumulación DASH) y condición realista (precisión/hardware mixto).
3. Medir distancia (L2 y coseno) del estado fusionado.
4. Comparar contra distorsión base del método de fusión.
5. **Paso Frente D:** Calcular Laplaciano de haz simplificado sobre el par. Comparar el valor bajo ambas condiciones para validar si capta la misma deriva que la distancia L2.

---

## 7. Próximos pasos sugeridos

1. Registrar como RES nuevo o anexo v2 de Tarea 8.
2. Montar experimento mínimo de sección 6.
3. Evaluar extensión de metadata `AgentIdentityRecord` (RES.228).
4. Decidir si abrir línea de estudio sobre teoría de haces celulares (Frente D) para dimensionar la curva de entrada (cohomología, Laplaciano).

---

*(Referencias incluidas en documento base)*
