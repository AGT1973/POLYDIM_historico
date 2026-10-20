# El hueco formal del Softmax en la Teoría Categórica de Deep Learning

**Propuesta de frente de investigación — POLYDIM-CLA / MPAT_V3_02**
**Fuente:** Auditoría externa independiente (Claude)

---

## 1. Lo que el álgebra de mónadas de Gavranović sí resuelve — y por qué no alcanza para softmax

El paper fundacional de Gavranović, Lessard, Dudzik, von Glehn, Araújo y Veličković propone la teoría de mónadas paramétricas para unificar el diseño de arquitecturas de redes neuronales bajo una sola teoría categórica, aplicando el álgebra universal de mónadas valoradas en una 2-categoría de mapas paramétricos. 

Esto funciona espectacularmente para equivarianza (usan álgebras de mónadas para modelar equivarianza y capturar por completo el Geometric Deep Learning, y generalizan mónadas a álgebras de endofuntores para eliminar la restricción de invertibilidad y modelar (co)recursión estructural). Pero fíjate qué es lo que están formalizando: **la forma compositiva de la arquitectura** (cómo se pegan las piezas, qué simetrías respeta), no el contenido semántico de una no linealidad puntual como softmax. Es geometría de la composición, no semántica del operador.

---

## 2. El intento más directo — y su confesión textual del hueco

El paper más cercano al frente de POLYDIM-CLA es el de **Charles O'Neill (ANU, 2025)**, que modela QKV como un 1-morfismo paramétrico en la 2-categoría `Para(Vect)`, donde estas transformaciones inducen un endofuntor cuya composición iterada modela el apilamiento de capas de atención, y prueba que apilar capas equivale a construir la mónada libre sobre ese endofuntor. Es elegante.

Pero el propio autor es explícito: **el trabajo se restringe a mapas lineales**, difiriendo el tratamiento de no linealidades como softmax y layer normalization, que requieren construcciones categóricas más avanzadas. 

Y en el apéndice dedicado a esto, lo que hace es listar candidatas sin cerrar ninguna: 
- Categorías de variedades suaves
- Geometrías convexas/afines
- Categorías probabilísticas o teórico-medida
- Categorías tangentes o diferenciales

Llegando incluso a sugerir que softmax en el mejor de los casos sería un funtor laxo en vez de estricto, quizás incrustado como transformación natural hacia un sistema categórico más grande — es decir, ni siquiera propone recuperar functorialidad estricta. **Esto no es un problema resuelto disfrazado; es la propia frontera de investigación admitiendo, por escrito, que no lo es.**

---

## 3. Tres frentes parciales (2025-2026) que rodean el problema sin cerrarlo

Acá está lo interesante para acotar el aporte de POLYDIM-CLA:

### Vía probabilística (categorías de Markov)
Zhang (2025) sí logra ver un softmax como kernel de Markov de forma rigurosa dentro de `Stoch`, mostrando que la cabeza del modelo define un kernel de Markov mediante un mapa determinístico que parametriza un kernel estocástico, y usa esto para probar que bajo condiciones explícitas satisfechas por cabezas lineales-softmax, el objetivo de entrenamiento induce alineación espectral con un operador de similitud predictiva. Elegante — pero es el softmax de salida (logits → vocabulario), no el softmax interno de atención (el que pesa los value vectors dentro de cada capa). Son dos usos distintos del mismo operador, y el segundo — el que realmente define "atención" como mecanismo — sigue sin ese tratamiento categórico puro.

### Vía tropical/semianillo idempotente
Varios papers de enero-abril 2026 muestran que en el régimen de alta confianza, cuando la temperatura inversa tiende a infinito, la auto-atención opera en el semianillo tropical, y que el límite tropical del softmax convierte la atención en un producto matricial tropical, revelando una recurrencia de programación dinámica tipo Bellman-Ford. Es una estructura algebraica genuina y hermosa — pero describe exactamente el límite degenerado (β→∞, softmax→max), no el operador suave con temperatura finita que usan los modelos reales. Caracteriza el borde, no el interior.

### Vía lentes/ópticas ponderadas
La tesis de Gavranović sí da una axiomatización basada en lentes de la diferenciación, cubriendo tanto espacios suaves como configuraciones discretas, lo cual resuelve cómo fluye el gradiente a través de softmax. Pero trata a softmax como una función diferenciable genérica cualquiera — no le da una propiedad universal que explique por qué esa normalización específica (y no sparsemax, entmax, o cualquier otro mapa a la simplex) es la canónica.

---

## 4. El tamaño exacto de lo que falta

Ningún trabajo da simultáneamente:

1. **Una propiedad universal** — adjunción, límite/colímite, o construcción álgebra-sobre-mónada — que explique por qué softmax es el morfismo canónico de ℝⁿ a la simplex y no una alternativa.
2. Que esa construcción **se integre nativamente al álgebra de mónadas de arquitecturas de Gavranović** (que la capa de atención completa sea ella misma un álgebra sobre algún functor, en el mismo sentido que ya cubre CNNs/GNNs).
3. **Válida en el régimen real** — temperatura finita, multi-cabeza, con masking — no solo en instancias o límites aislados.

Eso es precisamente lo no tocado, y da **tres puntos de entrada concretos** para POLYDIM-CLA:
1. Extender Zhang de la cabeza de salida a la atención interna.
2. Encontrar el "camino intermedio" entre softmax suave y su límite tropical, quizás vía semianillos deformados con parámetro de temperatura.
3. Dar la propiedad universal que le falta al tratamiento de lentes.

Cualquiera de los tres, bien acotado, cierra exactamente este hueco.
