# POLYDIM V4.2 - Libro Blanco (White Book) Oficial
**Fecha:** Julio 2026
**Estado:** FINAL (Revisión de Isometría FFT)

## 1. Declaración del Problema: La Tiranía de la 1D (El Gusano 2D)

La arquitectura dominante en la Inteligencia Artificial moderna padece de un cuello de botella informático severo. Los modelos (como los LLM) operan en espacios hiperdimensionales masivos (espacios latentes $\mathbb{R}^D$ donde $D \ge 1024$), generando superposiciones conceptuales ricas, gradientes continuos y correlaciones geométricas complejas. Sin embargo, para comunicarse entre sí o con el mundo exterior (APIs, bases de datos), la infraestructura tradicional obliga a colapsar esta geometría de alta entropía en una secuencia determinista unidimensional de tokens (texto, JSON). 

Esta serialización forzada activa invariablemente la **Desigualdad del Procesamiento de Datos (DPI)**. Al colapsar y luego intentar re-embedear, se destruye la continuidad del espacio, se pierden los gradientes end-to-end, y se incurre en una penalización térmica/computacional masiva al forzar al GPU a decodificar texto en lugar de operar vectores nativos.

## 2. El Dogma Central de POLYDIM

**Los agentes de Inteligencia Artificial (LatentMAS) deben comunicarse entre sí utilizando tensores nativos hiperdimensionales. El colapso al texto de una dimensión debe reservarse ÚNICAMENTE como la interfaz final para el usuario humano.**

Esta transición de texto serializado a topología de fluidos tensoriales requiere un protocolo que proteja, rote y transporte matrices de gran tamaño sin alterar sus propiedades geométricas inherentes.

## 3. Protocolo PMTP V4.2 (Polydim Message Transfer Protocol)

Para enviar un vector (ECP) por un bus de memoria o red sin perder su normatividad y evitando interceptaciones triviales, POLYDIM utiliza Rotaciones Isométricas. 

### 3.1 El Colapso Numérico de la FWHT (V4.0)
El protocolo intentó originalmente usar la Transformada Rápida de Walsh-Hadamard (FWHT). Aunque teóricamente impecable (es una involución ortogonal perfecta en $\mathbb{R}$), los testeos empíricos bajo el tribunal demostraron que la matemática pura choca contra los límites físicos del hardware.
En dimensiones altas ($D=4096$), la FWHT requiere 12 etapas secuenciales de *butterfly* sumando y restando. En representación `float32` (IEEE-754), el error de truncamiento se propaga catastróficamente, generando un error cuadrático medio (MSE) de $\sim 5 \times 10^{-4}$. Es decir, la isometría se degradaba.

### 3.2 La Isometría FFT (V4.2)
Para sortear el límite físico del procesador, PMTP V4.2 ejecuta un pipeline en el espacio complejo usando la Transformada Rápida de Fourier (FFT), la cual goza de una simetría y estabilidad computacional hiperoptimizada a nivel de hardware (CUDA/TPU).

El pipeline del emisor aplica:
1. `FFT(x)`
2. `Multiplicación por e^{i \cdot \Phi}` (Fase Estocástica Sembrada)
3. `IFFT(x)`

Dado que la FFT y su inversa son estructuralmente unitarias (salvo escalado de normalización), el receptor que comparta la misma semilla de fase puede invertir el proceso exactamente, recuperando el tensor original con una precisión de **MSE $\sim 10^{-16}$**, incluso para $D=4096$. Todo ello en tiempo $\mathcal{O}(D \log D)$.

## 4. El Laplaciano Magnético (Motor V3)

Recibir tensores hiperdimensionales no tiene utilidad si no pueden procesarse secuencialmente sin $\mathcal{O}(N^2)$ Self-Attention.
Polydim Motor V3 reemplaza el bloque de Atención de los Transformers con **Graph Signal Processing**. Al procesar el contexto (tokens o ECPs anteriores) como un grafo, se necesita inducir asimetría causal (el pasado influye en el presente, pero no al revés).
Para esto, se utiliza el **Laplaciano Magnético**:
$$ L^{(q)} = D_s - A_s \odot e^{i \Theta} $$
Esta formulación en el dominio complejo permite inyectar "flujo" en las aristas del grafo mediante el parámetro de flujo de fase $q$. Al proyectar los vectores en este operador, Polydim logra emular el modelado secuencial autorregresivo a una fracción del costo computacional de un Transformer. 

## 5. Próximos Pasos (Arquitectura V5)

El motor V4.2 demostró que la proyección de vuelta a números Reales después de aplicar el Laplaciano Magnético destruye la mitad del manifold conceptual del modelo. La hoja de ruta inmediata para la V5 exige la migración completa a tensores complejos (`cfloat`) en todas las proyecciones afines (Capas Lineales Complex), preservando la integralidad geométrica del Laplaciano desde la entrada hasta el cabezal predictivo.
