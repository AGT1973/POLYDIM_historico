# Plan de Implementación: Refactorización y Endurecimiento de POLYDIM

Basado en la evaluación forense (Red Teaming), este plan aborda de manera sistemática las 6 vulnerabilidades críticas descubiertas en el código y la documentación. 

El objetivo es cerrar la brecha entre la ambición teórica y la realidad de la implementación, logrando que el proyecto sea académicamente inexpugnable.

## 1. Correcciones Matemáticas Críticas (Rust VM)

### [MODIFY] `core/polydim_core.rs`
- **Bug de Identidad:** La función `identity()` actualmente retorna la función cero. 
  - *Solución:* Modificar `identity(n)` para que retorne una transformación de rango completo (`rank = n`), donde $U$ y $V^T$ sean matrices identidad $N \times N$, logrando $T(v) = v$ matemáticamente correcto.
- **Bug de Subnormales FP16:** `f16_to_f32` decodifica números muy cercanos a cero con hasta 1025x de error.
  - *Solución:* Reemplazar el branch subnormal con un bucle de renormalización de mantisa (desplazamiento hasta encontrar el bit 1 implícito y ajuste del exponente).

## 2. Limpieza y Homogeneización de la Constitución

### [MODIFY] `POLYDIM_CONSTITUCION.md`
- **Duplicación del Art. 11:** Eliminar las líneas duplicadas del Volumen IX / Artículo 11.
- **Catálogo de Dimensiones Nativas:** Sincronizar las 9 dimensiones (`DIM_SQL`, `DIM_FLUTTER`, etc.) para que la Constitución coincida 100% con la constante `NATIVE_DIMS` del código Rust.
- **Artefactos de IA:** Rastrear y eliminar los 10 enlaces con `?utm_source=openai` y la sección de autoevaluación (9.5/10) generada por LLMs.

## 3. Re-estructuración del Paper y Citas

### [MODIFY] `POLYDIM.md`
- **Teoremas Fundacionales:** Mover T1 (Asociatividad), T2 (Linealidad), T4 (Banach) y T5 (GEO_ID) a una nueva sección **"Sección 2: Preliminares Matemáticos"**.
- **Bibliografía:** Añadir los papers faltantes que soportan el marco teórico (Sargsyan, Dudzik, Mašulović, Kamiura) y arreglar la cita de Gavranović (ICML 2024 vs ESOP 2022).
- **Aclaración Interlat:** Corregir la cita de Interlat para atribuir correctamente el dato de 40,000 bits a Zhu et al.

## 4. Alineación de Especificaciones Binarias

### [MODIFY] `SPEC_FORMATO_BINARIO_V0.md`
- **`algebra_tolerance_epsilon`:** Modificar la tabla de bytes del HEADER para incluir explícitamente el campo `algebra_tolerance_epsilon` (float32, 4 bytes) en lugar de usar bytes reservados, cumpliendo con la Regla R12 de la Constitución.

## 5. Implementación del Protocolo ALIGN

### [MODIFY] `core/polydim_core.rs`
- **Añadir ALIGN:** Implementar el esqueleto real del protocolo Procrustes Ortogonal (SVD) para alinear dos espacios latentes, cerrando el gap entre lo que anuncia el paper y lo que hace la VM.

## Open Questions para el Usuario
> [!IMPORTANT]
> **Sobre el módulo faltante (`polydim_fft_bind.rs`):** El código lo importa pero no existe en el repositorio entregado. ¿Querés que cree un esqueleto ("stub") de este archivo para que el compilador pase, o querés que implementemos la convolución circular (FFT) real para la mezcla ortogonal?

> [!TIP]
> **Sobre el protocolo ALIGN:** La implementación de Procrustes Ortogonal requiere SVD (Descomposición en Valores Singulares), lo que normalmente necesita una librería externa de álgebra lineal (`ndarray-linalg` o similar). Siendo que POLYDIM busca "dependencias cero", ¿preferís que implemente una SVD aproximada pura en Rust, o agregamos una dependencia matemática ligera a `Cargo.toml`?
