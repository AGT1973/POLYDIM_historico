# SPEC: Rust (Gateway & Seguridad del Orquestador)

## Objetivo
Construir un gateway de altísimo rendimiento y una fortaleza de memoria para el nodo central del Orquestador (Aladino).

## Tareas a Generar
1. **Enrutador de Enjambre (Latent Router)**:
   - Servidor asíncrono (usando `tokio` y `axum`) capaz de manejar miles de conexiones concurrentes de los workers de Colab.
   - Recepción y encolamiento de tensores (vía buffers cero-copia) eludiendo por completo el GIL de Python.
2. **Motor de Verificación `GeoID_HIT` en Rust**:
   - Módulo estricto que valide la Regla R10 (Punto Fijo Global).
   - Escrito en Rust para asegurar matemáticamente (gracias al borrow checker) que la Esencia (el hipervector 0-esqueleto) sea inmutable a nivel hardware/memoria. Nadie puede corromper la identidad, ni por error de punteros ni por ataques.
3. **FFI (Foreign Function Interface)**:
   - Exportar los binarios de Rust mediante `PyO3` para que el `MotorV2` en Python pueda invocar las validaciones del 2-esqueleto a la velocidad de C/C++.

## Notas Técnicas
- Todo el manejo de tensores en memoria compartida debe ser `unsafe`-free, utilizando librerías nativas o puentes con `ndarray`.
