# SPEC: JAX (Motor Topológico Nativo TPU)

## Objetivo
Migrar los componentes matemáticos pesados del `MotorV2` de PyTorch a JAX para maximizar el paralelismo puro en TPUs.

## Tareas a Generar
1. **Traducción de Álgebra Geométrica (JAX/Flax)**:
   - Traducir `CartanDieudonneStiefelRotor` a JAX, utilizando `jax.numpy.linalg`.
   - Implementar la Identidad de Woodbury (`woodbury_invert`) optimizada con `jax.jit` para compilación estricta XLA (O(NK^2)).
2. **DeepONet en JAX**:
   - Codificar el `DeepONetOperator` (Branch/Trunk nets) en Flax. JAX es inmensamente superior para resolver operadores neuronales de dimensión infinita gracias a su manejo funcional puro de los gradientes (`jax.grad` y `jax.vmap`).
3. **Ecuación del Calor (Consenso de Dirichlet)**:
   - Implementar el Laplaciano de Haces y el descenso de gradiente de la Energía de Dirichlet usando `jax.lax.scan` para evitar bucles for en Python.

## Notas Técnicas
- JAX es puramente funcional. Se prohíbe el estado mutable interno. Las clases deben refactorizarse hacia tuplas de estado y funciones puras `(params, state, input) -> (new_state, output)`.
