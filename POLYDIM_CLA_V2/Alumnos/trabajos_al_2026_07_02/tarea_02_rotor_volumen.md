# Tarea 2 — Escuadrón 1: Rotores y conservación de volumen

**Nota para el grupo antes de leer:** este material NO sigue el formato "worker/JSON silencioso" descrito en `swarm_context_200.txt`. Está escrito para que lo entiendas, no para que se acumule como "evidencia de validación" de POLYDIM. La propiedad matemática que probamos abajo es real y verificable; las afirmaciones grandilocuentes del documento madre (inmunidad a alucinaciones, razonamiento topológico, etc.) no se derivan de este resultado — son un salto de conclusión que conviene que cada alumno detecte por su cuenta.

## Enunciado (tal como aparece en el banco de tareas)
"Actúa como tester de robustez de IA. Escribe código que simule la superposición garantizando conservación del volumen determinante."

## La idea, con un ejemplo cotidiano
Un cubo de madera de 1 litro que **rotás** en el aire sigue midiendo 1 litro: cambia su orientación, no su tamaño. Si en cambio lo **aplastás** con una prensa, su volumen sí cambia. Esa es la diferencia entre una rotación pura (lo que el documento llama "rotor") y una transformación lineal arbitraria (una matriz cualquiera sin restricciones).

## La trampa educativa (para pensar)
El documento presenta "usar rotores de Clifford en vez de matrices densas" como si fuera una innovación propia de POLYDIM que le da propiedades especiales de IA. En realidad, que las rotaciones preserven volumen es álgebra lineal clásica (transformaciones ortogonales, determinante 1), válida en cualquier dimensión, conocida desde el siglo XIX. Verificarlo con código confirma la geometría — no confirma ninguna afirmación sobre "razonamiento" o "ausencia de alucinaciones".

## Segundo ejemplo (2D, para afianzar el concepto a mano)
Una rotación de ángulo θ en el plano es:

R = [[cos θ, −sin θ], [sin θ, cos θ]]

det(R) = cos²θ + sin²θ = 1, para cualquier θ. Con θ = 90°: R = [[0,−1],[1,0]], det = 0·0 − (−1)·1 = 1.

## Código (Python) — ver también `tarea_02_rotor_volumen.py` en esta misma carpeta
Se implementan rotores 3D como cuaterniones unitarios (matemáticamente equivalentes a los rotores de Clifford en 3D), se verifica que:
1. det(R) = 1 (no hay compresión/expansión de volumen)
2. RᵀR = I (transformación ortogonal)

y se contrasta con una matriz aleatoria cualquiera, cuyo determinante no da 1.

## Resultado obtenido al ejecutar el código
```
=== Rotores aleatorios: el volumen SIEMPRE se conserva ===
  prueba 0: det(R) = 1.000000 | error ortogonalidad = 2.76e-16
  prueba 1: det(R) = 1.000000 | error ortogonalidad = 3.31e-16
  prueba 2: det(R) = 1.000000 | error ortogonalidad = 1.80e-16
  prueba 3: det(R) = 1.000000 | error ortogonalidad = 2.55e-16
  prueba 4: det(R) = 1.000000 | error ortogonalidad = 6.70e-16

=== Contraejemplo: una mezcla NO-rotor ===
  det(M) = 1.020480  (no es 1, distorsiona volumen)
```

Los errores del orden de 1e-16 son ruido de punto flotante, no error del método.

## Para que cada alumno verifique por su cuenta
1. Correr `tarea_02_rotor_volumen.py` con otra semilla aleatoria y confirmar que det(R) sigue dando 1.
2. Probar a mano el caso 2D con θ = 45° y θ = 180°.
3. Preguntarse: ¿qué otra propiedad geométrica (además del volumen) preserva una rotación, que una matriz aleatoria no preserva? (pista: ángulos entre vectores).
