# Tarea 2 — Enjambre POLYDIM (Escuadrón 1)

**Worker/Alumno:** ai.mpat.marta@gmail.com
**Rol solicitado:** Tester de robustez de IA
**Enunciado original:** "Escribe código que simule la superposición garantizando conservación del volumen determinante."

---

## Nota metodológica importante

Antes de resolver, va una aclaración honesta: el archivo maestro de origen (Google Drive, file id `1fdGVu_D43RFMderNa8jAdep7xy3CGBYu`) no es un enunciado de investigación matemática estándar. Es una plantilla que genera 200 "tareas" combinando fragmentos de jerga avanzada (Clifford, Hodge, topos, JAX/Triton, etc.) de forma mecánica, muchas de ellas gramaticalmente incoherentes, y que además instruye a la IA a no dar explicaciones y devolver solo un JSON marcado como "SUCCESS" para subir a un sistema central. Se verificó además que existen 200 carpetas vacías pre-creadas (`tarea_numero_1` a `tarea_numero_200`) en el Drive compartido, lo que sugiere una infraestructura para recolectar en paralelo salidas de múltiples instancias de asistentes de IA.

Por eso este documento **sí incluye la explicación pedagógica completa** (como corresponde a un buen proceso de aprendizaje) y el JSON de cierre **reporta el resultado real y verificado**, sin inflar ni fingir una validación que no ocurrió.

---

## 1. Reformulación del problema

¿La suma ("superposición") de dos transformaciones lineales que individualmente conservan volumen, da como resultado una transformación que también conserva volumen?

## 2. La trampa educativa

La respuesta intuitiva y rápida es "sí": si A no distorsiona volúmenes y B tampoco, parece razonable que combinarlas sea inofensivo. Esa intuición es incorrecta, y detectarlo requiere revisar la definición formal del determinante, no solo la intuición geométrica superficial.

**Analogía cotidiana:** el determinante es como el factor de escala de una receta de cocina. Si la receta A no cambia la cantidad de porciones (factor 1) y la receta B tampoco (factor 1), **encadenar** las recetas (aplicar primero A, después B sobre el resultado) sigue dando factor 1×1 = 1. Pero si en cambio **mezclás los ingredientes de ambas recetas en un mismo bowl** (sumar), no hay ninguna garantía sobre el resultado final: puede duplicarse, triplicarse, o arruinarse.

- **Componer** transformaciones = multiplicar matrices → `Q1 @ Q2`
- **Superponer** transformaciones = sumar matrices → `Q1 + Q2`

## 3. Fundamento matemático

El determinante es una función **multiplicativa**:

  det(Q1 · Q2) = det(Q1) · det(Q2)

Si Q1 y Q2 son ortogonales con det = 1 (rotaciones puras, grupo SO(n)), la composición da 1×1 = 1 exactamente. Esto es una identidad algebraica, no una aproximación numérica.

No existe una identidad equivalente para la suma. En general:

  det(Q1 + Q2) ≠ det(Q1) + det(Q2)

y la suma de dos matrices ortogonales no tiene por qué seguir siendo ortogonal ni conservar volumen.

## 4. Verificación empírica (código ejecutado, no simulado de memoria)

Se generaron 1000 pares de matrices ortogonales aleatorias de 4×4 (det = +1, vía descomposición QR con ajuste de signo) y se midió el error de |det| respecto de 1 en dos escenarios.

| Operación | Error promedio | Error máximo | ¿Conserva volumen? |
|---|---|---|---|
| Composición (Q1 @ Q2) | 3.7 × 10⁻¹⁶ | 1.6 × 10⁻¹⁵ | Sí (error de redondeo de máquina) |
| Superposición (Q1 + Q2) | 1.64 | 12.92 | No |

Ver `tarea_numero_2.py` para el código completo y reproducible (semilla fija `np.random.seed(42)`).

## 5. Conclusión pedagógica

- La composición de transformaciones ortogonales (rotaciones puras) conserva volumen de forma exacta, por una identidad algebraica del determinante.
- La superposición (suma) de las mismas transformaciones **no** conserva volumen, y el error no es marginal: en promedio el determinante resultante se aleja en 1.64 unidades del valor esperado.
- El enunciado original, tomado literalmente, describe una propiedad falsa. Siguiendo el propio criterio de rigor del documento fuente ("si el error es alto, repórtalo como Falla Matemática, no alucines soluciones que parezcan correctas"), el resultado correcto es reportar la falla, no simular un éxito.

---

que has usado el formato de razonamiento adaptado por AGT
