# Estrategia de Transición: De POLYDIM a la Programación Cognitiva (v1.2)

## 1. ¿Qué hacemos con el Paper publicado?

Dado que el paper ya es público (Zenodo/OSF), la norma científica dicta **no borrarlo ni parchearlo con anexos**. Debes hacer lo que hacen los grandes laboratorios: **Publicar la Versión 2 (v2)**.

- **Acción:** En Zenodo o arXiv, existe la opción "Upload new version".
- **Contenido:** Subiremos el `POLYDIM.md` refactorizado que acabamos de crear, donde el *Abstract* y la *Introducción* declaran explícitamente que POLYDIM no es "la teoría del todo", sino el **primer caso de estudio formal** de un programa de investigación mucho mayor llamado Programación Cognitiva.
- **Ventaja:** Esto demuestra rigor. Muestra que la investigación evolucionó tras la primera publicación.

## 2. ¿Qué hacemos con la Constitución?

**Debe rehacerse por completo.**
La constitución actual estaba diseñada para gobernar el desarrollo de un *lenguaje de programación*. La nueva constitución debe gobernar el desarrollo de una *rama científica*.

Debe incluir:
- Los 10 Principios de la Programación Cognitiva (Fase XXVI).
- La tupla del programa cognitivo $P = (S, G, T, O, C, \Pi)$.
- Las directrices estrictas para los agentes: "Su objetivo no es optimizar código clásico, sino explorar la composición, preservación y verificación semántica".

*(Podemos reescribirla en nuestro próximo paso iterativo).*

## 3. ¿Cómo organizamos a los alumnos y sus IAs?

El "verdadero trabajo científico" ocurre ahora. No podemos dejar que las IAs de los alumnos sobreescriban el canon teórico (el `THEORY.md` o el `POLYDIM.md` originales).

**El Modelo de Aislamiento (Sandboxing):**
1. Existirá una carpeta `/Alumnos/` en la raíz.
2. Cada alumno creará su propia carpeta: `/Alumnos/JuanPerez/`.
3. Le daremos a los alumnos un **System Prompt** específico que pegarán en su Cherry Studio / Cline.
4. Ese prompt instruirá a la IA del alumno a:
   - Leer (SOLO LECTURA) la teoría fundacional en `/docs/`.
   - Escribir (LECTURA Y ESCRITURA) sus experimentos o demostraciones matemáticas *única y exclusivamente* dentro de `/Alumnos/JuanPerez/`.
