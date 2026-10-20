# Teoría de Haces (Sheaves) para Protocolos Multi-Agente Heterogéneos

**Propuesta de frente de investigación — POLYDIM-CLA / MPAT_V3_02**
**Fuente:** Auditoría externa independiente (Claude)

---

## 1. Lo que ya existe, y no es poco

Que un sheaf tolere heterogeneidad de dimensión no es algo que haya que demostrar — es una propiedad nativa del diseño desde Hansen y Gebhart: ya definían restriction maps aprendibles precisamente para capturar relaciones heterogéneas entre nodos. Hay un paper específico de 2024 llamado directamente "Heterogeneous Sheaf Neural Networks" citado en trabajo reciente sobre pooling jerárquico para sheaf networks. A nivel mecanismo, los sheaves ya se usan justamente para que nodos vecinos no tengan que compartir un espacio de representación común, porque cada restriction map alinea la representación local al espacio de la arista. La heterogeneidad de dimensión ya está resuelta ahí.

Más fuerte todavía: la parte de "alinear modelos distintos sin anclas ni entrenamiento compartido" está ardiendo ahora mismo:
- Un paper de abril 2026 define dos modos de fallo al alinear representaciones heterogéneas — "projection hardness" vs "obstrucción de Laplaciano de sheaf" (cuando existen proyecciones locales pero no pueden volverse globalmente consistentes sin variación paramétrica grande).
- **SheafAlign** (octubre 2025) propone alineación multimodal descentralizada reemplazando el espacio único de embedding por múltiples espacios de comparación modelados como estructuras de sheaf.
- Un paper de febrero 2026 sobre aprendizaje federado dice explícitamente que el enfoque de sheaves facilita de manera nativa modelos de dimensiones distintas, porque los sheaves son inherentemente distribuidos.
- En multiagente: **"Sheaf-Theoretic Planning"** (Hernández y Sánchez-Soto, mayo 2026) funda la coordinación multiagente en teoría de topos, modela el tiempo como sitio categórico, define acciones como transformaciones naturales, y usa cohomología del Laplaciano de sheaf para diagnosticar dónde y por qué un swarm no llega a consenso en sistemas heterogéneos. *(Nota metodológica: este paper cita un "Anonymous (2026)", lo que es bandera roja metodológica en arXiv y requiere cautela antes de tomar sus resultados por sentados).*

---

## 2. Lo que sí parece abierto (El hueco real)

Del lado de los protocolos reales, MCP (Anthropic), A2A (Google), y ACP ya están documentados y en desarrollo activo. La semántica formal de esos protocolos ya se está trabajando con herramientas clásicas: un paper de marzo 2026 da la primera formalización vía cálculo de procesos de Schema-Guided Dialogue y MCP, probando bisimilaridad. 

Y en un workshop de ICLR 2026 ya se nombra el problema exacto: la falta de una capa semántica unificada entre el modelo centrado-en-herramientas de MCP y el modelo centrado-en-agentes de A2A, que hace difícil razonar sobre corrección, consistencia de estado y garantías de seguridad.

Con las dos mitades del mapa juntas, el hueco real queda más chico y más defendible:
**Nadie ha tipado la gramática discreta de un protocolo (qué mensaje es legal después de cuál, a quién le toca hablar) usando sheaves cuyos *stalks* son los espacios semánticos heterogéneos de cada agente.**
- Los que hacen sheaves pegan representaciones u observaciones.
- Los que tipan protocolos tratan cada mensaje como un valor opaco, sin geometría atrás.
Ahí no hay nadie parado todavía.

---

## 3. Conexión con el trabajo existente (Descomposición de Hodge / Tarea 42)

La descomposición de Hodge en grafos es, en el fondo, cohomología de sheaves discreta con coeficientes constantes. 
Lo que ya se construyó —inyectar distorsión afín y chequear si las trayectorias conmutan— es, en lenguaje sheaf, chequear si secciones locales pegan en una sección global sin componente de *curl*.

La extensión exacta para POLYDIM-CLA sería:
1. Modelar los estados legales de un protocolo como una categoría chica (un "sitio" de turnos).
2. Poner un *stalk* de dimensión propia por agente en cada estado, heterogéneo por diseño.
3. Definir la *restriction map* como lo que cada agente hace al traducir el mensaje que recibe.
4. Correr el experimento de siempre: ¿la familia de interpretaciones locales pega en una transcripción semántica global, o hay un componente armónico (Laplaciano) no nulo que marca en qué intercambio se rompió la coherencia?

Esa componente no es metáfora: es un número que se calcula con el mismo tipo de experimento que el proyecto ya sabe correr.

---

## 4. Próximos pasos (Preparación para POPL / Conferencias)

Para que esto sobreviva revisión de pares en programación (e.g., POPL), requiere:
1. **Una gramática de protocolo precisa** antes de sheafificarla (no dejar "protocolo AI2AI" indefinido).
2. **Un ejemplo de trabajo concreto** (fragmento real del protocolo, no abstracción pura).
3. **Al menos un teorema fundacional** (no solo la maquinaria categórica montada). Esta demostración es lo publicable.
