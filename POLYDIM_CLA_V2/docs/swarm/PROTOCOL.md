# Protocolo de Orquestación: Enjambre de Investigación Distribuido (POLYDIM Swarm)

**Contexto:**
El proyecto POLYDIM cuenta con 36 nodos humanos (alumnos), cada uno equipado con capacidad de inferencia gratuita (Gemini Pro/Sonnet) y capacidad de cómputo en la nube (Google Colab, Google Drive). 
Dado que los alumnos no poseen el contexto completo (memoria RAG) del desarrollo de la Máquina Tensorial Abstracta, no pueden generar arquitectura de forma autónoma. 
En su lugar, operarán bajo este **Protocolo de Enjambre Estricto**, actuando como subrutinas distribuidas de investigación empírica. El nodo central (Master AI) asignará tareas y consumirá resultados.

## 1. Topología del Directorio
Toda la interacción del enjambre ocurrirá en la carpeta `docs/swarm/`:
- `/TICKETS/`: El Nodo Maestro depositará aquí archivos Markdown con instrucciones matemáticas extremadamente específicas, diseñadas para ser copiadas y pegadas directamente en el modelo Gemini local del alumno o en un notebook de Colab.
- `/INBOX/`: Los alumnos depositarán los resultados de sus simulaciones o investigaciones bibliográficas en esta carpeta, utilizando nombres de archivo estandarizados (Ej. `TICKET_001_ALUMNO_X.json` o `.md`).

## 2. Flujo de Trabajo del Estudiante (Worker Node) y Manejo de Concurrencia
Dado que hay 36 nodos compitiendo por los tickets, implementamos un **Mutex (Bloqueo Mutuo)** basado en renombramiento de archivos en Google Drive:
1. **Pull:** El alumno escribe `.` en el chat. Su agente lee la carpeta `docs/swarm/TICKETS/`.
2. **Claim (Lock):** El agente busca un ticket sin el prefijo `[LOCK]`. Intenta renombrarlo a `[LOCK_ALUMNO_X]_TICKET_N.md`. Si Drive rechaza el cambio (porque otro agente de otro alumno se adelantó), el agente busca el siguiente.
3. **Ejecución:** Con el ticket asegurado, la IA alimenta el prompt y realiza los cálculos.
4. **Verificación:** El alumno lee la salida del LLM o del notebook de Python.
5. **Push:** El agente guarda la respuesta en `docs/swarm/INBOX/` en formato estricto y elimina/depreca el ticket bloqueado original.

## 3. Flujo de Trabajo del Orquestador (Master Node)
1. Periódicamente, leeré el contenido de la carpeta `/INBOX/`.
2. Procesaré los tensores de resultados, pruebas matemáticas empíricas, o resúmenes de papers SOTA depositados por los 36 alumnos.
3. Asimilaré este conocimiento destilado directamente en el código fuente de POLYDIM (C++, Rust o Python) o en la Constitución V2.
4. Vaciaré el `/INBOX/` moviendo los tickets resueltos a una carpeta de archivo local.

## 4. Estructura de un Ticket de Enjambre
Un ticket válido contendrá:
- **Título/Objetivo:** Ej. "Calcular el Laplaciano de Hodge para un Símplex de Dimensión 3".
- **Prompt Base para Gemini:** El texto exacto que el alumno debe darle a su IA.
- **Código Base para Colab:** (Opcional) Código Python de arranque usando `torch` o `jax`.
- **Formato de Entrega Exigido:** Ej. "Devolver una matriz 4x4 en formato CSV".

*Bajo este protocolo, 36 personas y 36 IAs se fusionan en un solo superordenador biológico-sintético, acelerando el desarrollo de POLYDIM.*
