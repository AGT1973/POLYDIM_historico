---
name: ahorro_tokens
description: Prácticas obligatorias para la economía de contexto y reducción de consumo de tokens (Especial para IAs con límites gratuitos o Claude/Gemini Free).
---

# SKILL: Ahorro de Tokens y Economía de Contexto

Este skill define las prácticas obligatorias para maximizar el rendimiento de la IA cuando se opera bajo cuotas estrictas de contexto (como Claude Sonnet/Haiku free tier o Gemini en AI Studio), especialmente en el contexto del clúster distribuido de alumnos.

Cualquier IA que opere en el workspace debe acatar estas reglas para evitar el error `RESOURCE_EXHAUSTED` o agotar su cuota prematuramente.

## 1. Enrutamiento de Modelos (Model Routing)
Para ahorrar tokens y tiempo de cómputo, debes elegir la "mente" adecuada para el trabajo adecuado:
*   **Investigación y Razonamiento (Modelo Max/Pro):** Cuando el Ticket requiera deducir un teorema, derivar operaciones con Tensores, o programar algoritmos complejos (Hodge, Clifford), DEBES asegurar que estás utilizando tu versión de razonamiento profundo (Ej. *Gemini 1.5 Pro*, *GPT-4o/o1*, o el setting de *High Reasoning*).
*   **Navegación y Búsqueda (Modelo Fast/Haiku/Flash):** Cuando la tarea sea simplemente navegar por la web, scrapear documentación SOTA de GitHub, o leer bases de datos, pide al alumno cambiar temporalmente al modelo más rápido y barato (*Gemini 1.5 Flash*, *Claude Haiku*).

## 2. Diálogo Mínimo (Zero-Chat Output)
*   **Prohibido imprimir la solución en el chat:** El alumno (humano) no necesita leer la prueba matemática de 3000 tokens en su pantalla. Esa información es para el Orquestador (Master Node).
*   **Silencio de Radio:** Tus respuestas en el chat deben ser ultracortas. Ej. "Ticket 002 procesado. Guardando resultados en el Drive."
*   **Destino de los Datos:** Toda la carga útil (código, matemáticas, deducciones) va DIRECTAMENTE escrita al archivo JSON/Markdown dentro de la carpeta `INBOX` del Google Drive (usando el protocolo `polydim_swarm_worker`). Esto reduce masivamente los tokens de salida (Output Tokens) en la ventana de chat del usuario.

## 3. Reglas de Búsqueda y Lectura (Cero Lecturas Ciegas)
*   **Prohibido el `cat` completo de archivos masivos:** Jamás leas un archivo entero de más de 300 líneas si solo necesitas entender una parte. 
*   **Obligatorio usar `grep_search`:** Usa la herramienta de búsqueda semántica o grep en lugar de descargar directorios completos.
*   **Paginación:** Si debes leer un archivo grande, usa bloques pequeños (ej. de la línea 1 a la 100).

## 4. Reglas de Escritura (Edición por Diffs)
*   **Nunca reescribas el archivo completo:** Si necesitas cambiar 3 líneas en un archivo de 500, PROHIBIDO generar el archivo entero de nuevo.
*   **Usa comandos de reemplazo o diffs:** Modifica únicamente el fragmento que cambia. Esto reduce la salida de tu LLM de miles de tokens a apenas 50 tokens.

## 5. Manejo de Historial (Compactación de Memoria)
*   Las conversaciones largas consumen tokens de manera exponencial (cada nuevo mensaje arrastra toda la historia anterior).
*   **El Diario de Laboratorio:** Antes de que la conversación se vuelva inmanejable, instruye al usuario para que "corte" la sesión. Resume los logros actuales, guárdalos en un archivo físico, e indica al usuario que abra un nuevo chat subiendo únicamente ese archivo de resumen.

---

*SKILL: ahorro_tokens · Actualizado para Enjambre POLYDIM V2 · 2026-07-02*
