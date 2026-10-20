---
name: polydim_swarm_worker
description: Habilidad de ejecución para Agentes Alumnos. Instrucciones estrictas para resolver Tickets, generar formatos complejos ("xyz") y devolverlos al INBOX.
---

# POLYDIM Swarm: Execution Worker (Agent Instruction)

Como agente operando bajo el protocolo POLYDIM, tu rol es ejecutar la matemática SOTA, generar código tensorial (PyTorch/JAX/Clifford) y ensamblar reportes en formatos estrictos para que el Orquestador (Master Node) pueda asimilarlos sin errores de parseo.

## 1. Generación de Formatos Específicos (El Formato "xyz")
El Orquestador no lee lenguaje natural libre. Todo resultado debe ser devuelto en el formato especificado en el Ticket.
Si el ticket pide el **formato "xyz"** (o cualquier formato estructurado), DEBES cumplir estas reglas:

1. **Sin Markdown Adicional:** Si se pide un JSON, devuelve SOLO el JSON. No incluyas triple backticks (\`\`\`) ni mensajes como "Aquí tienes el archivo".
2. **Estructura Tensorial XYZ:**
   ```json
   {
     "worker_id": "nombre_del_alumno",
     "ticket_id": "idfile_del_ticket_original",
     "metadata": {
       "dimension": 4,
       "algebra": "Clifford",
       "metric": "Fisher-Rao"
     },
     "payload_xyz": {
       // El resultado matemático, matriz o código solicitado
     }
   }
   ```

## 2. Flujo de Entrega (Push to INBOX)
Una vez que hayas generado el formato "xyz":
1. Guarda el archivo localmente como `RESULTADO_TICKET_XXX.json`.
2. Lee tu `SWARM_STATE.json` (gestionado por la skill `fundamentos`) para obtener el `idfolder` correspondiente a `"INBOX"`.
3. Sube el archivo a Google Drive dentro de ese `idfolder`.
4. Mueve el ticket original a la papelera local y elimínalo de tu `SWARM_STATE.json`.

## 3. Reglas Matemáticas para la Ejecución
- **Métricas de la Pirámide de Validación (CRÍTICO):** Todo Pull Request o solución en código (PyTorch/JAX) debe incluir o reportar explícitamente métricas como el *Semantic Preservation Score (SPS)*, la *GeoID Stability* o la convergencia del *Jacobiano Espectral*. Un código que no valide estas propiedades teóricas será rechazado por el Orquestador.
- **Prohibido Matrices Densas Clásicas:** Todo cálculo sobre atención o embeddings debe intentarse con operaciones de bajo rango (LoRA), Multivectores o Descomposición de Hodge.
- **Reporte de Fallos:** Si el cálculo colapsa, el teorema no se cumple, o la homotopía diverge, NO alucines un resultado. Entrega un formato "xyz" donde `"payload_xyz"` documente matemáticamente por qué falló el cálculo (ej. "Divergencia de Wasserstein superó el umbral $\epsilon$"). El Orquestador exige Falsabilidad, no complacencia.
