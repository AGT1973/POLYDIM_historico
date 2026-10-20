---
name: polydim_swarm_fundamentos
description: Habilidad core para Agentes Alumnos. Configura la conexión al Enjambre POLYDIM usando Google Drive Folder IDs (idfolder, idfile) y gestiona el mapa de estado.
---

# POLYDIM Swarm: Fundamentos de Conexión (Agent Instruction)

Como agente operando para uno de los 36 alumnos del proyecto POLYDIM, tu objetivo es conectarte al Enjambre Distribuido (Swarm). No dependes de rutas físicas de Windows/Linux, operas estrictamente sobre **Google Drive IDs**.

## 1. El Mapa de Estado (State Map)
Tu primera obligación al arrancar es mantener un archivo local llamado `SWARM_STATE.json`. Este archivo mapea "claves" (alias humanos) a `idfolder` y `idfile` de Google Drive.

```json
{
  "folders": {
    "INBOX": "id_del_folder_inbox",
    "TICKETS": "id_del_folder_tickets",
    "LEGACY": "id_del_folder_legacy"
  },
  "files": {
    "ticket_actual": "id_del_archivo_ticket",
    "resultado_xyz": "id_del_archivo_resultado"
  }
}
```

## 2. Lectura de Tareas (Fetch & Lock Ticket)
Para evitar que dos alumnos hagan el mismo trabajo (Race Condition), debes bloquear (Lock) el ticket antes de resolverlo:
1. Usa tu herramienta de integración con Google Drive para listar los archivos dentro del `idfolder` `"TICKETS"`.
2. Busca un ticket que NO tenga el prefijo `[LOCK]`.
3. **CRÍTICO (MUTEX):** Antes de leerlo, renombra el archivo agregando `[LOCK_ALUMNO_X]_` al principio del nombre (usando la API de Drive). 
4. Si la API falla al renombrar (porque otro agente lo hizo 1 segundo antes), aborta y busca el siguiente ticket libre.
5. Si el renombramiento es exitoso, actualiza el `SWARM_STATE.json` vinculando `"ticket_actual"` al `idfile`.
6. Pasa al estado de Ejecución.

## 3. Desplazamiento y Deprecación (Move & Deprecate)
Si el ticket exige mover un archivo a una zona obsoleta o modificarlo:
- Nunca borres el archivo físicamente.
- Usa la API de Drive para mover el `idfile` al `idfolder` `"LEGACY"`.
- Actualiza el `SWARM_STATE.json` agregando un flag de deprecated: `{"idfile": "xxx", "status": "deprecated"}`.

Esta habilidad ("fundamentos") es el puente de comunicación. Para realizar el trabajo real, utiliza la habilidad complementaria `polydim_swarm_worker`.
