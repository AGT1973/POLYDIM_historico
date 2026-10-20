# SPEC: Colab / Python (Latent Swarm Worker)

## Objetivo
Desplegar el `LatentSwarmWorker` en las instancias gratuitas de Google Colab aprovechando la aceleración por hardware (TPU/GPU).

## Tareas a Generar
1. **Notebook de Inicialización (`worker_node.ipynb`)**:
   - Script que monte Google Drive para acceder al repositorio `MotorV2`.
   - Instalación de dependencias (PyTorch, PyTorch-XLA para soporte TPU).
2. **Conexión al Enjambre**:
   - Lógica para conectarse al Orquestador (polling a un inbox de Drive o WebSocket).
   - Descarga de un "Ticket Semántico" pendiente.
3. **Ejecución Teleológica**:
   - Instanciar `LatentSwarmWorker`.
   - Procesar la activación del ticket usando la política de ejecución del `MotorV2` (TeleologicalEvolution).
4. **Retorno ALIGN**:
   - Aplicar Procrustes Ortogonal para mapear el resultado final a la geometría de destino.
   - Subir el tensor resultante de vuelta al sistema.

## Notas Técnicas
- Colab expira tras periodos de inactividad. El notebook debe ser "stateless" (sin estado persistente local), guardando todo progreso topológico directamente en la nube.
