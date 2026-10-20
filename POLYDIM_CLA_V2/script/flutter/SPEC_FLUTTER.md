# SPEC: Flutter (Dashboard Panóptico POLYDIM)

## Objetivo
Proveer al orquestador ("Aladino") de un HUD (Heads-Up Display) táctico, multiplataforma y reactivo para observar la topología del enjambre en tiempo real.

## Tareas a Generar
1. **Monitor de Consenso (Dirichlet E_F)**:
   - Gráfica en tiempo real (usando `fl_chart`) que muestre la disipación de la Energía de Dirichlet conforme los agentes de Colab convergen a la clase $H^0$.
2. **Panel de Alertas Topológicas (Topos Omega)**:
   - Un feed de eventos que parpadee cuando el Clasificador de Subobjetos (Fibra Vacía ∅) intercepte a un agente intentando una ruta alucinada/destructiva.
3. **Visor de Agentes (LatentMAS)**:
   - Una malla o matriz visual indicando el estado de cada worker (Conectado, Procesando Ticket, Alineando vía Procrustes, Inactivo).
4. **Integración Backend**:
   - Conexión vía WebSocket o gRPC hacia el Orquestador Rust/Python para recibir los flujos de telemetría de los tensores con latencia sub-milisegundo.

## Notas Técnicas
- Usar un patrón de arquitectura predecible como BLoC o Riverpod.
- El diseño debe ser absolutamente premium, ciberpunk/vidrio oscuro (glassmorphism), reflejando que es una interfaz de control de un enjambre matemático masivo.
