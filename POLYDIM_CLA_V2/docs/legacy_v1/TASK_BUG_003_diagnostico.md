# TASK_BUG_003 — Diagnostico real (revisa el framing original de la auditoria)

**Fecha:** 2026-06-19
**Metodo:** Tests reales ejecutados contra polydim_runtime_v03.py V0.3.1,
no solo lectura de codigo.

## Lo que decia AUDITORIA_2026-06-17.md

"align_transform es de bajo rango (proyecta sobre solo 19 sondas de 10000
dims). Score de 0.92 es optimista."

## Lo que los tests reales muestran

1. CONFIRMADO: A y B (matrices de sondas) tienen shape (28, 10000) — rango
   maximo 28. Esto es matematicamente correcto.

2. REFUTADO (parcialmente): el impacto practico NO es "el score es
   optimista en general". Depende del tipo de contenido:
   - Comparando dimensiones NATIVAS puras (sin props): error de fidelidad
     ~0.6-1.5%. Bajo, no es un problema practico real.
   - Comparando el MISMO objeto (mismas props exactas) si el vocabulario
     de esas props esta incluido en las sondas: similitud 0.978 (alto,
     casi perfecto).
   - Comparando objetos con vocabulario de props NO incluido en las
     sondas: la transformacion no generaliza en absoluto (sim ~0.50,
     puro azar) — esto SI es un problema real.

3. Aumentar el numero de sondas (probado con 100, 500, 1000, 2000
   sondas aleatorias) NO resuelve el problema de generalizacion. Mas
   sondas dan mas rango, pero el rango no es la causa real del fallo de
   generalizacion a vocabulario nuevo.

4. CAUSA RAIZ REAL: dos Space independientes generan vectores NO
   relacionados para el mismo string (sp_a.sym("x") y sp_b.sym("x") tienen
   sim()~0.50, azar puro, por diseno: cada Space tiene su propio seed).
   align_transform aprende una transformacion lineal ancada SOLO en los
   pares de sondas explicitas. No hay manera de que generalice a un string
   que ninguna de las dos partes declaro como sonda — no es un problema de
   "poca cobertura dimensional", es la ausencia de cualquier ancla para
   ese contenido especifico.

## Fix correcto (no es "mas sondas aleatorias")

El handshake debe negociar explicitamente que VOCABULARIO de props se va
a usar en la sesion, e incluirlo como sondas junto a NATIVE+SONDAS antes
de calcular A/B. Esto es responsabilidad del protocolo (InitMsg/AcceptMsg),
no de align_transform en si — align_transform ya funciona correctamente
dado el vocabulario adecuado (0.978 de fidelidad confirmado).

Cambio propuesto para implementacion futura:
- InitMsg/AcceptMsg ganan un campo opcional `vocab_extra: List[str]` con
  los valores de props que la sesion va a transmitir.
- align() incluye ese vocab_extra en las sondas ademas de SONDAS fijas.
- Si una sesion transmite contenido con vocabulario no anticipado en el
  handshake, debe re-negociar (re-alinear) antes de enviarlo, o aceptar
  degradacion a MODO_S para ese paquete especifico.

## Por que no se implementa el fix completo en esta sesion

Cambiar InitMsg/AcceptMsg y el flujo de handshake es un cambio de protocolo
mas amplio que un parche puntual (afecta TASK_023 mesh multi-agente, que
esta DISPONIBLE y sin empezar — mejor coordinar ambos cambios juntos en vez
de tocar el handshake dos veces). Se deja el diagnostico completo y
verificado para que quien tome TASK_023 (mesh) o una nueva iteracion de
TASK_BUG_003 lo implemente con el contexto completo.

## Conclusion para el backlog

TASK_BUG_003 pasa de DISPONIBLE a DIAGNOSTICADO: el problema es real pero
distinto de como se describio originalmente. Fix concreto identificado y
documentado, pendiente de implementacion coordinada con el handshake.
