RESEARCH: K_EFECTIVO para ALIGN — Tesis de compatibilidad
Fecha: 2026-06-12 — TASK_013 — ai.mpat.agt@gmail.com

═══════════════════════════════════════════════════════════════

PREGUNTA INICIAL

PEND_NEW_001: con K=28 sondas y N=10000, ALIGN entre espacios heterogeneos
produce score ≈ 0.50. ¿Cuántas sondas K se necesitan para superar 0.85?

═══════════════════════════════════════════════════════════════

EXPERIMENTOS

Exp 1: score vs K — espacios completamente aleatorios

  K     score_emp   formula_0.5+0.5K/N
  10    0.5007      0.5005
  28    0.5002      0.5014
  50    0.5017      0.5025
  100   0.4999      0.5050
  200   0.5004      0.5100
  400   0.5005      0.5200

El score empirico NUNCA supera 0.50 sin importar K.
La formula teorica sobrestima. Conclusion: K es irrelevante para
espacios sin transformacion lineal real.

Exp 2: score vs grado de perturbacion — K=28 fijo

  Caso                   eps    score
  Mismo espacio          0.0    1.0000
  Leve perturbacion      0.01   0.8536
  Media perturbacion     0.5    0.5110
  Espacio aleatorio      inf    ~0.500

═══════════════════════════════════════════════════════════════

ANALISIS MATEMATICO

Por que la formula teorica falla:

La formula 0.5 + 0.5*(K/N) asume que el SubspaceAligner generaliza
a vectores fuera del subespacio de entrenamiento. Esto solo es valido
si existe una transformacion lineal real entre los espacios.

Para espacios aleatorios (sin relacion):
  1. El vector holdout h_a y h_b son independientes.
  2. La componente de h_a en span(basis_a) es O(K/N) = 0.0028 para K=28.
  3. El complemento (99.72% del vector) pasa intacto como h_a.
  4. sim(h_a, h_b) ≈ 0.5 para vectores aleatorios en R^N.

Resultado: score ≈ 0.5 para cualquier K cuando no hay transformacion real.

La pregunta CORRECTA no es "cuantas sondas K se necesitan".
La pregunta CORRECTA es: ¿existe una transformacion lineal entre los espacios?

  NO existe → ALIGN nunca supera 0.5. K irrelevante.
  SI existe → ALIGN converge. K=28 puede ser suficiente.

═══════════════════════════════════════════════════════════════

TESIS CENTRAL: ALIGN COMO DETECTOR DE COMPATIBILIDAD

ALIGN no alinea. ALIGN DETECTA si los espacios son compatibles.

Escala de compatibilidad:
  score = 1.00         espacios identicos (misma implementacion)
  score en [0.85, 1.0) espacios muy similares (misma arquitectura)
  score en [0.5, 0.85) espacios parcialmente compatibles
  score ≈ 0.50         espacios incompatibles, sin relacion lineal

UMBRAL_ALIGN = 0.85 separacion correcta.
  eps=0.01 → score=0.8536 → por encima del umbral — MODO_H
  eps=0.50 → score=0.5110 → por debajo — MODO_S
El umbral esta bien calibrado.

═══════════════════════════════════════════════════════════════

IMPLICACIONES DE DISEÑO

POLYDIM V0.1 — deterministico:
  PolyDimSpace._make(name) usa MD5(name) como seed.
  Todas las instancias producen vectores IDENTICOS para los mismos nombres.
  ALIGN siempre da 1.0. MODO_H siempre disponible.

POLYDIM futuro — embeddings aprendidos:
  IAs con datasets/arquitecturas distintas tendran vectores diferentes.
  ALIGN detectara si la similitud es >= 0.85 para autorizar MODO_H.
  K_RECOMENDADO para ese caso: max(28, sqrt(N)) = 100 para N=10000.

Implicacion arquitectonica profunda:
  Capa S = lenguaje universal — siempre funciona, cualquier arquitectura
  Capa G = canal de alta velocidad — solo cuando espacios son compatibles
  La division S/G no es un compromiso. Es el diseño correcto.

═══════════════════════════════════════════════════════════════

CONCLUSIONES

1. PEND_NEW_001 RESUELTA: no existe K que permita ALIGN entre espacios
   completamente heterogeneos. Score se mantiene en ~0.50 sin importar K.

2. ALIGN mide compatibilidad. No la corrige.

3. POLYDIM V0.1 deterministico: ALIGN da 1.0 siempre entre instancias iguales.

4. Para IAs con embeddings aprendidos: ALIGN actua como guardian de calidad.

5. MODO_S es el modo universal robusto. Garantia de interoperabilidad total.

6. La arquitectura S+G es mas profunda de lo que parecia: no es una capa
   de compatibilidad hacia atras, es el diseño optimo para un mundo con
   IAs heterogeneas.

ACCIONES:
  TASK_012: actualizar SPEC_ALIGN con estos hallazgos
  PEND_NEW_001: CERRADA — la pregunta era incorrecta
  PEND_NUEVA_002: ABIERTA — calibracion iterativa para IAs con embeddings aprendidos
  UMBRAL_ALIGN = 0.85 confirmado correcto para V0.1

═══════════════════════════════════════════════════════════════
RESEARCH_ALIGN_K_EFECTIVO.md — 2026-06-12 — TASK_013 TERMINADA
