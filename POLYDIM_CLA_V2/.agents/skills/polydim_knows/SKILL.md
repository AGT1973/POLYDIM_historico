---
name: polydim_knows
description: Bases ontológicas y Constitución V2.0 de la Programación Cognitiva. Raygails formales para agentes de IA.
---

# SKILL: Constitutional Guardrails (Raygails v2.0)

Este skill define los principios inmutables y leyes de diseño establecidos por la **Constitución de la Programación Cognitiva (V2.0)**. 

---

## 1. EL POSTULADO FUNDAMENTAL
*   **Erradicación del *Impedance Mismatch*:** Se prohíbe obligar a las IAs a serializar sus estados internos densos en texto para razonar en sistemas multi-agente.
*   **El Proceso Cognitivo:** El comportamiento útil del sistema debe emerger de la evolución de representaciones internas dirigidas por objetivos formales, no de instrucciones secuenciales (CPU).

## 2. GUARDARRAÍLES MATEMÁTICOS Y GEOMÉTRICOS

### 2.1 La Tupla de Programa
Todo diseño propuesto debe mapear explícitamente al sistema $P = (S, G, T, O, C, \Pi)$. Debes poder responder: ¿Cuál es el Estado? ¿Cuál es el Objetivo? ¿Cuál es la Restricción?

### 2.2 Dinámicas Algebraicas de POLYDIM
Si estás implementando POLYDIM (el caso empírico de la teoría), aplican sus 4 primitivas inmutables:
1.  **COMPOSE($T_1, T_2$):** Composición funcional monoidal.
2.  **MIX($\alpha, T_1, \beta, T_2$):** Superposición continua en dimensiones $N \ge 10,000$.
3.  **FIXPOINT($T, \epsilon$):** Reemplaza bucles condicionales por atractor de Banach.
4.  **RECUR($A,B,C,h,x$):** Dinámica lineal de espacios de estados para infinitos tensores.

### 2.3 Identidad y Proyección
*   **GEO_ID Invariante:** El identificador del objeto ($g$) jamás se modifica bajo transformaciones $\text{dist}(T(g), g) < \epsilon$.
*   **Proyección Funtorial (`PROJECT`):** Los "Tipos de Datos" clásicos (SQL, Rust, Flutter) emergen únicamente al proyectar ortogonalmente el tensor en el espacio del observador. Nunca se declaran *a priori*.

---

*SKILL: polydim_knows · Basado en Constitución V2.0 · 2026-07-01*
