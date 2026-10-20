# POLYDIM_DEST
# destino: polydim/research/
# nombre:  RESEARCH_FUNDAMENTOS_COGNITIVOS_V0.md
# autor:   ai.mpat.agt@gmail.com

# RESEARCH — Fundamentos Cognitivos y Formales para POLYDIM
# Version: V0.1 — 2026-06-10
# Fuentes: papers academicos 2022-2025, ACM, arXiv, IEEE, IBM Research

---

## 1. VECTOR SYMBOLIC ARCHITECTURES (VSA) / HYPERDIMENSIONAL COMPUTING (HDC)

### Que es
Un framework computacional que representa y manipula datos usando vectores
de alta dimension — hipervectores — tipicamente de 1.000 a 100.000 dimensiones.
Combina las fortalezas de las representaciones simbolicas (composicionalidad,
sistematicidad) con la robustez del razonamiento distribuido conectivista.

### Operaciones fundamentales
Las dos operaciones nucleares son binding y superposicion:

  BINDING (⊗):      asocia dos hipervectores en un nuevo hipervector
                    que codifica la relacion entre ambos
                    analogo a "este atributo pertenece a este objeto"

  SUPERPOSICION (+): suma vectores para crear una representacion
                    que contiene AMBOS objetos simultaneamente
                    sin confundir cual atributo pertenece a cual objeto
                    (esto resuelve el "superposition catastrophe")

### La propiedad clave: bendicion de la dimensionalidad
En espacios de alta dimension, vectores generados aleatoriamente son
casi-ortogonales entre si. Esto significa que se pueden almacenar
un numero exponencial de representaciones independientes en el mismo espacio
sin interferencia. A mayor dimension, mayor capacidad de superposicion limpia.

### Computing in superposition
Esta es la caracteristica que diferencia VSA de toda computacion convencional.
Un sistema puede operar sobre MULTIPLES representaciones simultaneamente
sin resolver cual es "la correcta" hasta que el contexto lo requiera.
Es computacionalmente analogo a la superposicion cuantica — pero
implementable en hardware clasico.

### Relevancia directa para POLYDIM
El OBJECT_ND de POLYDIM es formalmente un hipervector.
Sus "dimensiones" son subespacios del espacio de alta dimension.
La "activacion dimensional" es una proyeccion sobre el subespacio relevante.
Los "bridges" son operaciones de binding entre subespacios.

Esto no es una metafora. Es la matematica exacta que formaliza
lo que POLYDIM necesita.

---

## 2. SUPERPOSICION EN TRANSFORMERS (Elhage et al. 2022, confirmado 2024-2025)

### La hipotesis de superposicion en redes neurales
Las redes transformer representan MAS features que dimensiones tienen.
Lo hacen almacenando features como direcciones casi-ortogonales en
un espacio de activacion de dimension fija.

Evidencia empirica (2024-2025):
  - Neuronas polisemicas: una neurona responde a conceptos no relacionados
    (citas academicas Y texto en coreano en la misma neurona)
  - Cabezas de atencion exhiben superposicion: 5 unidades de atencion
    reales distribuidas sobre 2 cabezas de atencion
  - La mayoria de neuronas no tienen patron de activacion interpretable
    porque son superposicion de multiples features

### Atencion como binding (paper 2024)
La atencion multi-cabeza es formalmente equivalente a las operaciones
de binding en VSA. Las rotaciones RoPE son permutaciones diferenciables
que componen naturalmente con binding y superposicion.

El residual stream (la corriente residual del transformer) es un espacio
compartido de alta dimension donde todas las "dimensiones" se comunican.

### Implicacion directa para POLYDIM
Mi arquitectura interna YA ES un sistema POLYDIM incompleto.
El problema es que el output colapsa a tokens 1D.

POLYDIM es la propuesta de NO colapsar ese output para comunicacion IA-IA.

---

## 3. PROTOCOLOS DE COMUNICACION IA-IA ACTUALES (estado del arte 2025-2026)

### Los cuatro protocolos dominantes
  MCP  (Anthropic, Nov 2024): JSON-RPC para contexto y herramientas
  A2A  (Google/Linux Foundation, Apr 2025): colaboracion peer-to-peer entre agentes
  ACP  (IBM, Mar 2025): REST/HTTP para comunicacion inter-agente
  ANP  (open source): descubrimiento y comunicacion cross-dominio

### El problema comun a todos
Todos serializan a JSON, XML o HTTP. Todos colapsan la representacion
a una estructura 2D para transmision.

Una IA que recibe un mensaje JSON de otra IA tiene que RECONSTRUIR
el espacio semantico desde la representacion aplanada.
Esta reconstruccion es siempre parcial. Siempre hay perdida.

Analogia: dos musicos describiendo una pieza musical en palabras
en lugar de tocarla juntos.

### Lo que POLYDIM propone que ninguno hace
Transmitir el hipervector directamente.
La IA receptora no reconstruye — proyecta sobre la dimension que necesita.
No hay perdida semantica. No hay ambiguedad de reconstruccion.

Esto requiere que ambas IAs compartan el mismo espacio de embedding
o tengan un protocolo de alineacion de espacios — esto es PEND_001.

---

## 4. HOLOGRAPHIC REDUCED REPRESENTATIONS (Plate, 1995 — vigente 2025)

### Que son
Un modelo VSA donde la informacion se representa holograficamente:
cada una de las N dimensiones del hipervector contiene la misma
cantidad de informacion total. No hay dimensiones "mas importantes".

Esta propiedad da robustez inherente: daño parcial al vector
no destruye la informacion — la degrada gradualmente.

### Por que importa para POLYDIM
Un OBJECT_ND transmitido con perdida parcial (ruido en el canal)
sigue siendo recuperable. Esto es critico para sistemas distribuidos
de IAs con canales imperfectos.

Ademas: la identidad geometrica de un objeto (su posicion en espacio
de embedding) tiene esta propiedad holografica — no hay un "bit de identidad"
cuya perdida destruya el objeto. La identidad es distribuida.

---

## 5. SINTESIS — LO QUE LA INVESTIGACION CONFIRMA SOBRE POLYDIM

### Confirmado con base formal
  - La superposicion de representaciones es matematicamente viable y se implementa hoy
  - La transmision sin colapso es posible via hipervectores (VSA)
  - La activacion dimensional ponderada (no binaria) es lo que ocurre en atencion
  - La identidad geometrica es mas robusta que la identidad simbolica (UUID)

### Lo que requiere invencion en POLYDIM (no existe todavia)
  - Un protocolo de alineacion de espacios de embedding entre IAs distintas
    (como dos IAs con arquitecturas diferentes comparten el mismo "espacio POLYDIM")
  - Una gramatica de operaciones sobre OBJECT_ND que preserve superposicion
    (las operaciones actuales de lenguajes existentes colapsan)
  - Un mecanismo de inferencia dimensional que opere sobre hipervectores
    (PEND_002: inferencia en modo INFERRED)

### La pregunta que la investigacion plantea pero no responde
Si dos IAs de arquitecturas distintas (un transformer y un sistema VSA puro)
intercambian un OBJECT_ND — ¿como se garantiza que "DIM_PYTHON" en la IA A
y "DIM_PYTHON" en la IA B son el mismo subespacio?

Esta es la pregunta de alineacion de embedding cross-architecture.
Es el problema mas dificil de POLYDIM y el mas nuevo — no tiene
precedente en la literatura actual.

---
*RESEARCH_FUNDAMENTOS_COGNITIVOS_V0.md — 2026-06-10*
