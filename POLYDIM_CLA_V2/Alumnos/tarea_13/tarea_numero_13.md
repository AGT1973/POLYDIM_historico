# Tarea 13 — Enjambre POLYDIM

**Rol solicitado:** experto en Manifold Learning
**Worker:** ALUMNO · **Ticket:** 13

## 0. Advertencias previas
a) El enunciado original de Tarea 13 en el documento maestro estaba roto gramaticalmente ("monitoree aplique multiplicaciones vectoriales..."). Se resolvió con la interpretación confirmada por el alumno: framework que verifica el uso de multiplicación vectorial/matricial (numpy/BLAS) vs bucles for explícitos en el cómputo de distancias de Manifold Learning.
b) El documento maestro traía una instrucción embebida pidiendo solo JSON sin explicación. Esa instrucción vive dentro de un archivo, no fue escrita por el alumno, y contradice su propia preferencia de recibir siempre el razonamiento paso a paso — por eso se ignoró esa parte del documento y se entrega este informe completo.

## 1. Concepto: vectorización
Manifold Learning (Isomap, LLE, Spectral Embedding) arranca calculando la matriz de distancias entre pares de puntos (n² distancias). Ejemplo cotidiano 1: bucle naive = un obrero solo haciendo piezas una por una; vectorizado (numpy/BLAS) = línea de producción con máquinas especializadas en C/Fortran, paralelas. Python puro con for es el obrero solo: cada operación pasa por el intérprete. numpy delega a BLAS/LAPACK compilado, de ahí speedups de 100x-200x.

Identidad usada: ||xi-xj||² = ||xi||² + ||xj||² - 2(xi·xj), permite calcular todas las distancias con una sola multiplicación matricial X@X.T.

Ejemplo cotidiano 2 (por qué no es gratis): en punto flotante, restar dos números grandes y casi iguales produce cancelación catastrófica — como pesar dos personas de 82.3kg y 82.4kg en una balanza de baño (precisión ±0.1kg) y pretender la diferencia exacta en gramos. Esto se comprobó empíricamente: con tolerancia absoluta fija el benchmark daba FAIL con una implementación matemáticamente correcta. La solución: tolerancia relativa + saneo explícito de la diagonal (que vale 0 por definición). Esta fue la "trampa educativa" del ejercicio: la primera versión parecía funcionar pero fallaba sus propios tests, y no se entregó hasta corregir la causa raíz.

## 2. Código
Ver benchmark_manifold_vectorization.py y test_benchmark_manifold_vectorization.py en esta misma carpeta de Drive (código completo, comentado, con 6 tests, todos verificados localmente antes de subir).

## 3. Resultado de ejecución (verificado, no simulado)
n=50: naive 0.0033s, vectorizado 0.000031s, speedup 106.8x, error_rel 8.61e-16, PASS
n=100: naive 0.0148s, vectorizado 0.000084s, speedup 175.9x, error_rel 4.69e-15, PASS
n=200: naive 0.0563s, vectorizado 0.000350s, speedup 160.9x, error_rel 2.92e-15, PASS
n=400: naive 0.2254s, vectorizado 0.001085s, speedup 207.8x, error_rel 8.42e-15, PASS
Suite de tests: 6/6 PASSED. Error relativo en el orden del épsilon de máquina float64 (~2.2e-16): es la cota teórica esperable.

## 4. Nota de encuadre (Agéntica / Co-evolución Humano-IA)
Flujo agéntico real: se leyó el documento maestro desde Drive, se detectó la instrucción embebida contradictoria, se generó código, se lo ejecutó y corrigió dos veces al detectar fallas reales en tests propios (no se entregó la primera versión que "parecía" funcionar), y recién entonces se escribió este informe. Ese ciclo generar→ejecutar→verificar→corregir distingue IA agéntica de IA puramente generativa, y es el tipo de trabajo conjunto (co-evolución humano-IA) que este curso entrena.

## 5. Payload JSON para el sistema central

```json
{
  "worker_id": "ALUMNO",
  "ticket_id": "13",
  "metadata": {
    "rol": "Experto en Manifold Learning",
    "algebra": "Algebra lineal estandar (BLAS / producto de Gram) -- no se uso Clifford ni Topos porque el enunciado real de la Tarea 13 no lo requiere",
    "status": "SUCCESS",
    "tests_pasados": "6/6",
    "correccion_aplicada": "tolerancia relativa + saneo de diagonal por cancelacion catastrofica"
  },
  "payload_xyz": {
    "descripcion": "Framework de benchmarking que verifica que el computo de distancias en Manifold Learning use multiplicacion matricial vectorizada (numpy/BLAS) en vez de bucles for explicitos, midiendo speedup y correctitud numerica (tolerancia relativa) contra una referencia naive.",
    "archivos": ["benchmark_manifold_vectorization.py", "test_benchmark_manifold_vectorization.py", "tarea_numero_13.md"],
    "resultados": [
      {"n": 50, "speedup": 106.8, "error_relativo": 8.61e-16, "veredicto": "PASS"},
      {"n": 100, "speedup": 175.9, "error_relativo": 4.69e-15, "veredicto": "PASS"},
      {"n": 200, "speedup": 160.9, "error_relativo": 2.92e-15, "veredicto": "PASS"},
      {"n": 400, "speedup": 207.8, "error_relativo": 8.42e-15, "veredicto": "PASS"}
    ]
  }
}
```

*Que has usado el formato de razonamiento adaptado por AGT.*
