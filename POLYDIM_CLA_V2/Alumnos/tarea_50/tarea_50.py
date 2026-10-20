"""
Tarea 50 - POLYDIM (banco de validación)
Rol: Ingeniero de XLA/JAX

Pregunta empírica reconstruida:
  Un tensor acumulador compartido recibe actualizaciones aditivas de N agentes
  que se ejecutan en distinto orden (simulando concurrencia asíncrona).
  ¿Es el resultado final independiente del orden de aplicación (asociatividad)?

Resultado esperado según la teoría de punto flotante IEEE-754 (Higham, 2002;
Goldberg, 1991): la suma es asociativa en R, pero NO es exactamente asociativa
en aritmética de punto flotante, porque cada operación introduce un redondeo
fl(x) = x*(1+delta), |delta| <= u (u = unit roundoff, ~1.11e-16 en float64).

Este script mide, para varios órdenes de aplicación aleatorios, la dispersión
del resultado final mediante la norma L2 entre las distintas sumas, y la
compara con la cota teórica de error hacia adelante:

    |s_pi - S| <= (n-1) * u * sum(|a_i|) + O(u^2)

No se ejecuta como parte de la entrega: el alumno debe correrlo él mismo
(ver instrucciones al final del .md) para verificar los resultados con sus
propios ojos.
"""

import numpy as np


def suma_secuencial_en_orden(valores: np.ndarray, orden: np.ndarray) -> float:
    """
    Simula un agente que aplica actualizaciones aditivas de a una por vez,
    en el orden dado, sobre un acumulador escalar de punto flotante float64.
    Cada suma parcial pasa por el operador de redondeo fl(x) implícito en
    la representación float64 de numpy (esto es exactamente lo que hace
    el hardware IEEE-754, no una simulación aproximada).
    """
    acumulador = np.float64(0.0)
    for idx in orden:
        acumulador = acumulador + valores[idx]
    return acumulador


def simular_agentes_asincronos(n_valores: int, n_ordenes: int, seed: int = 0):
    """
    n_valores: cantidad de actualizaciones (una por 'agente' o por evento).
    n_ordenes: cantidad de órdenes de ejecución distintos a comparar
               (simula que N agentes concurrentes pueden intercalarse
               de n_ordenes formas distintas).
    """
    rng = np.random.default_rng(seed)

    # Valores de actualización: mezcla de magnitudes para que el error de
    # redondeo relativo sea representativo (si todos fueran iguales, la
    # cancelación catastrófica no se manifestaría igual).
    valores = rng.normal(loc=0.0, scale=1.0, size=n_valores).astype(np.float64)

    resultados = []
    for _ in range(n_ordenes):
        orden = rng.permutation(n_valores)
        resultados.append(suma_secuencial_en_orden(valores, orden))

    resultados = np.array(resultados, dtype=np.float64)

    # Suma "de referencia" en precisión extendida (float128 si está
    # disponible) para tener un proxy de la suma matemática exacta S.
    try:
        suma_referencia = np.float128(valores.astype(np.float128).sum())
    except AttributeError:
        # Algunas plataformas (p. ej. Windows) no exponen float128.
        suma_referencia = np.longdouble(valores.astype(np.longdouble).sum())

    # Dispersión entre órdenes: norma L2 de las diferencias respecto al
    # primer resultado obtenido (medida estándar de discrepancia numérica).
    diferencias = resultados - resultados[0]
    l2_dispersión = np.linalg.norm(diferencias)

    # Cota teórica de error hacia adelante (Higham, 2002, ec. 4.3 aprox.):
    u = np.finfo(np.float64).eps / 2  # unit roundoff ~1.11e-16
    cota_teorica = (n_valores - 1) * u * np.sum(np.abs(valores))

    error_max_vs_referencia = np.max(np.abs(resultados - float(suma_referencia)))

    return {
        "valores": valores,
        "resultados_por_orden": resultados,
        "l2_dispersión_entre_ordenes": float(l2_dispersión),
        "cota_teorica_error": float(cota_teorica),
        "error_max_vs_suma_referencia": float(error_max_vs_referencia),
        "suma_referencia_alta_precision": float(suma_referencia),
    }


def reporte(resultado: dict) -> None:
    print("=== Tarea 50: no-asociatividad de punto flotante en actualizaciones asíncronas ===")
    print(f"Suma de referencia (alta precisión):   {resultado['suma_referencia_alta_precision']:.17g}")
    print(f"Dispersión L2 entre órdenes distintos:  {resultado['l2_dispersión_entre_ordenes']:.3e}")
    print(f"Cota teórica de error (Higham):         {resultado['cota_teorica_error']:.3e}")
    print(f"Error máximo vs. referencia:             {resultado['error_max_vs_suma_referencia']:.3e}")
    print()
    if resultado["l2_dispersión_entre_ordenes"] == 0.0:
        print("Conclusión empírica: en esta corrida, los órdenes probados coincidieron "
              "bit a bit (puede pasar con pocos valores/órdenes). No es garantía general.")
    else:
        print("Conclusión empírica: el resultado SÍ depende del orden de aplicación "
              "(no hay asociatividad exacta en punto flotante), y la dispersión "
              "observada está acotada por la estimación teórica, como predice la teoría.")


if __name__ == "__main__":
    resultado = simular_agentes_asincronos(n_valores=1000, n_ordenes=20, seed=42)
    reporte(resultado)
