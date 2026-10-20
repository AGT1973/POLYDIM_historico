"""
Tests explícitos de Tarea 13. Se ejecutan ANTES de entregar el resultado,
siguiendo la preferencia de revisar toda entrega antes de mandarla.
"""
import numpy as np
from benchmark_manifold_vectorization import (
    generar_swiss_roll,
    distancias_bucle_naive,
    distancias_vectorizadas,
    benchmark_una_escala,
    correr_framework,
)


def test_distancia_a_si_mismo_es_cero():
    datos = generar_swiss_roll(20, ruido=0.0)
    D = distancias_vectorizadas(datos)
    assert np.allclose(np.diag(D), 0.0, atol=1e-9)


def test_simetria_de_la_matriz_de_distancias():
    datos = generar_swiss_roll(30)
    D = distancias_vectorizadas(datos)
    assert np.allclose(D, D.T, atol=1e-9)


def test_equivalencia_naive_vs_vectorizado_caso_pequeno():
    datos = generar_swiss_roll(15)
    ref = distancias_bucle_naive(datos)
    vec = distancias_vectorizadas(datos)
    error_rel = np.max(np.abs(ref - vec)) / np.max(ref)
    assert error_rel < 1e-5, f"error relativo {error_rel} excede tolerancia"


def test_caso_conocido_dos_puntos():
    # Caso trivial verificable a mano: dos puntos separados por (3,4,0) -> distancia 5.
    datos = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    D = distancias_vectorizadas(datos)
    assert abs(D[0, 1] - 5.0) < 1e-9
    assert abs(D[1, 0] - 5.0) < 1e-9


def test_benchmark_reporta_pass_en_escalas_estandar():
    resultados = correr_framework(escalas=(20, 40))
    assert all(r.correcto for r in resultados)
    assert all(r.speedup > 1.0 for r in resultados), "la versión vectorizada debe ser más rápida"


def test_benchmark_detecta_fallo_si_se_corrompe_la_implementacion():
    # Sanity check del propio framework: si la vectorizada estuviera mal,
    # el framework tiene que poder detectarlo (no solo reportar éxitos).
    r = benchmark_una_escala(30, tolerancia_relativa=0.0)
    # con tolerancia cero, cualquier ruido de punto flotante residual
    # (inevitable al restar en la identidad de norma expandida) debe
    # hacer que falle -> confirma que el chequeo realmente compara y
    # no siempre reporta PASS de forma vacía.
    assert r.correcto is False


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
