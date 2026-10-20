"""
POLYDIM SWARM - Tarea 8 (Escuadron 1: Relajacion Topologica y Homotopia)
Rol solicitado: Arquitecto de Sistemas Distribuidos
Enunciado: Escribe una demostracion teorica implementada como codigo donde
expliques por que aplicar multiplicaciones vectoriales usando librerias
matematicas compatibles (en vez de implementaciones ad hoc, mezcladas
o incompatibles) preserva la estructura topologica del manifold de memoria
en un sistema distribuido.

Este script es la evidencia empirica que respalda la demostracion teorica
explicada en tarea_numero_8.md. No usa estructuras IF/ELSE para el ruteo
de datos: el "ruteo" entre nodos se define por la forma (shape) de los
tensores, tal como exige la Constitucion V2 del enjambre.

worker_id: cursos.python.agt@gmail.com
ticket_id: 8
"""

import numpy as np

RNG = np.random.default_rng(seed=8)

# ---------------------------------------------------------------------------
# 1. CONFIGURACION DEL EXPERIMENTO
# ---------------------------------------------------------------------------
DIM_ENTRADA = 512      # dimension del "vector de memoria" (embedding)
RANGO_BAJO = 16         # rango r de la factorizacion de bajo rango (r << n)
N_NODOS = 64            # numero de shards / nodos distribuidos simulados
N_REPETICIONES = 12      # repeticiones para medir estabilidad (varianza)
TOLERANCIA_SPS = 0.999   # umbral de aceptacion para Semantic Preservation Score
TOLERANCIA_GEOID = 1e-4  # umbral de aceptacion para GeoID Stability (var. relativa)

assert DIM_ENTRADA % N_NODOS == 0, "El vector debe poder repartirse en shards iguales"
TAM_SHARD = DIM_ENTRADA // N_NODOS


# ---------------------------------------------------------------------------
# 2. OPERADOR "COMPATIBLE": factorizacion de bajo rango (permitido por la
#    Constitucion V2, art. 1: prohibido usar matrices densas si no es
#    estrictamente necesario).
#    W = U @ V^T   con U, V de rango r.  Aplicar W @ x se hace SIN
#    materializar nunca la matriz densa n x n: se contrae primero con V
#    (proyeccion al subespacio latente) y luego con U (reconstruccion).
#    Todas las contracciones se hacen con la MISMA libreria (numpy),
#    el MISMO dtype (float32) y el MISMO orden de reduccion en todos
#    los nodos: por eso decimos que es "compatible".
# ---------------------------------------------------------------------------
U = RNG.standard_normal((DIM_ENTRADA, RANGO_BAJO)).astype(np.float32) / np.sqrt(RANGO_BAJO)
V = RNG.standard_normal((DIM_ENTRADA, RANGO_BAJO)).astype(np.float32) / np.sqrt(RANGO_BAJO)


def multiplicacion_compatible(x, U=U, V=V, n_nodos=N_NODOS):
    """
    Aplica W @ x = U @ (V^T @ x) repartiendo x en n_nodos shards.
    Cada nodo calcula su contribucion parcial al subespacio latente
    (V_shard^T @ x_shard, un vector de tamano r, NO n) y esas
    contribuciones se suman en un orden de reduccion FIJO (arbol binario
    determinista) antes de reconstruir con U. El trafico de red es
    O(n_nodos * r), no O(n_nodos * n): eso es lo que hace viable la
    version distribuida sin usar matrices densas.
    """
    x = x.astype(np.float32)
    parciales_latentes = []
    for i in range(n_nodos):
        ini, fin = i * TAM_SHARD, (i + 1) * TAM_SHARD
        x_shard = x[ini:fin]
        V_shard = V[ini:fin, :]
        # contraccion local: (r,) = (tam_shard, r)^T @ (tam_shard,)
        parciales_latentes.append(V_shard.T @ x_shard)

    # reduccion en arbol binario determinista (mismo orden siempre)
    nivel = parciales_latentes
    while len(nivel) > 1:
        siguiente = []
        for i in range(0, len(nivel) - 1, 2):
            siguiente.append(nivel[i] + nivel[i + 1])
        if len(nivel) % 2 == 1:
            siguiente.append(nivel[-1])
        nivel = siguiente
    latente = nivel[0]  # (r,)

    return U @ latente  # (n,)


# ---------------------------------------------------------------------------
# 3. OPERADOR "INCOMPATIBLE": incluso usando la MISMA matriz densa
#    equivalente W_densa = U @ V^T y la MISMA libreria numpy, si cada nodo
#    (o cada corrida) suma sus contribuciones parciales EN UN ORDEN
#    DIFERENTE (por ejemplo: orden secuencial vs orden aleatorio, como
#    ocurriria si el scheduler de un cluster real reordena la llegada
#    de mensajes), el resultado en float32 no es identico. Este es el
#    nucleo de la "trampa educativa": el problema no es la libreria en
#    si, sino que la suma en punto flotante NO es asociativa.
# ---------------------------------------------------------------------------
W_densa = (U @ V.T).astype(np.float32)


def multiplicacion_incompatible(x, orden_shards, U=U, V=V):
    """
    Mismo calculo matematico que multiplicacion_compatible, pero con dos
    diferencias que en la practica ocurren cuando los nodos de un cluster
    NO acordaron el mismo "contrato numerico" (aunque todos usen numpy):

      (a) cada nodo acumula su contribucion parcial en float16 -- una
          eleccion habitual para ahorrar ancho de banda de red al enviar
          activaciones entre nodos -- y luego se sube a float32 recien
          al llegar al coordinador;
      (b) el coordinador va sumando esas contribuciones en el orden de
          llegada por red, que es ARBITRARIO (jitter), no en el arbol
          binario determinista de la version compatible.

    Ninguna de las dos cosas es "ilegal" ni un error de programacion:
    cada nodo, por separado, esta matematicamente en lo correcto. Ahi
    esta la trampa.
    """
    x = x.astype(np.float32)
    parciales_latentes = []
    for i in orden_shards:
        ini, fin = i * TAM_SHARD, (i + 1) * TAM_SHARD
        x_shard = x[ini:fin]
        V_shard = V[ini:fin, :]
        parcial_f16 = (V_shard.T @ x_shard).astype(np.float16)  # (a)
        parciales_latentes.append(parcial_f16.astype(np.float32))

    latente = parciales_latentes[0].copy()
    for p in parciales_latentes[1:]:
        latente = latente + p  # (b) suma secuencial en el orden de llegada

    return U @ latente


# ---------------------------------------------------------------------------
# 4. GROUND TRUTH EN FLOAT64 (referencia de alta precision, no distribuida)
# ---------------------------------------------------------------------------
def ground_truth_float64(x):
    U64, V64 = U.astype(np.float64), V.astype(np.float64)
    return U64 @ (V64.T @ x.astype(np.float64))


# ---------------------------------------------------------------------------
# 5. METRICAS DE LA PIRAMIDE DE VALIDACION
# ---------------------------------------------------------------------------
def sps_semantic_preservation_score(y_ref, y_test):
    """Similitud coseno entre el vector de referencia y el vector obtenido."""
    y_ref = y_ref.astype(np.float64)
    y_test = y_test.astype(np.float64)
    num = np.dot(y_ref, y_test)
    den = np.linalg.norm(y_ref) * np.linalg.norm(y_test)
    return float(num / den) if den > 0 else 0.0


def geoid_stability(resultados):
    """
    Estabilidad del identificador geometrico: varianza relativa entre
    N_REPETICIONES corridas del mismo calculo lógico. Un operador
    "compatible" (orden de reduccion fijo) debe dar varianza EXACTAMENTE
    cero. Un operador "incompatible" (orden variable) da varianza > 0.
    """
    resultados = np.stack(resultados).astype(np.float64)
    norma_media = np.linalg.norm(resultados.mean(axis=0))
    dispersion = resultados.std(axis=0)
    return float(np.linalg.norm(dispersion) / norma_media) if norma_media > 0 else 0.0


# ---------------------------------------------------------------------------
# 6. EJECUCION DEL EXPERIMENTO
# ---------------------------------------------------------------------------
def ejecutar_experimento():
    # Vector de entrada con magnitudes MUY dispares (varios ordenes de
    # magnitud), tal como ocurre con activaciones reales tras una
    # normalizacion imperfecta o con outliers de atencion. Esto es lo
    # que realmente pone a prueba la asociatividad de la suma en punto
    # flotante: sumar numeros de magnitudes muy distintas en distinto
    # orden SI cambia el resultado de forma medible.
    signos = RNG.choice([-1.0, 1.0], size=DIM_ENTRADA)
    magnitudes = 10.0 ** RNG.uniform(-3, 3, size=DIM_ENTRADA)  # 1e-3 .. 1e3
    x = (signos * magnitudes).astype(np.float32)
    y_ref = ground_truth_float64(x)

    # --- Rama compatible: repetimos N veces, el orden de reduccion nunca cambia
    resultados_compatibles = [multiplicacion_compatible(x) for _ in range(N_REPETICIONES)]
    sps_compatible = sps_semantic_preservation_score(y_ref, resultados_compatibles[0])
    geoid_compatible = geoid_stability(resultados_compatibles)

    # --- Rama incompatible: repetimos N veces, cada vez con un orden de
    #     llegada de shards distinto (simulando jitter de red real)
    resultados_incompatibles = []
    for _ in range(N_REPETICIONES):
        orden = RNG.permutation(N_NODOS)
        resultados_incompatibles.append(multiplicacion_incompatible(x, orden))
    sps_incompatible = sps_semantic_preservation_score(y_ref, resultados_incompatibles[0])
    geoid_incompatible = geoid_stability(resultados_incompatibles)

    # --- Chequeo adicional: norma espectral (proxy de estabilidad de Jacobiano)
    #     del operador de bajo rango vs una matriz densa aleatoria del mismo
    #     tamano pero SIN restriccion de rango (la que prohibe la Constitucion).
    W_densa_libre = RNG.standard_normal((DIM_ENTRADA, DIM_ENTRADA)).astype(np.float32)
    sv_bajo_rango = np.linalg.svd(W_densa, compute_uv=False)[0]
    sv_densa_libre = np.linalg.svd(W_densa_libre, compute_uv=False)[0]

    # --- Extension: efecto acumulado al ENCADENAR la operacion muchas
    #     veces, como ocurriria al atravesar muchas capas/saltos de
    #     ruteo en el manifold. Un desvio de 1e-8 por paso parece
    #     inofensivo, pero en un sistema distribuido real la misma
    #     operacion se ejecuta miles de veces (una por capa, por token,
    #     por salto de red). Aca lo hacemos explicito componiendo el
    #     operador consigo mismo N_CAPAS veces y comparando la deriva
    #     acumulada de la version compatible vs la incompatible.
    N_CAPAS = 500

    def encadenar(operador, x0):
        x = x0.copy()
        for _ in range(N_CAPAS):
            y = operador(x)
            x = y / (np.linalg.norm(y) + 1e-12) * np.linalg.norm(x0)  # renormalizar
        return x

    x0 = x.copy()
    final_compatible = encadenar(lambda v: multiplicacion_compatible(v), x0)
    orden_fijo_incompatible = RNG.permutation(N_NODOS)
    final_incompatible = encadenar(
        lambda v: multiplicacion_incompatible(v, orden_fijo_incompatible), x0
    )
    y_ref_final = x0.copy()
    for _ in range(N_CAPAS):
        y = ground_truth_float64(y_ref_final)
        y_ref_final = (y / (np.linalg.norm(y) + 1e-12) * np.linalg.norm(x0)).astype(np.float32)

    sps_encadenado_compatible = sps_semantic_preservation_score(y_ref_final, final_compatible)
    sps_encadenado_incompatible = sps_semantic_preservation_score(y_ref_final, final_incompatible)

    reporte = {
        "dim_entrada": DIM_ENTRADA,
        "rango_bajo": RANGO_BAJO,
        "n_nodos": N_NODOS,
        "n_repeticiones": N_REPETICIONES,
        "compatible": {
            "sps": sps_compatible,
            "geoid_stability_varianza_relativa": geoid_compatible,
            "resultado_status": "SUCCESS" if (
                sps_compatible >= TOLERANCIA_SPS and geoid_compatible <= TOLERANCIA_GEOID
            ) else "MATH_FAILURE",
        },
        "incompatible": {
            "sps": sps_incompatible,
            "geoid_stability_varianza_relativa": geoid_incompatible,
            "resultado_status": "SUCCESS" if (
                sps_incompatible >= TOLERANCIA_SPS and geoid_incompatible <= TOLERANCIA_GEOID
            ) else "MATH_FAILURE",
        },
        "jacobiano_proxy": {
            "mayor_valor_singular_bajo_rango": float(sv_bajo_rango),
            "mayor_valor_singular_densa_libre": float(sv_densa_libre),
        },
        "encadenamiento_n_capas": N_CAPAS,
        "sps_encadenado_compatible": sps_encadenado_compatible,
        "sps_encadenado_incompatible": sps_encadenado_incompatible,
        "sps_encadenado_status_incompatible": "SUCCESS" if sps_encadenado_incompatible >= TOLERANCIA_SPS else "MATH_FAILURE",
    }
    return reporte


if __name__ == "__main__":
    reporte = ejecutar_experimento()
    import json
    print(json.dumps(reporte, indent=2, ensure_ascii=False))

# que has usado el formato de razonamiento adaptado por AGT
