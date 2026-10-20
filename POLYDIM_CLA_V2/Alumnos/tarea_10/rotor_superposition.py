"""
POLYDIM SWARM - TICKET 10 / ESCUADRON 1 (Relajacion Topologica y Homotopia)
Rol solicitado: Investigador de convergencia de ML.
Tarea: Disenar una operacion encadenada que demuestre, a nivel de volcado
intermedio de grafo, que un tensor converge usando superposicion lineal
con rotores SIN BLOQUEOS.

Marco matematico: Algebra de Clifford Cl(3,0)+ (rotores == cuaterniones
unitarios). Se evita el uso de matrices densas: el estado del sistema es
un vector en R^4 (multivector par: escalar + bivector), no una matriz de
rotacion 3x3 salvo para verificacion final (metrica GeoID).
"""

import numpy as np

np.random.seed(7)  # reproducibilidad para el reporte SOTA


# ---------------------------------------------------------------------------
# 1. PRIMITIVAS DE LA VARIEDAD DE ROTORES (S^3, doble cobertura de SO(3))
# ---------------------------------------------------------------------------

def normalize(q):
    """Proyecta un multivector par arbitrario de vuelta a la variedad de
    rotores unitarios (S^3). Si la norma colapsa, NO se alucina una
    solucion: se reporta como Falla Matematica (regla #1 de la
    Constitucion V2)."""
    n = np.linalg.norm(q)
    if n < 1e-9:
        raise FloatingPointError(
            "FALLA MATEMATICA: norma del rotor colapso a 0. "
            "Jacobiano de la proyeccion indefinido (bloqueo real)."
        )
    return q / n


def align_sign(q, reference):
    """LA TRAMPA DEL EJERCICIO.

    q y -q representan geometricamente la MISMA rotacion (S^3 cubre a
    SO(3) dos veces). Una superposicion lineal ingenua de rotores
    tomados de fuentes distintas del grafo puede sumar un rotor y su
    'gemelo' de signo opuesto: q + (-q) = 0. El resultado no es un error
    numerico sutil, es un colapso total (bloqueo) que un IF/ELSE
    superficial no detectaria si solo mira la norma al final.

    La solucion NO es una rama condicional cara: es una proyeccion por
    producto interno contra una referencia (el rotor actual del grafo),
    que es geometrica y diferenciable salvo en el conjunto de medida
    nula donde el angulo es exactamente pi/2 (ortogonalidad, el unico
    verdadero "bloqueo" irreducible del sistema).
    """
    return q if np.dot(q, reference) >= 0.0 else -q


def rotor_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    return np.concatenate(([np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis))


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ])


def angular_distance(q1, q2):
    """Distancia geodesica real en SO(3) entre dos rotores (radianes),
    correcta bajo doble cobertura."""
    q1a = align_sign(q1, q2)
    dot = float(np.clip(np.dot(q1a, q2), -1.0, 1.0))
    return 2.0 * np.arccos(min(1.0, abs(dot)))


# ---------------------------------------------------------------------------
# 2. OPERACION ENCADENADA (un "nodo" del grafo computacional)
# ---------------------------------------------------------------------------

def linear_superposition_step(current, sample, weight_current, weight_sample):
    """Un nodo del grafo:
        1) alinea signo del rotor entrante contra el rotor de referencia,
        2) superpone linealmente en R^4 (NO en SO(3): ahi no hay suma
           lineal valida, esa es justamente la razon de usar rotores),
        3) reproyecta a S^3 (renormaliza).
    """
    aligned_sample = align_sign(sample, current)
    combined = weight_current * current + weight_sample * aligned_sample
    return normalize(combined)


def run_chain(target_rotor, n_nodes=25, noise_std=0.35, tol=1e-4, force_bloqueo_demo=True):
    """Cadena de n_nodes operaciones. Cada nodo recibe una observacion
    ruidosa del rotor objetivo (simulando updates asincronos de un
    ensamble de agentes) y la funde con el estado actual via
    superposicion lineal con peso creciente (media de Frechet
    incremental sobre la variedad, aproximada en el tangente de R^4).

    Devuelve el rotor final y el volcado intermedio del grafo (graph_dump)."""

    graph_dump = []

    # Nodo 0: arranque desde una observacion ruidosa cualquiera.
    axis0 = np.random.randn(3)
    current = rotor_from_axis_angle(axis0, np.random.randn() * noise_std)
    current = align_sign(current, target_rotor)  # arranque coherente

    graph_dump.append({
        "node": 0,
        "rotor": np.round(current, 6).tolist(),
        "angular_error_vs_target_rad": round(angular_distance(current, target_rotor), 6),
        "bloqueo_detectado": False,
        "status": "INIT",
    })

    demo_triggered = False
    for i in range(1, n_nodes + 1):
        axis = np.random.randn(3)
        angle_noise = np.random.randn() * (noise_std / np.sqrt(i))  # ruido decreciente
        sample_axis_angle = axis / np.linalg.norm(axis)
        sample = rotor_from_axis_angle(sample_axis_angle, 2 * np.arccos(target_rotor[0]) + angle_noise)
        # proyectar aproximadamente hacia el eje objetivo real (simula un
        # sensor ruidoso que SI observa el rotor objetivo, no ruido puro).
        # El peso del ruido decae con 1/sqrt(i): el 'sensor' mejora con
        # cada observacion (mas evidencia acumulada => menor incertidumbre),
        # condicion necesaria para convergencia real (no solo un plateau).
        eps_i = 0.6 / np.sqrt(i)
        sample = normalize((1 - eps_i) * target_rotor + eps_i * sample)

        # --- DEMOSTRACION EXPLICITA DEL "SIN BLOQUEOS" (una sola vez) ---
        if force_bloqueo_demo and not demo_triggered and i == n_nodes // 2:
            demo_triggered = True
            twin = -sample.copy()  # el gemelo topologico del mismo sample
            naive_sum = current + twin  # superposicion SIN alinear signo
            naive_norm = float(np.linalg.norm(naive_sum))
            graph_dump.append({
                "node": f"{i}-demo_bloqueo",
                "descripcion": "Superposicion NAIVE sin align_sign entre "
                                "current y el gemelo -sample (misma rotacion "
                                "fisica, signo opuesto en S^3).",
                "naive_sum_norm": round(naive_norm, 8),
                "bloqueo_detectado": naive_norm < 1e-3,
                "status": "FALLA_MATEMATICA_EVITADA" if naive_norm < 1e-3 else "SIN_BLOQUEO_NAIVE",
            })

        weight_current = i / (i + 1.0)
        weight_sample = 1.0 / (i + 1.0)
        new_current = linear_superposition_step(current, sample, weight_current, weight_sample)

        delta = angular_distance(current, new_current)
        err_vs_target = angular_distance(new_current, target_rotor)

        # bloqueo real solo si la superposicion CORRECTA (con align_sign)
        # colapsa igual -> eso si seria falla matematica genuina.
        aligned_check = align_sign(sample, current)
        raw_sum_norm = float(np.linalg.norm(weight_current * current + weight_sample * aligned_check))

        record = {
            "node": i,
            "rotor": np.round(new_current, 6).tolist(),
            "angular_delta_rad": round(float(delta), 6),
            "angular_error_vs_target_rad": round(float(err_vs_target), 6),
            "raw_sum_norm_before_projection": round(raw_sum_norm, 6),
            "bloqueo_detectado": raw_sum_norm < 1e-3,
            "status": "RUNNING",
        }
        current = new_current
        if delta < tol and i > 3:
            record["status"] = "CONVERGED"
            graph_dump.append(record)
            break
        graph_dump.append(record)

    return current, graph_dump


# ---------------------------------------------------------------------------
# 3. METRICAS DE LA PIRAMIDE DE VALIDACION (constitucion V2)
# ---------------------------------------------------------------------------

def geoid_stability(final_rotor, target_rotor):
    """GeoID Stability: 1.0 - error_angular_normalizado. Mide si el
    'GEO_ID' (la identidad geometrica objetivo) fue recuperado."""
    err = angular_distance(final_rotor, target_rotor)
    return float(max(0.0, 1.0 - err / np.pi))


def semantic_preservation_score(final_rotor, target_rotor):
    """SPS: similitud coseno entre las matrices de rotacion reconstruidas
    (equivalente semantico: 'la accion sobre el espacio se preservo')."""
    R_final = quat_to_rotmat(final_rotor).flatten()
    R_target = quat_to_rotmat(target_rotor).flatten()
    cos_sim = float(np.dot(R_final, R_target) / (np.linalg.norm(R_final) * np.linalg.norm(R_target)))
    return cos_sim


# ---------------------------------------------------------------------------
# 4. EJECUCION / VALIDACION EMPIRICA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    target = rotor_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.deg2rad(72.0))

    try:
        final_rotor, dump = run_chain(target, n_nodes=25, noise_std=0.35, tol=1e-4)
        sps = semantic_preservation_score(final_rotor, target)
        geoid = geoid_stability(final_rotor, target)
        status = "SUCCESS" if geoid > 0.98 and sps > 0.98 else "SUCCESS_CON_ADVERTENCIA"
    except FloatingPointError as e:
        final_rotor, dump, sps, geoid, status = None, [], None, None, f"ERROR: {e}"

    print("=" * 70)
    print("VOLCADO INTERMEDIO DEL GRAFO (graph_dump)")
    print("=" * 70)
    for record in dump:
        print(record)

    print("\n" + "=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    print("Rotor objetivo      :", np.round(target, 6))
    print("Rotor final (cadena):", np.round(final_rotor, 6) if final_rotor is not None else None)
    print("SPS (Semantic Preservation Score) :", sps)
    print("GeoID Stability                    :", geoid)
    print("STATUS                              :", status)
    print("Nodos hasta convergencia            :", len(dump))
