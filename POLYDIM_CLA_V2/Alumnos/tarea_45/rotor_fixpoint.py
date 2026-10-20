"""
Tarea 45 - POLYDIM: Convergencia de un tensor de estado mediante superposicion
lineal de rotores (Spin(3) / cuaterniones unitarios, subalgebra par de Cl(3,0))
comparada contra composicion de angulos de Euler, con inspeccion de volcados
intermedios del grafo de computo (usando jax.make_jaxpr).

Hipotesis matematica a verificar empiricamente (NO asumida por enunciado, se
prueba aqui): la composicion encadenada de rotores de Clifford (equivalente a
cuaterniones unitarios en Spin(3)) es una operacion suave en toda la variedad
S^3, por lo que una iteracion de punto fijo (tipo Banach) que interpola
geodesicamente hacia un rotor objetivo converge sin discontinuidades ni
perdida de rango en el Jacobiano. En contraste, la composicion equivalente via
angulos de Euler (parametrizacion local de SO(3)) sufre gimbal lock: existen
configuraciones donde el Jacobiano de la carta se vuelve singular.

Dependencias: numpy, jax (opcional para el volcado de grafo).
Instalacion:  pip install numpy jax --break-system-packages
Ejecucion:    python rotor_fixpoint.py
"""

import numpy as np

# ----------------------------------------------------------------------------
# 1. Algebra de cuaterniones unitarios == subalgebra par de Cl(3,0)
#    Un rotor R = cos(theta/2) - sin(theta/2) * B  (B bivector unitario)
#    se representa aqui como q = (w, x, y, z), con B <-> (x,y,z) el eje.
# ----------------------------------------------------------------------------

def quat_from_axis_angle(axis, theta):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    w = np.cos(theta / 2.0)
    xyz = np.sin(theta / 2.0) * axis
    return np.array([w, *xyz])


def quat_mul(q1, q2):
    """Producto geometrico (Hamilton) de dos rotores. No conmutativo."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def sandwich(R, v):
    """Aplica el rotor R a un vector v via v' = R v R^-1 (producto sandwich)."""
    qv = np.array([0.0, *v])
    q_res = quat_mul(quat_mul(R, qv), quat_conj(R))
    return q_res[1:]


def quat_geodesic_distance(q1, q2):
    """Distancia geodesica en S^3 (doble cubrimiento de SO(3))."""
    d = abs(np.clip(np.dot(q1, q2), -1.0, 1.0))
    return 2.0 * np.arccos(d)


def slerp(q1, q2, t):
    """Interpolacion esferica: suave en toda la variedad, sin cartas locales."""
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return quat_normalize(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q_perp = quat_normalize(q2 - q1 * dot)
    return q1 * np.cos(theta) + q_perp * np.sin(theta)


# ----------------------------------------------------------------------------
# 2. Composicion encadenada de rotores == superposicion lineal en el "MIX"
#    de POLYDIM. Se construye una cadena de N rotores aleatorios y se compone.
# ----------------------------------------------------------------------------

def chain_compose_rotors(rotors):
    R_total = np.array([1.0, 0.0, 0.0, 0.0])  # identidad (GEO_ID)
    for R in rotors:
        R_total = quat_mul(R, R_total)
    return quat_normalize(R_total)


# ----------------------------------------------------------------------------
# 3. Iteracion de punto fijo tipo FIXPOINT (contraccion de Banach) usando
#    interpolacion geodesica hacia un rotor objetivo R_target.
#    lambda_ = 1 - t  actua como constante de Lipschitz de la contraccion.
# ----------------------------------------------------------------------------

def fixpoint_iteration_rotor(R_start, R_target, t=0.35, n_iters=40, tol=1e-9):
    history = []
    R = R_start.copy()
    for k in range(n_iters):
        d = quat_geodesic_distance(R, R_target)
        history.append(d)
        if d < tol:
            break
        R = slerp(R, R_target, t)
    return R, history


# ----------------------------------------------------------------------------
# 4. Contraparte con angulos de Euler (ZYX) para exhibir el gimbal lock.
#    Composicion de rotaciones -> matrices; se mide el Jacobiano de la carta
#    (mapa angulos -> matriz de rotacion) via diferencias finitas y su rango.
# ----------------------------------------------------------------------------

def euler_zyx_to_matrix(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def euler_jacobian_condition_number(angles, eps=1e-6):
    """Jacobiano numerico del mapa (yaw,pitch,roll) -> vec(R) (9 componentes)."""
    yaw, pitch, roll = angles
    base = euler_zyx_to_matrix(yaw, pitch, roll).flatten()
    J = np.zeros((9, 3))
    for i in range(3):
        d = np.array(angles, dtype=float)
        d[i] += eps
        pert = euler_zyx_to_matrix(*d).flatten()
        J[:, i] = (pert - base) / eps
    singular_values = np.linalg.svd(J, compute_uv=False)
    sv = singular_values[singular_values > 1e-14]
    if len(sv) < 2:
        return np.inf  # Jacobiano con rango deficiente: singularidad real
    return sv[0] / sv[-1]


# ----------------------------------------------------------------------------
# 5. Volcado intermedio del grafo de computo (opcional, requiere jax) para
#    exhibir que la composicion de rotores no introduce ramas condicionales,
#    mientras que un manejo robusto de Euler cerca del gimbal lock si lo haria.
# ----------------------------------------------------------------------------

def dump_computation_graph():
    try:
        import jax
        import jax.numpy as jnp

        def rotor_chain_fn(thetas):
            axis = jnp.array([0.0, 0.0, 1.0])
            R = jnp.array([1.0, 0.0, 0.0, 0.0])
            for th in thetas:
                w = jnp.cos(th / 2.0)
                xyz = jnp.sin(th / 2.0) * axis
                Rk = jnp.concatenate([jnp.array([w]), xyz])
                w1, x1, y1, z1 = Rk
                w2, x2, y2, z2 = R
                R = jnp.array([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                ])
            return R

        thetas = jnp.array([0.3, 0.7, 1.1, 0.2])
        jaxpr = jax.make_jaxpr(rotor_chain_fn)(thetas)
        print("=== Volcado jaxpr (grafo intermedio, composicion de rotores) ===")
        print(jaxpr)
        print("Numero de ecuaciones en el grafo:", len(jaxpr.jaxpr.eqns))
        print("Presencia de nodos condicionales (cond/switch):",
              any(eq.primitive.name in ("cond", "switch") for eq in jaxpr.jaxpr.eqns))
    except ImportError:
        print("[aviso] jax no esta instalado; se omite el volcado de grafo. "
              "Instalar con: pip install jax --break-system-packages")


# ----------------------------------------------------------------------------
# Demostracion principal
# ----------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)

    print("### Paso 1: composicion encadenada de N rotores aleatorios ###")
    axes = rng.normal(size=(8, 3))
    thetas = rng.uniform(0.1, 1.5, size=8)
    rotors = [quat_normalize(quat_from_axis_angle(a, t)) for a, t in zip(axes, thetas)]
    R_chain = chain_compose_rotors(rotors)
    v0 = np.array([1.0, 0.0, 0.0])
    v1 = sandwich(R_chain, v0)
    print("Norma de v0:", np.linalg.norm(v0), " Norma de v1 (post-rotor):", np.linalg.norm(v1))
    print("-> La superposicion lineal de rotores preserva la norma (volumen) del tensor.\n")

    print("### Paso 2: iteracion de punto fijo (FIXPOINT) via slerp ###")
    R_target = quat_normalize(quat_from_axis_angle([0.0, 1.0, 0.0], np.pi / 2))
    R_final, history = fixpoint_iteration_rotor(R_chain, R_target, t=0.35, n_iters=40)
    print(f"Iteraciones hasta tolerancia: {len(history)}")
    print(f"Distancia geodesica inicial: {history[0]:.6f}, final: {history[-1]:.3e}")
    ratios = [history[i+1] / history[i] for i in range(len(history) - 1) if history[i] > 1e-12]
    if ratios:
        print(f"Constante de contraccion empirica (promedio): {np.mean(ratios):.4f} "
              f"(teorica: {1 - 0.35:.4f})\n")

    print("### Paso 3: comparacion con angulos de Euler (ZYX) - deteccion de gimbal lock ###")
    test_configs = [
        ("configuracion regular", (0.3, 0.2, 0.5)),
        ("cerca de pitch = +90 deg (gimbal lock)", (0.3, np.pi/2 - 1e-4, 0.5)),
        ("exactamente en pitch = 90 deg", (0.3, np.pi/2, 0.5)),
    ]
    for name, angles in test_configs:
        cond = euler_jacobian_condition_number(angles)
        print(f"  {name}: numero de condicion del Jacobiano = {cond}")
    print("-> El numero de condicion diverge (o el rango cae) cerca de pitch=90 deg: "
          "gimbal lock. La parametrizacion por rotores/cuaterniones no exhibe esto "
          "porque no depende de una carta angular local.\n")

    print("### Paso 4: volcado intermedio del grafo de computo (jax) ###")
    dump_computation_graph()


if __name__ == "__main__":
    main()
