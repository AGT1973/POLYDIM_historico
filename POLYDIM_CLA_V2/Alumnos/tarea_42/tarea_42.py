"""
Tarea 42 - Escuadron 3: Descomposicion de Hodge y Grafos Esparsos
Rol: experto en Categorical Deep Learning

Experimento: inyectar distorsion afin aleatoria sobre dos transformaciones
del espacio latente (rotores de Clifford + traslacion) y validar
empiricamente si las dos trayectorias de composicion posibles
(T1 despues T2, vs T2 despues T1) conmutan.

No se ejecuta como parte de la entrega final: el alumno lo corre en su
propia cuenta. Requiere: numpy, scipy (pip install numpy scipy --break-system-packages)
"""
import numpy as np
from scipy.linalg import expm


def random_bivector(n, rng):
    """Genera un generador antisimetrico aleatorio (bivector de Clifford) en R^n."""
    A = rng.standard_normal((n, n))
    return (A - A.T) / 2.0


def rotor(theta, B):
    """Rotor de Spin: R = exp(theta * B). B antisimetrico => R ortogonal (SO(n))."""
    return expm(theta * B)


def commutator_experiment(n=8, n_axes_samples=30, n_points=200, theta=0.7, seed=42,
                           zero_translation=False):
    """
    Barre un parametro de alineacion delta in [0,1] entre dos bivectores
    generadores (delta=1: mismo generador; delta=0: generadores
    independientes) y mide:
      - E_comm: error medio de composicion ||T1(T2(x)) - T2(T1(x))||
      - lin_commutator: ||R1 R2 - R2 R1||_F  (conmutador puramente rotacional)

    Parametros:
        n: dimension del espacio latente
        n_axes_samples: cuantos valores de delta probar
        n_points: cuantos puntos x se muestrean para promediar el error
        theta: angulo de rotacion (mismo para T1 y T2)
        seed: semilla para reproducibilidad
        zero_translation: si True, fuerza t1=t2=0 para aislar el efecto
                           puramente rotacional (ver sugerencia del .md)
    """
    rng = np.random.default_rng(seed)
    results = []

    B_ref = random_bivector(n, rng)
    B_ref /= np.linalg.norm(B_ref)

    for delta_align in np.linspace(0.0, 1.0, n_axes_samples):
        B_indep = random_bivector(n, rng)
        B_indep /= np.linalg.norm(B_indep)
        B2 = delta_align * B_ref + (1 - delta_align) * B_indep

        R1 = rotor(theta, B_ref)
        R2 = rotor(theta, B2)

        if zero_translation:
            t1 = np.zeros(n)
            t2 = np.zeros(n)
        else:
            t1 = rng.standard_normal(n) * 0.3
            t2 = rng.standard_normal(n) * 0.3

        X = rng.standard_normal((n_points, n))

        T2x = X @ R2.T + t2
        T1_T2x = T2x @ R1.T + t1

        T1x = X @ R1.T + t1
        T2_T1x = T1x @ R2.T + t2

        err = np.linalg.norm(T1_T2x - T2_T1x, axis=1).mean()
        lin_commutator = np.linalg.norm(R1 @ R2 - R2 @ R1, ord='fro')
        results.append((delta_align, err, lin_commutator))

    return results


def print_table(results, titulo):
    print(f"\n{titulo}")
    print(f"{'alineacion':>10} | {'error_composicion':>18} | {'||[R1,R2]||_F':>14}")
    for delta_align, err, lin_commutator in results:
        print(f"{delta_align:10.3f} | {err:18.6f} | {lin_commutator:14.6f}")


if __name__ == "__main__":
    # Caso principal: distorsion afin completa (rotacion + traslacion aleatoria)
    res_full = commutator_experiment(n=8, theta=0.7, seed=42, zero_translation=False)
    print_table(res_full, "Caso 1: distorsion afin completa (rotacion + traslacion)")

    # Caso de control sugerido en el .md: aislar el efecto rotacional puro
    res_notrans = commutator_experiment(n=8, theta=0.7, seed=42, zero_translation=True)
    print_table(res_notrans, "Caso 2: control con traslacion nula (t1=t2=0)")

    print(
        "\nVerificacion esperada: en el Caso 2, con delta=1.0, el error de "
        "composicion deberia caer a (casi) cero, aislando el efecto de la "
        "traslacion observado en el Caso 1."
    )
