"""
POLYDIM - Tarea 21
Verificacion numerica: preservacion geodesica local entre variedades
de curvatura distinta, usando el teorema de Bertrand-Diguet-Puiseux.

Rol: matematico numerico.
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

u, v = sp.symbols('u v')


def build_geodesic_system(X_expr):
    """X_expr: sympy Matrix 3x1, embebido en terminos de (u, v).
    Devuelve funciones numericas: metrica (E, F, G), curvatura de Gauss K,
    y simbolos de Christoffel para la ecuacion geodesica."""
    Xu = X_expr.diff(u)
    Xv = X_expr.diff(v)
    E = sp.simplify(Xu.dot(Xu))
    F = sp.simplify(Xu.dot(Xv))
    G = sp.simplify(Xv.dot(Xv))

    Xuu = Xu.diff(u)
    Xuv = Xu.diff(v)
    Xvv = Xv.diff(v)

    n = Xu.cross(Xv)
    n = n / sp.sqrt(sp.simplify(n.dot(n)))
    L = sp.simplify(Xuu.dot(n))
    M = sp.simplify(Xuv.dot(n))
    N = sp.simplify(Xvv.dot(n))

    K_expr = sp.simplify((L * N - M**2) / (E * G - F**2))

    g = sp.Matrix([[E, F], [F, G]])
    ginv = g.inv()
    coords = [u, v]
    Gamma = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    for k in range(2):
        for i in range(2):
            for j in range(2):
                s = 0
                for l in range(2):
                    s += ginv[k, l] * (sp.diff(g[l, i], coords[j]) +
                                        sp.diff(g[l, j], coords[i]) -
                                        sp.diff(g[i, j], coords[l]))
                Gamma[k][i][j] = sp.simplify(s / 2)

    X_func = sp.lambdify((u, v), X_expr, 'numpy')
    E_func = sp.lambdify((u, v), E, 'numpy')
    F_func = sp.lambdify((u, v), F, 'numpy')
    G_func = sp.lambdify((u, v), G, 'numpy')
    K_func = sp.lambdify((u, v), K_expr, 'numpy')
    Gamma_func = [[[sp.lambdify((u, v), Gamma[k][i][j], 'numpy')
                    for j in range(2)] for i in range(2)] for k in range(2)]

    return dict(X=X_func, E=E_func, F=F_func, G=G_func, K=K_func, Gamma=Gamma_func)


def geodesic_rhs(t, y, funcs):
    uu, vv, du, dv = y
    Gamma = funcs['Gamma']
    dduu = -(Gamma[0][0][0](uu, vv) * du * du +
             2 * Gamma[0][0][1](uu, vv) * du * dv +
             Gamma[0][1][1](uu, vv) * dv * dv)
    ddvv = -(Gamma[1][0][0](uu, vv) * du * du +
             2 * Gamma[1][0][1](uu, vv) * du * dv +
             Gamma[1][1][1](uu, vv) * dv * dv)
    return [du, dv, dduu, ddvv]


def shoot_geodesic(funcs, u0, v0, theta, r_max, n_pts=50):
    E0, F0, G0 = funcs['E'](u0, v0), funcs['F'](u0, v0), funcs['G'](u0, v0)

    def inner(a, b):
        return E0 * a[0] * b[0] + F0 * (a[0] * b[1] + a[1] * b[0]) + G0 * a[1] * b[1]

    e1 = np.array([1.0 / np.sqrt(E0), 0.0])
    e2_raw = np.array([0.0, 1.0])
    e2_raw = e2_raw - inner(e2_raw, e1) * e1
    e2 = e2_raw / np.sqrt(inner(e2_raw, e2_raw))
    d0 = np.cos(theta) * e1 + np.sin(theta) * e2

    y0 = [u0, v0, d0[0], d0[1]]
    sol = solve_ivp(geodesic_rhs, [0, r_max], y0,
                     t_eval=np.linspace(0, r_max, n_pts),
                     args=(funcs,), rtol=1e-9, atol=1e-12)
    return sol


def circle_circumference(funcs, u0, v0, r, n_theta=48):
    pts = []
    for theta in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
        sol = shoot_geodesic(funcs, u0, v0, theta, r, n_pts=30)
        uf, vf = sol.y[0, -1], sol.y[1, -1]
        p = np.array(funcs['X'](uf, vf)).flatten()
        pts.append(p)
    pts.append(pts[0])
    pts = np.array(pts)
    return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))


if __name__ == "__main__":
    # Superficie A: esfera (curvatura constante positiva)
    R = 2.0
    X_sphere = sp.Matrix([R * sp.sin(u) * sp.cos(v),
                           R * sp.sin(u) * sp.sin(v),
                           R * sp.cos(u)])
    funcs_sphere = build_geodesic_system(X_sphere)
    u0_s, v0_s = 1.0, 0.0
    K_sphere = float(funcs_sphere['K'](u0_s, v0_s))

    # Superficie B: paraboloide hiperbolico (curvatura negativa en el origen)
    a = 0.5
    X_para = sp.Matrix([u, v, a * (u**2 - v**2)])
    funcs_para = build_geodesic_system(X_para)
    u0_p, v0_p = 0.0, 0.0
    K_para = float(funcs_para['K'](u0_p, v0_p))

    print("K esfera:", K_sphere, " (esperado 1/R^2 =", 1 / R**2, ")")
    print("K paraboloide en origen:", K_para, " (esperado -4a^2 =", -4 * a**2, ")")

    print("\n r      L_esfera   BDP_esfera   L_parab    BDP_parab   |L_esf-L_par|")
    for r in np.linspace(0.05, 0.5, 6):
        Ls = circle_circumference(funcs_sphere, u0_s, v0_s, r)
        Lp = circle_circumference(funcs_para, u0_p, v0_p, r)
        pred_s = 2 * np.pi * r * (1 - K_sphere * r**2 / 6)
        pred_p = 2 * np.pi * r * (1 - K_para * r**2 / 6)
        print(f"{r:.3f}  {Ls:9.5f}  {pred_s:9.5f}   {Lp:9.5f}  {pred_p:9.5f}   {abs(Ls-Lp):.5f}")
