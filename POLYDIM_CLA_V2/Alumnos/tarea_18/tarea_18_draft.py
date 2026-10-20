import numpy as np

RNG = np.random.default_rng(18)


def grade_of(bitmask):
    return bin(bitmask).count("1")


def _blade_sign(a, b):
    a >>= 1
    total = 0
    while a:
        total += bin(a & b).count("1")
        a >>= 1
    return -1.0 if (total % 2) else 1.0


class MV:
    __slots__ = ("n", "c")

    def __init__(self, n, c=None):
        self.n = n
        self.c = {k: v for k, v in (c or {}).items() if abs(v) > 1e-13}

    @staticmethod
    def scalar(n, value=1.0):
        return MV(n, {0: value})

    @staticmethod
    def from_vector(x):
        return MV(len(x), {1 << i: float(xi) for i, xi in enumerate(x)})

    def to_vector(self):
        x = np.zeros(self.n)
        leak = 0.0
        for bm, v in self.c.items():
            if grade_of(bm) == 1:
                x[bm.bit_length() - 1] = v
            else:
                leak += v * v
        return x, np.sqrt(leak)

    def grade(self, k):
        return MV(self.n, {bm: v for bm, v in self.c.items() if grade_of(bm) == k})

    def energy_by_grade(self):
        out = {}
        for bm, v in self.c.items():
            g = grade_of(bm)
            out[g] = out.get(g, 0.0) + v * v
        return out

    def __add__(self, other):
        c = dict(self.c)
        for bm, v in other.c.items():
            c[bm] = c.get(bm, 0.0) + v
        return MV(self.n, c)

    def __neg__(self):
        return MV(self.n, {k: -v for k, v in self.c.items()})

    def __sub__(self, other):
        return self + (-other)

    def scale(self, s):
        return MV(self.n, {k: v * s for k, v in self.c.items()})

    def __mul__(self, other):
        out = {}
        for ba, ca in self.c.items():
            for bb, cb in other.c.items():
                s = _blade_sign(ba, bb)
                bm = ba ^ bb
                out[bm] = out.get(bm, 0.0) + s * ca * cb
        return MV(self.n, out)

    def reverse(self):
        c = {}
        for bm, v in self.c.items():
            k = grade_of(bm)
            s = -1.0 if ((k * (k - 1) // 2) % 2) else 1.0
            c[bm] = s * v
        return MV(self.n, c)

    def scalar_part(self):
        return self.c.get(0, 0.0)

    def norm(self):
        return np.sqrt(max((self * self.reverse()).scalar_part(), 0.0))


def wedge(u, v):
    return (MV.from_vector(u) * MV.from_vector(v)).grade(2)


def simple_rotor(u, v, theta):
    n = len(u)
    B = wedge(u, v)
    mag = B.norm()
    if mag < 1e-12:
        return MV.scalar(n, 1.0)
    Bhat = B.scale(1.0 / mag)
    return MV.scalar(n, np.cos(theta / 2)) + Bhat.scale(np.sin(theta / 2))


def apply_rotor(R, x):
    y = R * MV.from_vector(x) * R.reverse()
    return y.to_vector()


def skew_from_bivector(B, n):
    Q = np.zeros((n, n))
    for bm, coef in B.c.items():
        idx = [i for i in range(n) if (bm >> i) & 1]
        if len(idx) == 2:
            i, j = idx
            Q[i, j] += coef
            Q[j, i] -= coef
    return Q


def expm_skew(Q, terms=60):
    n = Q.shape[0]
    total = np.eye(n)
    term = np.eye(n)
    for k in range(1, terms):
        term = term @ Q / k
        total = total + term
    return total


def rotor_matrix(R, n):
    cols = []
    for i in range(n):
        ei = np.eye(n)[i]
        yi, _ = apply_rotor(R, ei)
        cols.append(yi)
    return np.stack(cols, axis=1)


# ---- pruebas de verificacion algebraica ----

print("### Test 1: e2e3 * e1e2 (comparten eje) -> debe ser grado 2 puro")
e1e2 = MV(4, {0b0011: 1.0})
e2e3 = MV(4, {0b0110: 1.0})
prod = e2e3 * e1e2
print(" resultado:", prod.c, " grados presentes:", set(grade_of(k) for k in prod.c))

print("\n### Test 2: e1e2 * e3e4 (ejes disjuntos) -> debe ser grado 4 puro")
e3e4 = MV(4, {0b1100: 1.0})
prod2 = e1e2 * e3e4
print(" resultado:", prod2.c, " grados presentes:", set(grade_of(k) for k in prod2.c))

print("\n### Test 3: rotor individual, unitariedad y preservacion de norma")
n = 6
x = RNG.normal(size=n)
x = x / np.linalg.norm(x)
u, v = RNG.normal(size=n), RNG.normal(size=n)
theta = 0.77
R = simple_rotor(u, v, theta)
RRt = (R * R.reverse()).scalar_part()
y, leak = apply_rotor(R, x)
print(f" R*~R = {RRt:.12f}  (esperado 1.0)")
print(f" fuga fuera de grado1 en la salida = {leak:.3e}")
print(f" ||x|| = {np.linalg.norm(x):.12f}   ||RxR~|| = {np.linalg.norm(y):.12f}")

print("\n### Test 4: equivalencia con SO(n) via exponencial de matriz antisimetrica")
B = wedge(u, v)
Bhat = B.scale(1.0 / B.norm())
Q = skew_from_bivector(Bhat.scale(theta), n)
Qexp = expm_skew(Q)
y_mat = Qexp @ x
print(f" max|y_rotor - y_matriz| = {np.max(np.abs(y - y_mat)):.3e}")
print(f" det(Qexp) = {np.linalg.det(Qexp):.6f} (debe ser 1 -> SO(n), no O(n))")
print(f" max|Qexp^T Qexp - I| = {np.max(np.abs(Qexp.T @ Qexp - np.eye(n))):.3e}")
