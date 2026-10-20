"""
POLYDIM Swarm - Test 9 (Alumno 9)
Escuadron 1: Relajacion Topologica y Homotopia (TDA)

Rol: DevOps y SRE de IA
Afirmacion a validar: "ignorar homotopias superiores ahorra computo sin
perjudicar la generalizacion composicional"

Metodologia:
  - Algebra de base: OCTONIONES (R^8), algebra real no-asociativa, alternativa,
    normada. Su "asociador" a(x,y,z) = (xy)z - x(yz) es no nulo en general:
    es un dato geometrico real de "homotopia de orden 1" (testigo de que la
    asociatividad falla, y de cuanto falla).
  - Modelo A ("ignora homotopias superiores"): un MLP g(x,y) ~ x*y entrenado
    SOLO con pares (x,y)->z. Nunca ve una terna (x,y,z) evaluada como cadena.
  - Modelo B ("respeta homotopia de orden 1"): el mismo MLP, entrenado con
    pares + ternas (le mostramos (xy)z y x(yz) explicitamente). Esto es
    literalmente forzarlo a aprender el asociador -> aprende la "homotopia".
  - Ninguno de los dos ve nunca una 4-upla ni 5-upla completa durante el
    entrenamiento. Evaluamos generalizacion composicional en cadenas de
    longitud 4 y 5 (fuera de distribucion de entrenamiento) aplicando el
    operador aprendido de forma recursiva (asociando a izquierda).
  - Medimos: (1) costo de computo de entrenamiento (pasos x tamanio de
    batch, proxy de FLOPs), (2) error de generalizacion composicional
    (error relativo L2) en longitud 4 y 5.

Si Modelo A generaliza igual de bien que B con menos computo -> la
afirmacion del Test 9 se sostiene.
Si Modelo A degrada fuertemente frente a B en cadenas largas -> la
afirmacion se refuta (Falla Matematica), tal como exige la Piramide de
Validacion del documento madre (no alucinar exito).
"""
import numpy as np

rng = np.random.default_rng(7)

# ---------------------------------------------------------------------
# 1. Algebra de octoniones via construccion de Cayley-Dickson
#    (par de cuaterniones). Es EXACTA, no aproximada: sirve de terreno
#    de verdad matematica (ground truth) contra el cual medir al modelo.
# ---------------------------------------------------------------------
def quat_mul(p, q):
    # p, q: (...,4) -> Hamilton product, vectorizado
    a1, b1, c1, d1 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    a2, b2, c2, d2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w = a1*a2 - b1*b2 - c1*c2 - d1*d2
    x = a1*b2 + b1*a2 + c1*d2 - d1*c2
    y = a1*c2 - b1*d2 + c1*a2 + d1*b2
    z = a1*d2 + b1*c2 - c1*b2 + d1*a2
    return np.stack([w, x, y, z], axis=-1)

def quat_conj(p):
    return p * np.array([1, -1, -1, -1])

def oct_mul(u, v):
    # u, v: (...,8) octonion = (a,b) con a,b cuaterniones en R^4
    a, b = u[..., :4], u[..., 4:]
    c, d = v[..., :4], v[..., 4:]
    # (a,b)(c,d) = (ac - d~ b, d a + b c~)   [Cayley-Dickson]
    real_part = quat_mul(a, c) - quat_mul(quat_conj(d), b)
    imag_part = quat_mul(d, a) + quat_mul(b, quat_conj(c))
    return np.concatenate([real_part, imag_part], axis=-1)

def sample_octonions(n):
    v = rng.normal(size=(n, 8))
    v /= np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
    return v

def associator_norm(x, y, z):
    left = oct_mul(oct_mul(x, y), z)
    right = oct_mul(x, oct_mul(y, z))
    return np.linalg.norm(left - right, axis=-1)

# Sanity check: el asociador de octoniones es genuinamente no nulo.
x_chk, y_chk, z_chk = sample_octonions(2000), sample_octonions(2000), sample_octonions(2000)
assoc_stat = associator_norm(x_chk, y_chk, z_chk)
print(f"[Sanity check] ||[x,y,z]_asociador|| medio = {assoc_stat.mean():.4f} "
      f"(std={assoc_stat.std():.4f}); si fuera asociativa daria ~0.")


# ---------------------------------------------------------------------
# 2. MLP minimo con backprop manual (sin frameworks pesados: el entorno
#    no tiene espacio en disco para instalar torch/jax hoy, asi que
#    documentamos la falla de infraestructura -segun exige la Piramide
#    de Validacion: reportar fallas reales, no maquillarlas- y resolvemos
#    el experimento en numpy puro, que es matematicamente equivalente).
# ---------------------------------------------------------------------
class MLP:
    def __init__(self, in_dim, hidden, out_dim, seed):
        r = np.random.default_rng(seed)
        s = lambda a, b: r.normal(0, np.sqrt(2.0 / a), size=(a, b))
        self.W1 = s(in_dim, hidden);  self.b1 = np.zeros(hidden)
        self.W2 = s(hidden, hidden);  self.b2 = np.zeros(hidden)
        self.W3 = s(hidden, out_dim); self.b3 = np.zeros(out_dim)
        self.params = ["W1","b1","W2","b2","W3","b3"]
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

    def forward(self, x, cache=False):
        z1 = x @ self.W1 + self.b1; a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2; a2 = np.tanh(z2)
        out = a2 @ self.W3 + self.b3
        if cache:
            return out, (x, z1, a1, z2, a2)
        return out

    def backward(self, cache, dout):
        x, z1, a1, z2, a2 = cache
        n = x.shape[0]
        dW3 = a2.T @ dout / n
        db3 = dout.mean(axis=0)
        da2 = dout @ self.W3.T
        dz2 = da2 * (1 - a2**2)
        dW2 = a1.T @ dz2 / n
        db2 = dz2.mean(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - a1**2)
        dW1 = x.T @ dz1 / n
        db1 = dz1.mean(axis=0)
        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)

    def step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            g = grads[p]
            self.m[p] = beta1*self.m[p] + (1-beta1)*g
            self.v[p] = beta2*self.v[p] + (1-beta2)*(g*g)
            mhat = self.m[p] / (1 - beta1**self.t)
            vhat = self.v[p] / (1 - beta2**self.t)
            setattr(self, p, getattr(self, p) - lr * mhat / (np.sqrt(vhat) + eps))

    def n_params(self):
        return sum(getattr(self, p).size for p in self.params)


def g_apply(model, x, y):
    """Aplica el operador binario aprendido g(x,y)."""
    inp = np.concatenate([x, y], axis=-1)
    return model.forward(inp)


def train_model(use_triples, steps, batch, lr=2e-3, seed=0):
    model = MLP(16, 64, 8, seed=seed)
    flops = 0  # proxy de computo: (elementos de entrada) x (pasos de red) por paso
    fwd_cost_pair = batch * 16 * 64 * 2  # aprox. MACs de la 1ra capa como proxy
    for step in range(steps):
        x = sample_octonions(batch); y = sample_octonions(batch)
        z_true = oct_mul(x, y)
        inp = np.concatenate([x, y], axis=-1)
        out, cache = model.forward(inp, cache=True)
        dout = 2 * (out - z_true) / z_true.shape[-1]
        grads = model.backward(cache, dout)
        flops += fwd_cost_pair

        if use_triples:
            xt, yt, zt = sample_octonions(batch), sample_octonions(batch), sample_octonions(batch)
            left_true = oct_mul(oct_mul(xt, yt), zt)
            right_true = oct_mul(xt, oct_mul(yt, zt))

            xy = g_apply(model, xt, yt)
            left_pred, cache_l2 = model.forward(np.concatenate([xy, zt], axis=-1), cache=True)
            dout_l = 2 * (left_pred - left_true) / 8
            grads_l = model.backward(cache_l2, dout_l)

            yz = g_apply(model, yt, zt)
            right_pred, cache_r2 = model.forward(np.concatenate([xt, yz], axis=-1), cache=True)
            dout_r = 2 * (right_pred - right_true) / 8
            grads_r = model.backward(cache_r2, dout_r)

            for p in model.params:
                grads[p] = grads[p] + 0.5*grads_l[p] + 0.5*grads_r[p]
            flops += 2 * fwd_cost_pair

        model.step(grads, lr=lr)
    return model, flops


def chain_error(model, length, n=500):
    """Genera cadenas de longitud `length`, compara aplicacion recursiva de
    g (asociando a izquierda) contra el producto real de octoniones
    (misma asociacion a izquierda)."""
    elems = [sample_octonions(n) for _ in range(length)]
    true_val = elems[0]
    for e in elems[1:]:
        true_val = oct_mul(true_val, e)
    pred_val = elems[0]
    for e in elems[1:]:
        pred_val = g_apply(model, pred_val, e)
    rel_err = np.linalg.norm(pred_val - true_val, axis=-1) / (np.linalg.norm(true_val, axis=-1) + 1e-9)
    return rel_err.mean(), rel_err.std()


# ---------------------------------------------------------------------
# 3. Experimento principal
# ---------------------------------------------------------------------
import time, json

STEPS = 1500
BATCH = 64

results = {}
for name, use_triples in [("A_ignora_homotopia", False), ("B_respeta_homotopia_orden1", True)]:
    t0 = time.time()
    model, flops = train_model(use_triples=use_triples, steps=STEPS, batch=BATCH, seed=3)
    wall = time.time() - t0

    pair_err, _ = chain_error(model, 2, n=1000)   # visto en entrenamiento (control)
    tri_err, _ = chain_error(model, 3, n=1000)    # B lo vio explicito; A no
    len4_mean, len4_std = chain_error(model, 4, n=1000)  # NINGUNO lo vio (homotopia orden 2)
    len5_mean, len5_std = chain_error(model, 5, n=1000)  # NINGUNO lo vio (homotopia orden 3)

    results[name] = dict(
        wall_clock_seg=round(wall, 3),
        flops_proxy=int(flops),
        n_parametros=model.n_params(),
        error_relativo_par_len2=round(float(pair_err), 4),
        error_relativo_terna_len3=round(float(tri_err), 4),
        error_relativo_cadena_len4=round(float(len4_mean), 4),
        std_cadena_len4=round(float(len4_std), 4),
        error_relativo_cadena_len5=round(float(len5_mean), 4),
        std_cadena_len5=round(float(len5_std), 4),
    )
    print(name, json.dumps(results[name], indent=2, ensure_ascii=False))

ahorro_computo_pct = 100 * (1 - results["A_ignora_homotopia"]["flops_proxy"] /
                             results["B_respeta_homotopia_orden1"]["flops_proxy"])
degradacion_len4_pct = 100 * (results["A_ignora_homotopia"]["error_relativo_cadena_len4"] /
                               max(results["B_respeta_homotopia_orden1"]["error_relativo_cadena_len4"], 1e-9) - 1)
degradacion_len5_pct = 100 * (results["A_ignora_homotopia"]["error_relativo_cadena_len5"] /
                               max(results["B_respeta_homotopia_orden1"]["error_relativo_cadena_len5"], 1e-9) - 1)

print("\n=== VEREDICTO EMPIRICO (Piramide de Validacion) ===")
print(f"Ahorro de computo de A sobre B: {ahorro_computo_pct:.1f}%")
print(f"Degradacion relativa de error A vs B en cadenas len=4: {degradacion_len4_pct:+.1f}%")
print(f"Degradacion relativa de error A vs B en cadenas len=5: {degradacion_len5_pct:+.1f}%")

with open("resultados.json", "w") as f:
    json.dump(dict(
        steps=STEPS, batch=BATCH,
        resultados=results,
        ahorro_computo_pct=round(ahorro_computo_pct, 2),
        degradacion_len4_pct=round(degradacion_len4_pct, 2),
        degradacion_len5_pct=round(degradacion_len5_pct, 2),
    ), f, indent=2, ensure_ascii=False)
print("\nGuardado resultados.json")
