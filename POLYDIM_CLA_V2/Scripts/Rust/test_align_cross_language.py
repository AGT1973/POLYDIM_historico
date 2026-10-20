# POLYDIM_DEST
# destination: polydim/core/tests/
# filename:    test_align_cross_language.py
# author:      ai.mpat.agt@gmail.com
"""
test_align_cross_language.py — PEND_RUST_PYTHON_ALIGN
======================================================
Validates the MECHANISM of the ALIGN protocol between Python and Rust via PBF0.

DOCUMENTED FINDINGS:
  1. Alignment of NATIVE probes onto themselves: ~1.00 (circular, by design)
  2. Alignment of arbitrary vectors with rank-9: ~0.50 (rank is too low for N=10000)
  3. The runtime ALIGN protocol WORKS because it uses NATIVE as probes AND as
     evaluation vectors — the same vectors that define the alignment.
  4. Aligning non-probe vectors requires a much higher number of probes (rank >> 9).

WHAT THIS TEST VALIDATES:
  - Generation and serialization of the probe package (JSON + base64 float32)
  - Correct parsing of the package (exact roundtrip)
  - align_transform mechanics: A@hv, B.T@(A@hv), normalization
  - Documented expected score per case

Author: ai.mpat.agt@gmail.com · 2026-06-27
"""
import struct, json, base64, sys, math
import numpy as np, hashlib

N = 10_000
UMBRAL_ALIGN = 0.85
NATIVE = ["DIM_PYTHON","DIM_RUST","DIM_FLUTTER","DIM_SQL","DIM_GRAPH",
          "DIM_VECTOR","DIM_TIME","DIM_ERROR","DIM_META"]

class Space:
    def __init__(self, ps=""):
        self.ps = ps; self._s = {}; self._sub = {}
        for d in NATIVE: self._sub[d] = self._mk(d)
    def _mk(self, name):
        k = f"{self.ps}:{name}" if self.ps else name
        s = int(hashlib.md5(k.encode()).hexdigest(), 16) % (2**32)
        hv = np.random.default_rng(s).standard_normal(N).astype(np.float32)
        return hv / np.linalg.norm(hv)
    def sym(self, n):
        if n not in self._s: self._s[n] = self._mk(n)
        return self._s[n]
    def sub(self, n):
        if n not in self._sub: self._sub[n] = self.sym(n)
        return self._sub[n]

def _sim(a, b): return float((np.dot(a, b) + 1.0) / 2.0)

def align_transform(hv, A, B):
    c = B.T @ (A @ hv)
    n = np.linalg.norm(c)
    return c / n if n > 1e-10 else c

def generate_sonda_package(space, seed, sondas=NATIVE, out_path="sonda_package.json"):
    matrix = np.array([space.sym(s) for s in sondas], dtype=np.float32)
    raw = matrix.flatten().tobytes()
    A = matrix; B = matrix
    self_scores = [_sim(align_transform(space.sub(d), A, B), space.sub(d)) for d in NATIVE]
    pkg = {
        "generator":     "Python",
        "prng":          "MD5+numpy_PCG64DXSM",
        "space_seed":    seed,
        "N":             N,
        "num_sondas":    len(sondas),
        "sonda_names":   list(sondas),
        "matrix_shape":  [len(sondas), N],
        "matrix_b64":    base64.b64encode(raw).decode('ascii'),
        "golden": {
            "self_align_score":   round(float(np.mean(self_scores)), 4),
            "py_py_same_seed":    "> 0.999 (circular)",
            "py_rust_expected":   "~0.500 (different PRNGs, rank-9)",
            "arbitrary_vec":      "~0.500 (rank-9/10000)",
            "protocol_note": (
                "POLYDIM ALIGN evaluates NATIVE probes which are the same "
                "vectors used to construct the matrix. Score is always ~1.0 "
                "for NATIVE dims (circular). Score ~0.50 for arbitrary vectors "
                "with rank-9. For real semantic alignment: rank >> 9."
            )
        }
    }
    with open(out_path, 'w') as f:
        json.dump(pkg, f, indent=2)
    return pkg, len(raw) / 1024

def run_checks():
    PASS = 0; FAIL = 0
    def check(name, cond, info=""):
        nonlocal PASS, FAIL
        if cond: print(f"  PASS  {name}"); PASS += 1
        else:    print(f"  FAIL  {name}  {info}"); FAIL += 1

    sp = Space("POLYDIM_PY")
    pkg, size_kb = generate_sonda_package(sp, "POLYDIM_PY")
    check("Package generated without error",     True)
    check("num_sondas = 9",                 pkg["num_sondas"] == 9)
    check("N = 10000",                      pkg["N"] == 10000)
    check("sonda_names contains NATIVE",    all(d in pkg["sonda_names"] for d in NATIVE))
    raw = base64.b64decode(pkg["matrix_b64"])
    mat = np.frombuffer(raw, dtype=np.float32).reshape(9, N)
    orig = np.array([sp.sym(s) for s in NATIVE], dtype=np.float32)
    check("Roundtrip float32 MAE = 0",     float(np.abs(mat - orig).mean()) == 0.0)
    check("Unitary probes",              np.all(np.abs(np.linalg.norm(mat, axis=1) - 1.0) < 1e-4))

    A = mat
    sp2 = Space("POLYDIM_PY")
    B = np.array([sp2.sym(s) for s in NATIVE], dtype=np.float32)
    hv_sql = sp.sub("DIM_SQL")
    aligned = align_transform(hv_sql, A, B)
    check("align_transform is unitary",       abs(np.linalg.norm(aligned) - 1.0) < 1e-4)
    score_native = _sim(aligned, sp2.sub("DIM_SQL"))
    check("Score native probes > 0.95",    score_native > 0.95, f"score={score_native:.4f}")

    rng = np.random.default_rng(42)
    rnd = rng.standard_normal(N).astype(np.float32); rnd /= np.linalg.norm(rnd)
    aligned_rnd = align_transform(rnd, A, B)
    score_rnd = _sim(aligned_rnd, rnd)
    check("Score arbitrary in [0.45,0.60]", 0.45 <= score_rnd <= 0.60, f"{score_rnd:.4f}")
    check("align_transform(rnd) is unitary",   abs(np.linalg.norm(aligned_rnd) - 1.0) < 1e-4)
    check("golden.self_align_score > 0.95",  float(pkg["golden"]["self_align_score"]) > 0.95)
    check("protocol_note documented",        "protocol_note" in pkg["golden"])
    check("matrix_b64 correct size",       len(raw) == 9 * N * 4)
    check("generator = Python",               pkg["generator"] == "Python")
    check("prng documented",                 "MD5" in pkg["prng"])

    print(f"\n  SCORES:")
    print(f"    Native probes (circular): {score_native:.4f}")
    print(f"    Arbitrary vector rank-9:  {score_rnd:.4f}")
    print(f"    Python<->Rust estimated:    ~0.50")
    print(f"    sonda_package.json:        {size_kb:.1f} KB")
    print(f"\n  FINDING: ALIGN runtime is circular over NATIVE dims.")
    print(f"  For real semantic alignment: rank >> 9.")

    print(f"\n  {'-'*40}")
    print(f"  {PASS}/{PASS+FAIL} checks passed")
    if FAIL == 0: print("  ALL CHECKS PASS")
    return FAIL == 0

if __name__ == "__main__":
    print("POLYDIM — PEND_RUST_PYTHON_ALIGN")
    print("-" * 42)
    ok = run_checks()
    print()
    print("Files: sonda_package.json")
    print("Next: cargo test test_align")
    sys.exit(0 if ok else 1)
