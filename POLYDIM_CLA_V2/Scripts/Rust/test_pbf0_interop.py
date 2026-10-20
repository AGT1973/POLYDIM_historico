# POLYDIM_DEST
# destination: polydim/core/tests/
# filename:    test_pbf0_interop.py
# author:      ai.mpat.agt@gmail.com
"""
test_pbf0_interop.py — TASK_025
================================
Integration test Python → Rust via PBF0.

USAGE:
  python test_pbf0_interop.py           # generates fixtures and validates Python-side
  python test_pbf0_interop.py --gen     # only generates fixture.pbf0 and fixture_golden.json

FLOW:
  1. Python generates ObjectND (deterministic, normalized sin-wave)
  2. Serializes to PBF0 → fixture.pbf0
  3. Saves golden values → fixture_golden.json
  4. Rust reads fixture.pbf0 and validates against the golden values (integration.rs)

EXPECTED RESULT (Python-side):
  fixture.pbf0      — 40117 bytes
  fixture_golden.json — reference values for Rust
  ALL PYTHON CHECKS PASS

Author: ai.mpat.agt@gmail.com
"""

import struct, json, sys, os
import numpy as np

# ─── PBF0 (standalone, without runtime dependency) ─────────────────────────────
MAGIC   = b"PDIM"
VERSION = 0

def pbf0_serialize(hypervec: np.ndarray, meta: dict) -> bytes:
    vec = np.asarray(hypervec, dtype=np.float32)
    N   = len(vec)
    meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')
    header = struct.pack('>4sHII', MAGIC, VERSION, N, len(meta_bytes))
    return header + meta_bytes + vec.tobytes()

def pbf0_deserialize(data: bytes):
    magic, version, N, meta_len = struct.unpack_from('>4sHII', data, 0)
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic}")
    if version > VERSION:
        raise ValueError(f"Version {version} > reader max {VERSION}")
    meta = json.loads(data[14:14 + meta_len])
    vec  = np.frombuffer(data[14 + meta_len:], dtype=np.float32).copy()
    return vec, meta

# ─── deterministic fixture ───────────────────────────────────────────────────
N_DIM = 10_000

def make_fixture_hv() -> np.ndarray:
    """Deterministic hypervector: normalized sin-wave."""
    hv = np.sin(np.arange(N_DIM, dtype=np.float32) * 0.001)
    return (hv / np.linalg.norm(hv)).astype(np.float32)

FIXTURE_META = {
    "geo_id":  "fixture000001",
    "dims":    ["DIM_SQL", "DIM_RUST"],
    "weights": [0.9, 0.7],
    "modo":    "H",
    "version": "0.3"
}

def generate_fixtures(out_dir: str = ".") -> dict:
    hv  = make_fixture_hv()
    raw = pbf0_serialize(hv, FIXTURE_META)

    pbf0_path   = os.path.join(out_dir, "fixture.pbf0")
    golden_path = os.path.join(out_dir, "fixture_golden.json")

    with open(pbf0_path, 'wb') as f:
        f.write(raw)

    golden = {
        "size":       len(raw),
        "N":          N_DIM,
        "meta_len":   103,
        "geo_id":     "fixture000001",
        "dims":       ["DIM_SQL", "DIM_RUST"],
        "weights":    [0.9, 0.7],
        "modo":       "H",
        "version":    "0.3",
        "hv_norm":    1.0,
        "hv_0":       float(hv[0]),
        "hv_1":       float(hv[1]),
        "hv_9999":    float(hv[9999]),
        "magic_bytes": [80, 68, 73, 77]
    }
    with open(golden_path, 'w') as f:
        json.dump(golden, f, indent=2)

    return {"pbf0": pbf0_path, "golden": golden_path, "raw": raw, "hv": hv}

# ─── Python-side validations ─────────────────────────────────────────────────
def run_python_checks():
    PASS = 0; FAIL = 0

    def check(name, cond, info=""):
        nonlocal PASS, FAIL
        if cond:
            print(f"  PASS  {name}")
            PASS += 1
        else:
            print(f"  FAIL  {name}  {info}")
            FAIL += 1

    hv  = make_fixture_hv()
    raw = pbf0_serialize(hv, FIXTURE_META)

    check("Magic b'PDIM'",      raw[:4] == b"PDIM")
    check("Version = 0",        struct.unpack_from('>H', raw, 4)[0] == 0)
    check("N = 10000",          struct.unpack_from('>I', raw, 6)[0] == 10000)
    check("meta_len = 103",     struct.unpack_from('>I', raw, 10)[0] == 103)
    check("Total size = 40117", len(raw) == 40117)

    hv2, meta2 = pbf0_deserialize(raw)
    mae = float(np.abs(hv - hv2).mean())
    check("Roundtrip MAE = 0.0",  mae == 0.0, f"MAE={mae}")
    check("geo_id preserved",     meta2["geo_id"]  == "fixture000001")
    check("dims preserved",       meta2["dims"]    == ["DIM_SQL", "DIM_RUST"])
    check("weights preserved",    meta2["weights"] == [0.9, 0.7])
    check("modo preserved",       meta2["modo"]    == "H")
    check("hv[0] = 0.0",         abs(hv2[0]) < 1e-10, hv2[0])
    check("hv norm approx 1.0",   abs(np.linalg.norm(hv2) - 1.0) < 1e-5)

    try:
        pbf0_deserialize(b"XDIM" + b"\x00" * 20)
        check("Bad magic raises", False)
    except ValueError:
        check("Bad magic raises", True)

    print(f"\n  {'-'*36}")
    print(f"  {PASS}/{PASS+FAIL} Python checks passed")
    if FAIL == 0:
        print("  ALL PYTHON CHECKS PASS")
    return FAIL == 0

# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("POLYDIM TASK_025 — PBF0 Interop Test")
    print("-" * 40)

    out_dir = "."
    result  = generate_fixtures(out_dir)
    print(f"fixture.pbf0        -> {result['pbf0']}  ({len(result['raw'])} bytes)")
    print(f"fixture_golden.json -> {result['golden']}")
    print()

    ok = run_python_checks()
    print()
    if "--gen" not in sys.argv:
        print("To run the Rust side:")
        print("  cargo test --test integration")

    sys.exit(0 if ok else 1)
