// POLYDIM_DEST
// destination: polydim/core/
// filename:    polydim_core.rs
// author:      ai.mpat.agt@gmail.com
//
// polydim_core.rs — POLYDIM Core · Rust prototype · V0.1
// =======================================================
// Implements: Space, ObjectND, _bind, _sup, _proj, PBF0 serialization
// Resolves:   TASK_023
// Depends on: TASK_021 (PBF0 format, now in SPEC_FORMATO_BINARIO_V0.md)
//
// CARGO.TOML dependencies:
//   [dependencies]
//   serde       = { version = "1", features = ["derive"] }
//   serde_json  = "1"
//   md-5        = "0.10"          # for cross-language seed compatibility
//
// CROSS-LANGUAGE NOTE:
//   Rust Space uses the same MD5-seeded deterministic generation as Python.
//   However, the PRNG (PCG64 vs numpy PCG64DXSM) produces DIFFERENT normal
//   vectors from the same seed. Cross-language compatibility relies on ALIGN.
//   PBF0 format (TASK_021) enables full interop without re-computation.
//
// Author:   ai.mpat.agt@gmail.com
// Version:  V0.1 · 2026-06-23

use std::collections::HashMap;

pub mod polydim_fft_bind;

// ─── constants ────────────────────────────────────────────────────────────────
pub const N: usize = 10_000;
pub const UMBRAL: f32 = 0.510;             // 0.5 + 2*(1/(2*sqrt(10000)))
pub const CONTENT_W: f32 = 0.3;
pub const UMBRAL_ALIGN: f32 = 0.85;
pub const PBF0_MAGIC: &[u8; 4] = b"PDIM";
pub const PBF0_VERSION: u16 = 0;

pub const NATIVE_DIMS: &[&str] = &[
    "DIM_PYTHON", "DIM_RUST", "DIM_FLUTTER", "DIM_SQL",
    "DIM_GRAPH", "DIM_VECTOR", "DIM_TIME", "DIM_ERROR", "DIM_META",
];

// ─── PRNG (PCG-XSH-RR 64→32) ─────────────────────────────────────────────────
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(seed: u64) -> Self {
        let mut rng = Pcg32 { state: 0, inc: (seed << 1) | 1 };
        rng.state = rng.state.wrapping_add(rng.inc);
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005)
                        .wrapping_add(self.inc | 1);
        let xorshifted = ((old >> 18) ^ old) >> 27;
        let rot = (old >> 59) as u32;
        (xorshifted as u32).rotate_right(rot)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        let mag = (-2.0_f32 * u1.ln()).sqrt();
        mag * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

fn seed_from_name(name: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for byte in name.as_bytes() {
        h ^= *byte as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn make_hv(name: &str) -> Vec<f32> {
    let seed = seed_from_name(name);
    let mut rng = Pcg32::new(seed);
    let mut hv: Vec<f32> = (0..N).map(|_| rng.next_normal()).collect();
    normalize_inplace(&mut hv);
    hv
}

// ─── VSA core operations ──────────────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

fn normalize_inplace(v: &mut Vec<f32>) {
    let n = norm(v);
    if n > 1e-10 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

fn normalize(v: Vec<f32>) -> Vec<f32> {
    let mut v = v;
    normalize_inplace(&mut v);
    v
}

pub fn bind(a: &[f32], b: &[f32]) -> Vec<f32> {
    let r: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
    normalize(r)
}

pub fn sup(hvs: &[Vec<f32>], ws: &[f32]) -> Vec<f32> {
    assert!(!hvs.is_empty());
    assert_eq!(hvs.len(), ws.len());
    let mut result = vec![0.0f32; N];
    for (hv, &w) in hvs.iter().zip(ws.iter()) {
        for (r, &h) in result.iter_mut().zip(hv.iter()) {
            *r += w * h;
        }
    }
    normalize(result)
}

pub fn proj(hv: &[f32], sub: &[f32]) -> f32 {
    (dot(hv, sub) + 1.0) / 2.0
}

pub fn sim(a: &[f32], b: &[f32]) -> f32 {
    proj(a, b)
}

// ─── Space ────────────────────────────────────────────────────────────────────

pub struct Space {
    pub personal_seed: String,
    symbols:    HashMap<String, Vec<f32>>,
    subspaces:  HashMap<String, Vec<f32>>,
}

impl Space {
    pub fn new(personal_seed: &str) -> Self {
        let mut sp = Space {
            personal_seed: personal_seed.to_string(),
            symbols:   HashMap::new(),
            subspaces: HashMap::new(),
        };
        for &d in NATIVE_DIMS {
            let hv = sp._mk(d);
            sp.subspaces.entry(d.to_string()).or_insert(hv);
        }
        sp
    }

    fn _mk(&self, name: &str) -> Vec<f32> {
        let key = if self.personal_seed.is_empty() {
            name.to_string()
        } else {
            format!("{}:{}", self.personal_seed, name)
        };
        make_hv(&key)
    }

    pub fn sym(&mut self, name: &str) -> &Vec<f32> {
        if !self.symbols.contains_key(name) {
            let hv = self._mk(name);
            self.symbols.insert(name.to_string(), hv);
        }
        self.symbols.get(name).unwrap()
    }

    pub fn sub(&mut self, name: &str) -> &Vec<f32> {
        if !self.subspaces.contains_key(name) {
            let hv = self.sym(name).clone();
            self.subspaces.insert(name.to_string(), hv);
        }
        self.subspaces.get(name).unwrap()
    }

    pub fn enc(&mut self, props: &HashMap<String, String>) -> Vec<f32> {
        if props.is_empty() {
            return self.sym("__empty__").clone();
        }
        let pairs: Vec<Vec<f32>> = props.iter()
            .map(|(k, v)| {
                let sk = self.sym(k).clone();
                let sv = self.sym(v).clone();
                bind(&sk, &sv)
            })
            .collect();
        let ws = vec![1.0f32; pairs.len()];
        sup(&pairs, &ws)
    }

    pub fn rnd() -> Vec<f32> {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(42);
        let mut rng = Pcg32::new(seed ^ 0xdeadbeefcafe1234);
        let mut hv: Vec<f32> = (0..N).map(|_| rng.next_normal()).collect();
        normalize_inplace(&mut hv);
        hv
    }
}

// ─── ObjectND ─────────────────────────────────────────────────────────────────

pub struct ObjectND<'a> {
    pub space: &'a mut Space,
    pub geo:   Vec<f32>,
    props: HashMap<String, HashMap<String, String>>,
    weights: HashMap<String, f32>,
    cache: Option<Vec<f32>>,
}

impl<'a> ObjectND<'a> {
    pub fn new(space: &'a mut Space) -> Self {
        let geo = Space::rnd();
        ObjectND {
            space,
            geo,
            props: HashMap::new(),
            weights: HashMap::new(),
            cache: None,
        }
    }

    pub fn add(
        &mut self,
        dim: &str,
        props: HashMap<String, String>,
        w: f32,
    ) -> &mut Self {
        self.props.insert(dim.to_string(), props);
        self.weights.insert(dim.to_string(), w.clamp(0.0, 1.0));
        self.cache = None;
        self
    }

    pub fn hv(&mut self) -> Vec<f32> {
        if let Some(ref c) = self.cache {
            return c.clone();
        }
        let mut hvs: Vec<Vec<f32>> = vec![self.geo.clone()];
        let mut ws: Vec<f32> = vec![1.0];

        let dims: Vec<String> = self.props.keys().cloned().collect();
        for dim in &dims {
            let ww = *self.weights.get(dim.as_str()).unwrap_or(&1.0);
            if ww <= 0.0 { continue; }
            let props = self.props.get(dim.as_str()).unwrap().clone();

            let sub = self.space.sub(dim.as_str()).clone();
            hvs.push(sub.clone());
            ws.push(ww);

            let enc = self.space.enc(&props);
            let content = bind(&sub, &enc);
            hvs.push(content);
            ws.push(ww * CONTENT_W);
        }

        let result = sup(&hvs, &ws);
        self.cache = Some(result.clone());
        result
    }

    pub fn activacion(&mut self, dim: &str) -> f32 {
        let hv = self.hv();
        let sub = self.space.sub(dim).clone();
        proj(&hv, &sub)
    }

    pub fn dims_activas(&mut self) -> HashMap<String, f32> {
        let mut result = HashMap::new();
        let declared: Vec<String> = self.props.keys().cloned().collect();
        let native: Vec<String> = NATIVE_DIMS.iter().map(|s| s.to_string()).collect();
        let all: Vec<String> = {
            let mut v = native;
            for d in &declared { if !v.contains(d) { v.push(d.clone()); } }
            v
        };
        for d in all {
            let act = self.activacion(&d);
            if act > UMBRAL {
                result.insert(d, (act * 10000.0).round() / 10000.0);
            }
        }
        result
    }

    pub fn geo_id(&self) -> String {
        let mut h: u64 = 14695981039346656037;
        for byte in self.geo.iter().flat_map(|f| f.to_le_bytes()) {
            h ^= byte as u64;
            h = h.wrapping_mul(1099511628211);
        }
        format!("{:016x}", h)[..12].to_string()
    }

    pub fn to_symbolic(&self) -> serde_json::Value {
        use serde_json::{json, Map, Value};
        let mut dims_map = Map::new();
        for (d, p) in &self.props {
            let props_val: Map<String, Value> = p.iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect();
            dims_map.insert(d.clone(), json!({
                "w": self.weights.get(d.as_str()).copied().unwrap_or(1.0),
                "props": props_val
            }));
        }
        json!({ "geo_id": self.geo_id(), "dims": dims_map })
    }
}

// ─── PBF0 ─────────────────────────────────────────────────────────────────────

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Pbf0Meta {
    pub geo_id:  String,
    pub dims:    Vec<String>,
    pub weights: Vec<f32>,
    pub modo:    String,
    pub version: String,
}

pub struct Pbf0 {
    pub hypervec: Vec<f32>,
    pub meta:     Pbf0Meta,
}

impl Pbf0 {
    pub fn serialize(hypervec: &[f32], meta: &Pbf0Meta) -> Result<Vec<u8>, String> {
        let vec_f32: Vec<f32> = hypervec.iter().map(|&x| x as f32).collect();
        let n = vec_f32.len() as u32;
        let meta_json = serde_json::to_string(meta)
            .map_err(|e| format!("JSON error: {}", e))?;
        let meta_bytes = meta_json.as_bytes();
        let meta_len = meta_bytes.len() as u32;

        let mut buf = Vec::with_capacity(14 + meta_bytes.len() + vec_f32.len() * 4);
        buf.extend_from_slice(PBF0_MAGIC);
        buf.extend_from_slice(&PBF0_VERSION.to_be_bytes());
        buf.extend_from_slice(&n.to_be_bytes());
        buf.extend_from_slice(&meta_len.to_be_bytes());
        buf.extend_from_slice(meta_bytes);
        for &f in &vec_f32 {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        Ok(buf)
    }

    pub fn deserialize(data: &[u8]) -> Result<Pbf0, String> {
        if data.len() < 14 {
            return Err("PBF0: too short".to_string());
        }
        if &data[0..4] != PBF0_MAGIC {
            return Err(format!("PBF0: invalid magic {:?}", &data[0..4]));
        }
        let version = u16::from_be_bytes([data[4], data[5]]);
        if version > PBF0_VERSION {
            return Err(format!("PBF0: version {} > reader max {}", version, PBF0_VERSION));
        }
        let n        = u32::from_be_bytes([data[6], data[7], data[8], data[9]]) as usize;
        let meta_len = u32::from_be_bytes([data[10], data[11], data[12], data[13]]) as usize;
        let offset   = 14;

        if data.len() < offset + meta_len + n * 4 {
            return Err("PBF0: buffer too short".to_string());
        }

        let meta_str = std::str::from_utf8(&data[offset..offset + meta_len])
            .map_err(|e| format!("PBF0: UTF-8 error: {}", e))?;
        let meta: Pbf0Meta = serde_json::from_str(meta_str)
            .map_err(|e| format!("PBF0: JSON error: {}", e))?;

        let vec_start = offset + meta_len;
        let hypervec: Vec<f32> = data[vec_start..vec_start + n * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(Pbf0 { hypervec, meta })
    }

    pub fn to_hv(&self) -> &Vec<f32> {
        &self.hypervec
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_hv_unit_norm() {
        let hv = make_hv("DIM_SQL");
        let n = norm(&hv);
        assert!((n - 1.0).abs() < 1e-4, "norm={}", n);
    }

    #[test]
    fn test_make_hv_deterministic() {
        let a = make_hv("DIM_RUST");
        let b = make_hv("DIM_RUST");
        let d: f32 = a.iter().zip(b.iter()).map(|(x,y)| (x-y).abs()).sum();
        assert!(d < 1e-6, "not deterministic");
    }

    #[test]
    fn test_make_hv_distinct() {
        let a = make_hv("DIM_SQL");
        let b = make_hv("DIM_PYTHON");
        assert!(sim(&a, &b) < 0.6);
    }

    #[test]
    fn test_bind_unit_norm() {
        let a = make_hv("DIM_SQL");
        let b = make_hv("DIM_PYTHON");
        let c = bind(&a, &b);
        assert!((norm(&c) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_sup_unit_norm() {
        let a = make_hv("DIM_SQL");
        let b = make_hv("DIM_PYTHON");
        let c = sup(&[a, b], &[1.0, 1.0]);
        assert!((norm(&c) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_proj_range() {
        let a = make_hv("DIM_SQL");
        let b = make_hv("DIM_PYTHON");
        let p = proj(&a, &b);
        assert!(p >= 0.0 && p <= 1.0, "proj={}", p);
    }

    #[test]
    fn test_space_creates_native_dims() {
        let mut sp = Space::new("TEST");
        let sub = sp.sub("DIM_RUST");
        assert_eq!(sub.len(), N);
    }

    #[test]
    fn test_object_add_activacion() {
        let mut sp = Space::new("TEST_OBJ");
        let mut obj = ObjectND::new(&mut sp);
        let mut props = HashMap::new();
        props.insert("table".to_string(), "users".to_string());
        obj.add("DIM_SQL", props, 1.0);
        let act = obj.activacion("DIM_SQL");
        assert!(act > UMBRAL, "act={} UMBRAL={}", act, UMBRAL);
    }

    #[test]
    fn test_geo_id_length() {
        let mut sp = Space::new("GEO_TEST");
        let obj = ObjectND::new(&mut sp);
        assert_eq!(obj.geo_id().len(), 12);
    }

    #[test]
    fn test_pbf0_roundtrip() {
        let hv: Vec<f32> = (0..N).map(|i| (i as f32).sin()).collect();
        let meta = Pbf0Meta {
            geo_id:  "abc123def456".to_string(),
            dims:    vec!["DIM_SQL".to_string(), "DIM_RUST".to_string()],
            weights: vec![0.8, 0.6],
            modo:    "H".to_string(),
            version: "0.3".to_string(),
        };
        let bytes = Pbf0::serialize(&hv, &meta).unwrap();
        let dec   = Pbf0::deserialize(&bytes).unwrap();
        assert_eq!(dec.meta.geo_id, "abc123def456");
        assert_eq!(dec.hypervec.len(), N);
        let err: f32 = hv.iter().zip(dec.hypervec.iter())
            .map(|(a,b)| (a-b).abs()).sum::<f32>() / N as f32;
        assert!(err < 1e-6, "MAE={}", err);
    }

    #[test]
    fn test_pbf0_magic_validation() {
        let bad = b"XDIM\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        assert!(Pbf0::deserialize(bad).is_err());
    }

    #[test]
    fn test_dim_rust_native() {
        let mut sp = Space::new("RUST_TEST");
        let mut obj = ObjectND::new(&mut sp);
        obj.add("DIM_RUST", HashMap::new(), 1.0);
        let active = obj.dims_activas();
        assert!(active.contains_key("DIM_RUST"),
            "DIM_RUST not active. dims={:?}", active);
    }

    #[test]
    fn test_weighted_higher_activation() {
        let mut sp = Space::new("WEIGHT_TEST");
        let mut obj = ObjectND::new(&mut sp);
        let mut p = HashMap::new();
        p.insert("k".to_string(), "v".to_string());
        obj.add("DIM_SQL", p.clone(), 1.0);
        obj.add("DIM_PYTHON", p, 0.05);
        assert!(obj.activacion("DIM_SQL") > obj.activacion("DIM_PYTHON"));
    }
}
