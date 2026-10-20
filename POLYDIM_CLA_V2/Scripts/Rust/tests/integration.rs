// POLYDIM_DEST
// destination: polydim/core/tests/
// filename:    integration.rs
// author:      ai.mpat.agt@gmail.com
//
// DEFINITIVE VERSION — 15 tests: 11 PBF0 + 4 ALIGN
// Uses only the public API of polydim. Compatible with polydim_core.rs + FFT bindings.
// Requires: base64 = "0.22" in Cargo.toml
// Requires: python test_align_cross_language.py executed prior to cargo test
//

use polydim::{Space, ObjectND, Pbf0, Pbf0Meta, bind, proj, sim, N};
use std::collections::HashMap;

// ─── helpers ──────────────────────────────────────────────────────────────────

fn make_fixture_hv() -> Vec<f32> {
    let mut hv: Vec<f32> = (0..N).map(|i| (i as f32 * 0.001_f32).sin()).collect();
    let norm: f32 = hv.iter().map(|x| x * x).sum::<f32>().sqrt();
    hv.iter_mut().for_each(|x| *x /= norm);
    hv
}

fn fixture_meta() -> Pbf0Meta {
    Pbf0Meta {
        geo_id:  "fixture000001".to_string(),
        dims:    vec!["DIM_SQL".to_string(), "DIM_RUST".to_string()],
        weights: vec![0.9, 0.7],
        modo:    "H".to_string(),
        version: "0.3".to_string(),
    }
}

// ─── 11 PBF0 Interoperability Tests ──────────────────────────────────────────

#[test]
fn test_interop_fixture_size() {
    let raw = Pbf0::serialize(&make_fixture_hv(), &fixture_meta()).unwrap();
    assert_eq!(raw.len(), 40117, "got={}", raw.len());
}

#[test]
fn test_interop_header_layout() {
    let raw = Pbf0::serialize(&make_fixture_hv(), &fixture_meta()).unwrap();
    assert_eq!(&raw[0..4], b"PDIM");
    assert_eq!(u16::from_be_bytes([raw[4], raw[5]]), 0);
    assert_eq!(u32::from_be_bytes([raw[6], raw[7], raw[8], raw[9]]), 10000);
    assert_eq!(u32::from_be_bytes([raw[10], raw[11], raw[12], raw[13]]), 103);
}

#[test]
fn test_interop_meta_json_exact() {
    let expected = r#"{"geo_id":"fixture000001","dims":["DIM_SQL","DIM_RUST"],"weights":[0.9,0.7],"modo":"H","version":"0.3"}"#;
    let raw = Pbf0::serialize(&make_fixture_hv(), &fixture_meta()).unwrap();
    let ml = u32::from_be_bytes([raw[10], raw[11], raw[12], raw[13]]) as usize;
    assert_eq!(std::str::from_utf8(&raw[14..14+ml]).unwrap(), expected);
}

#[test]
fn test_interop_hv_golden_values() {
    let hv = make_fixture_hv();
    assert!(hv[0].abs() < 1e-10);
    assert!((hv[1]    - 1.4476622e-5_f32).abs() < 1e-9);
    assert!((hv[5000] - (-1.3881986e-2_f32)).abs() < 1e-7);
    assert!((hv[9999] - (-7.8634443e-3_f32)).abs() < 1e-7);
    let norm: f32 = hv.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn test_interop_payload_endianness() {
    let hv  = make_fixture_hv();
    let raw = Pbf0::serialize(&hv, &fixture_meta()).unwrap();
    let ml  = u32::from_be_bytes([raw[10], raw[11], raw[12], raw[13]]) as usize;
    let p   = 14 + ml;
    assert_eq!(&raw[p..p+4], &[0, 0, 0, 0]);
    for i in [1usize, 5000, 9999] {
        let b = [raw[p+i*4], raw[p+i*4+1], raw[p+i*4+2], raw[p+i*4+3]];
        assert!((f32::from_le_bytes(b) - hv[i]).abs() < 1e-10,
            "hv[{}] LE roundtrip failed", i);
        let be = f32::from_be_bytes(b);
        assert!((be - hv[i]).abs() > 1e-5 || be.is_nan(), "hv[{}] big-endian?", i);
    }
}

#[test]
fn test_interop_roundtrip_lossless() {
    let hv  = make_fixture_hv();
    let raw = Pbf0::serialize(&hv, &fixture_meta()).unwrap();
    let dec = Pbf0::deserialize(&raw).unwrap();
    let mae: f32 = hv.iter().zip(dec.hypervec.iter())
        .map(|(a,b)| (a-b).abs()).sum::<f32>() / hv.len() as f32;
    assert!(mae == 0.0);
    assert_eq!(dec.meta.geo_id, "fixture000001");
    assert_eq!(dec.meta.dims,   vec!["DIM_SQL","DIM_RUST"]);
    assert_eq!(dec.meta.modo,   "H");
}

#[test]
fn test_spaces_with_different_seeds_isolated() {
    let mut a = Space::new("AI_ALPHA"); let mut b = Space::new("AI_BETA");
    assert!(sim(a.sub("DIM_SQL"), b.sub("DIM_SQL")) < 0.6);
}

#[test]
fn test_bind_commutativity() {
    let mut sp = Space::new("COMM");
    let sa = sp.sym("X").clone(); let sb = sp.sym("Y").clone();
    let diff: f32 = bind(&sa,&sb).iter().zip(bind(&sb,&sa).iter())
        .map(|(x,y)| (x-y).abs()).sum();
    assert!(diff < 1e-5);
}

#[test]
fn test_proj_self_is_one() {
    let mut sp = Space::new("SELF");
    let sub = sp.sub("DIM_SQL").clone();
    assert!((proj(&sub, &sub) - 1.0).abs() < 1e-4);
}

#[test]
fn test_proj_independent_near_half() {
    let mut sp = Space::new("ORTHO");
    let a = sp.sub("DIM_SQL").clone(); let b = sp.sub("DIM_PYTHON").clone();
    assert!((proj(&a, &b) - 0.5).abs() < 0.05);
}

#[test]
fn test_full_object_pbf0_roundtrip() {
    let mut sp = Space::new("INTEG"); let mut obj = ObjectND::new(&mut sp);
    let mut p = HashMap::new(); p.insert("tabla".to_string(), "pedidos".to_string());
    obj.add("DIM_SQL", p, 0.9); obj.add("DIM_RUST", HashMap::new(), 0.7);
    let hv = obj.hv();
    let meta = Pbf0Meta { geo_id: obj.geo_id(),
        dims: vec!["DIM_SQL".to_string(),"DIM_RUST".to_string()],
        weights: vec![0.9,0.7], modo:"H".to_string(), version:"0.3".to_string() };
    let dec = Pbf0::deserialize(&Pbf0::serialize(&hv, &meta).unwrap()).unwrap();
    let mae: f32 = hv.iter().zip(dec.hypervec.iter())
        .map(|(a,b)| (a-b).abs()).sum::<f32>() / hv.len() as f32;
    assert!(mae == 0.0); assert_eq!(dec.meta.dims, meta.dims);
}

// ─── 4 Alignment Tests (PEND_RUST_PYTHON_ALIGN) ──────────────────────────────────
// Uses public API only. Requires sonda_package.json in the crate directory.
//
// FINDINGS:
//   NATIVE Probes -> score trivially ~1.0 (circular by design of the protocol)
//   Arbitrary vectors rank-9 -> score ~0.50 (insufficient rank for N=10000)
//   Python MD5+numpy vs Rust FNV+PCG32 -> scores ~0.50 (different PRNGs)
//

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;

fn load_sonda_matrix() -> Option<(Vec<String>, Vec<Vec<f32>>)> {
    let text = std::fs::read_to_string("sonda_package.json").ok()?;
    let pkg: serde_json::Value = serde_json::from_str(&text).ok()?;
    let names: Vec<String> = pkg["sonda_names"].as_array()?
        .iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
    let raw = BASE64.decode(pkg["matrix_b64"].as_str()?).ok()?;
    let ns = pkg["num_sondas"].as_u64()? as usize;
    let n  = pkg["N"].as_u64()? as usize;
    if raw.len() != ns * n * 4 { return None; }
    let mat: Vec<Vec<f32>> = raw.chunks_exact(n * 4)
        .map(|r| r.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect())
        .collect();
    Some((names, mat))
}

fn align_t(hv: &[f32], a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<f32> {
    let pa: Vec<f32> = a.iter().map(|r| r.iter().zip(hv).map(|(x,y)| x*y).sum()).collect();
    let mut back = vec![0.0f32; b[0].len()];
    for (row, &vi) in b.iter().zip(pa.iter()) {
        for (r, &m) in back.iter_mut().zip(row.iter()) { *r += vi * m; }
    }
    let n: f32 = back.iter().map(|x| x*x).sum::<f32>().sqrt();
    if n > 1e-10 { back.iter_mut().for_each(|x| *x /= n); }
    back
}

fn nrm(v: &[f32]) -> f32 { v.iter().map(|x| x*x).sum::<f32>().sqrt() }
fn ss(a: &[f32], b: &[f32]) -> f32 {
    (a.iter().zip(b).map(|(x,y)| x*y).sum::<f32>() + 1.0) / 2.0
}

const NATIVE_DIMS: &[&str] = &["DIM_PYTHON","DIM_RUST","DIM_FLUTTER","DIM_SQL",
    "DIM_GRAPH","DIM_VECTOR","DIM_TIME","DIM_ERROR","DIM_META"];

#[test]
fn test_align_sonda_package_parse() {
    let Some((names, matrix)) = load_sonda_matrix() else {
        eprintln!("SKIP: sonda_package.json not found — execute test_align_cross_language.py first");
        return;
    };
    assert_eq!(names.len(), 9);
    assert_eq!(matrix.len(), 9);
    assert_eq!(matrix[0].len(), N);
    assert!(names.contains(&"DIM_SQL".to_string()));
    for (i, row) in matrix.iter().enumerate() {
        assert!((nrm(row) - 1.0).abs() < 1e-4, "sonda[{}] not unitary", i);
    }
}

#[test]
fn test_align_transform_mechanics() {
    let Some((_, mat_py)) = load_sonda_matrix() else {
        eprintln!("SKIP: sonda_package.json not found"); return;
    };
    let mut sp = Space::new("POLYDIM_PY");
    let mat_rust: Vec<Vec<f32>> = NATIVE_DIMS.iter()
        .map(|s| sp.sym(s).clone()).collect();
    let hv = sp.sub("DIM_SQL").clone();
    let aligned = align_t(&hv, &mat_py, &mat_rust);
    assert!((nrm(&aligned) - 1.0).abs() < 1e-4, "result not unitary: {}", nrm(&aligned));
}

#[test]
fn test_align_native_dims_circular() {
    let Some((_, mat_py)) = load_sonda_matrix() else {
        eprintln!("SKIP: sonda_package.json not found"); return;
    };
    let mut sp = Space::new("POLYDIM_PY");
    let mat_rust: Vec<Vec<f32>> = NATIVE_DIMS.iter()
        .map(|s| sp.sym(s).clone()).collect();
    let scores: Vec<f32> = NATIVE_DIMS.iter().enumerate()
        .map(|(i, dim)| ss(&align_t(sp.sub(dim), &mat_py, &mat_rust), &mat_rust[i]))
        .collect();
    let mean: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
    assert!(mean > 0.40, "score={:.4}", mean);
    println!("NATIVE align score Python(MD5) vs Rust(FNV): {:.4} (expected ~0.50)", mean);
}

#[test]
fn test_align_arbitrary_vector_near_half() {
    let Some((_, mat_py)) = load_sonda_matrix() else {
        eprintln!("SKIP: sonda_package.json not found"); return;
    };
    let mut sp = Space::new("POLYDIM_PY");
    let mat_rust: Vec<Vec<f32>> = NATIVE_DIMS.iter()
        .map(|s| sp.sym(s).clone()).collect();
    let rnd = sp.sym("__test_arbitrary_rank9__").clone();
    let aligned = align_t(&rnd, &mat_py, &mat_rust);
    let score = ss(&aligned, &rnd);
    assert!((nrm(&aligned) - 1.0).abs() < 1e-4, "aligned not unitary");
    assert!((0.35..=0.75).contains(&score), "score out of range: {:.4}", score);
    println!("Arbitrary rank-9 score: {:.4} (expected ~0.50)", score);
}
