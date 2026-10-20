# POLYDIM WORKSPACE — RUST PROTOTYPE
**Date:** 2026-06-24
**Project:** POLYDIM — N-dimensional Language for Agent-to-Agent Communication
**Account:** ai.mpat.agt@gmail.com

---

## 1. Structure and Workflow
This directory acts as the **local ↔ cloud bridge** for the POLYDIM Rust prototype.
- **Local View (User):** Seen as a directory in Windows (`Scripts/Rust`).
- **Cloud View (AI Agent):** Accessed via Google Drive MCP.
- What the AI writes here instantly appears in the user's workspace.

---

## 2. Project Status (as of 2026-06-24)
- **Python Runtime V0.3:** 29/29 tests passed (Stable)
- **Rust Core V0.1:** 13/13 tests passed (Compiled & Verified on Windows)
- **PBF0 Binary Format:** Fully specified (`SPEC_FORMATO_BINARIO_V0.md`)
- **Weighted Inference:** Implemented (`polydim_weighted_inference.py`)

---

## 3. Rust Crate Organization
The Rust prototype is structured as a standard compilable Cargo package:
- `Cargo.toml`: Package manifest with serialization and FFT dependencies.
- `polydim_core.rs`: Core VSA library (hypervector space, circular binding, superposition, projections).
- `polydim_fft_bind.rs`: FFT-based circular convolution binding implementation.
- `tests/integration.rs`: Definitive integration test suite containing 15 tests (11 PBF0 interop tests + 4 ALIGN tests).

### How to Run the Tests:
1. Generate the interop binary fixtures from Python:
   ```bash
   python test_pbf0_interop.py
   python test_align_cross_language.py
   ```
2. Execute the Rust test engine:
   ```bash
   cargo test
   ```
