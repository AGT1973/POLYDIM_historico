// woodbury_vm.rs
//
// Tarea 23 — Arquetipo de máquina virtual mínima (Rust, target wasm32-unknown-unknown)
// Opcode central: WOODBURY_INV, que invierte una matrix de covarianza
// Sigma = D + U U^T  (diagonal + bajo rango, "DPLR")
// usando la identidad de Woodbury en vez de una inversión densa O(d^3).
//
// No usa crates externos (std puro) para que compile igual con:
//   rustc woodbury_vm.rs -o woodbury_vm            (nativo, para el sanity-check)
//   rustc --target wasm32-unknown-unknown --crate-type=cdylib woodbury_vm.rs -o woodbury_vm.wasm
//
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "shape mismatch");
        Matrix { rows, cols, data }
    }

    fn identity(n: usize) -> Self {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "incompatible shapes for mul");
        let mut out = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a_ik = self.get(i, k);
                if a_ik == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    out.data[i * out.cols + j] += a_ik * other.get(k, j);
                }
            }
        }
        out
    }

    fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Inversión densa por Gauss-Jordan con pivoteo parcial.
    /// Se usa (a) para invertir el bloque pequeño r x r de Woodbury,
    /// y (b) como referencia "naive" O(n^3) para verificar el resultado.
    fn invert_dense(&self) -> Option<Matrix> {
        assert_eq!(self.rows, self.cols, "invert_dense requires a square matrix");
        let n = self.rows;
        let mut aug = vec![0.0f64; n * 2 * n];
        for r in 0..n {
            for c in 0..n {
                aug[r * (2 * n) + c] = self.get(r, c);
            }
            aug[r * (2 * n) + (n + r)] = 1.0;
        }

        for col in 0..n {
            // pivoteo parcial
            let mut pivot_row = col;
            let mut best = aug[col * (2 * n) + col].abs();
            for r in (col + 1)..n {
                let v = aug[r * (2 * n) + col].abs();
                if v > best {
                    best = v;
                    pivot_row = r;
                }
            }
            if best < 1e-13 {
                return None; // matriz singular / mal condicionada
            }
            if pivot_row != col {
                for c in 0..(2 * n) {
                    aug.swap(col * (2 * n) + c, pivot_row * (2 * n) + c);
                }
            }
            let pivot = aug[col * (2 * n) + col];
            for c in 0..(2 * n) {
                aug[col * (2 * n) + c] /= pivot;
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let factor = aug[r * (2 * n) + col];
                if factor == 0.0 {
                    continue;
                }
                for c in 0..(2 * n) {
                    aug[r * (2 * n) + c] -= factor * aug[col * (2 * n) + c];
                }
            }
        }

        let mut inv = Matrix::zeros(n, n);
        for r in 0..n {
            for c in 0..n {
                inv.set(r, c, aug[r * (2 * n) + (n + c)]);
            }
        }
        Some(inv)
    }
}

/// Invierte Sigma = D + U U^T mediante la identidad de Woodbury:
///   Sigma^{-1} = D^{-1} - D^{-1} U (I_r + U^T D^{-1} U)^{-1} U^T D^{-1}
///
/// D es diagonal (d_diag[i] > 0), U es d x r con r << d.
/// Costo: O(d*r^2 + r^3) frente a O(d^3) de una inversión densa.
fn woodbury_inv(d_diag: &[f64], u: &Matrix) -> Option<Matrix> {
    let d = d_diag.len();
    assert_eq!(u.rows, d, "U debe tener d filas");
    let r = u.cols;

    // D^{-1} U  (d x r), escalando cada fila i por 1/d_diag[i]
    let mut dinv_u = Matrix::zeros(d, r);
    for i in 0..d {
        if d_diag[i].abs() < 1e-14 {
            return None; // D no invertible: falla numérica documentable
        }
        let inv_di = 1.0 / d_diag[i];
        for j in 0..r {
            dinv_u.set(i, j, inv_di * u.get(i, j));
        }
    }

    // Bloque pequeño: I_r + U^T D^{-1} U   (r x r)
    let ut = u.transpose();
    let inner = ut.mul(&dinv_u); // r x r
    let inner_plus_i = inner.add(&Matrix::identity(r));

    let inner_inv = inner_plus_i.invert_dense()?; // r x r, el único bloque "caro"

    // Sigma^{-1} = D^{-1} - (D^{-1}U) * inner_inv * (D^{-1}U)^T
    let mid = dinv_u.mul(&inner_inv).mul(&dinv_u.transpose()); // d x d
    let mut sigma_inv = Matrix::zeros(d, d);
    for i in 0..d {
        sigma_inv.set(i, i, 1.0 / d_diag[i]);
    }
    Some(sigma_inv.sub(&mid))
}

// ---------------------------------------------------------------------------
// Máquina virtual mínima: un stack machine de 5 opcodes que opera sobre
// "valores" que son o bien matrices o bien escalares (para VERIFY/HALT).
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Value {
    Mat(Matrix),
    Scalar(f64),
    Bool(bool),
}

enum Op {
    LoadDiag(Vec<f64>),
    LoadU(Matrix),
    WoodburyInv,
    Verify(f64), // tolerancia epsilon
    Halt,
}

struct Vm {
    stack: Vec<Value>,
    // últimos D y U cargados (el opcode WOODBURY_INV los consume)
    pending_d: Option<Vec<f64>>,
    pending_u: Option<Matrix>,
    trace: Vec<String>,
}

impl Vm {
    fn new() -> Self {
        Vm { stack: Vec::new(), pending_d: None, pending_u: None, trace: Vec::new() }
    }

    fn run(&mut self, program: &[Op]) {
        for op in program {
            match op {
                Op::LoadDiag(d) => {
                    self.trace.push(format!("LOAD_DIAG(len={})", d.len()));
                    self.pending_d = Some(d.clone());
                }
                Op::LoadU(u) => {
                    self.trace.push(format!("LOAD_U({}x{})", u.rows, u.cols));
                    self.pending_u = Some(u.clone());
                }
                Op::WoodburyInv => {
                    let d = self.pending_d.take().expect("falta LOAD_DIAG antes de WOODBURY_INV");
                    let u = self.pending_u.take().expect("falta LOAD_U antes de WOODBURY_INV");
                    match woodbury_inv(&d, &u) {
                        Some(sigma_inv) => {
                            self.trace.push("WOODBURY_INV -> ok".to_string());
                            self.stack.push(Value::Mat(sigma_inv));
                            // dejamos también Sigma reconstruida para VERIFY
                            let mut sigma = Matrix::zeros(d.len(), d.len());
                            for i in 0..d.len() {
                                sigma.set(i, i, d[i]);
                            }
                            let uut = u.mul(&u.transpose());
                            let sigma = sigma.add(&uut);
                            self.stack.push(Value::Mat(sigma));
                        }
                        None => {
                            self.trace.push("WOODBURY_INV -> FALLO (D singular o mal condicionada)".to_string());
                            self.stack.push(Value::Bool(false));
                        }
                    }
                }
                Op::Verify(eps) => {
                    // espera en el tope: Sigma, luego Sigma^{-1}
                    let sigma = self.stack.pop().expect("stack vacío en VERIFY");
                    let sigma_inv = self.stack.pop().expect("stack vacío en VERIFY");
                    if let (Value::Mat(sigma), Value::Mat(sigma_inv)) = (sigma, sigma_inv) {
                        let n = sigma.rows;
                        let prod = sigma.mul(&sigma_inv);
                        let residual = prod.sub(&Matrix::identity(n)).frobenius_norm();
                        let ok = residual < *eps;
                        self.trace.push(format!(
                            "VERIFY: ||Sigma*Sigma^-1 - I||_F = {:.3e} (eps={:.1e}) -> {}",
                            residual, eps, if ok { "PASA" } else { "FALLA" }
                        ));
                        self.stack.push(Value::Bool(ok));
                        self.stack.push(Value::Scalar(residual));
                    } else {
                        self.trace.push("VERIFY: no hay matrices válidas en el stack (inversión falló antes)".to_string());
                        self.stack.push(Value::Bool(false));
                    }
                }
                Op::Halt => {
                    self.trace.push("HALT".to_string());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Generador congruencial lineal simple (para no depender del crate `rand`)
// ---------------------------------------------------------------------------
struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }
    fn next_f64(&mut self) -> f64 {
        // parámetros de Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

fn main() {
    let d = 8usize; // dimensión ambiente
    let r = 2usize; // rango de la proyección (r << d)

    let mut rng = Lcg::new(42);

    // D: diagonal estrictamente positiva (varianzas "propias")
    let d_diag: Vec<f64> = (0..d).map(|_| rng.next_range(0.5, 2.0)).collect();

    // U: factor de bajo rango d x r
    let mut u = Matrix::zeros(d, r);
    for i in 0..d {
        for j in 0..r {
            u.set(i, j, rng.next_range(-1.0, 1.0));
        }
    }

    println!("=== Caso 1: D bien condicionada ===");
    let mut vm = Vm::new();
    let program = vec![
        Op::LoadDiag(d_diag.clone()),
        Op::LoadU(u.clone()),
        Op::WoodburyInv,
        Op::Verify(1e-8),
        Op::Halt,
    ];
    vm.run(&program);
    for line in &vm.trace {
        println!("{}", line);
    }

    // Contraste contra inversión densa "naive" (referencia O(d^3))
    let mut sigma = Matrix::zeros(d, d);
    for i in 0..d {
        sigma.set(i, i, d_diag[i]);
    }
    let sigma = sigma.add(&u.mul(&u.transpose()));
    let naive_inv = sigma.invert_dense().expect("Sigma debería ser invertible en el caso bien condicionado");
    let woodbury_result = woodbury_inv(&d_diag, &u).unwrap();
    let diff = naive_inv.sub(&woodbury_result).frobenius_norm();
    println!("Diferencia Woodbury vs inversión densa naive: {:.3e}", diff);

    println!("\n=== Caso 2: D con una entrada ~0 (falla numérica esperada) ===");
    let mut d_diag_bad = d_diag.clone();
    d_diag_bad[3] = 1e-16; // rompe la invertibilidad de D
    let mut vm2 = Vm::new();
    let program2 = vec![
        Op::LoadDiag(d_diag_bad),
        Op::LoadU(u.clone()),
        Op::WoodburyInv,
        Op::Halt,
    ];
    vm2.run(&program2);
    for line in &vm2.trace {
        println!("{}", line);
    }
}
