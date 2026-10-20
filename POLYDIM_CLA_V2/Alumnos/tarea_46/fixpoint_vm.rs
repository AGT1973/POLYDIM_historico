// fixpoint_vm.rs
// Arquetipo de Maquina Virtual minima ("Abstract Tensor Machine", instruccion FIXPOINT).
// No usa crates externos -> compila tanto en target nativo como en wasm32-unknown-unknown.
//
// Compilar nativo:  rustc -O fixpoint_vm.rs -o fixpoint_vm && ./fixpoint_vm
// Compilar a WASM:  rustc --target wasm32-unknown-unknown -O fixpoint_vm.rs -o fixpoint_vm.wasm
//                   (requiere: rustup target add wasm32-unknown-unknown)

/// Matriz cuadrada n x n almacenada en formato row-major (Vec<f64> plano).
#[derive(Clone, Debug)]
struct Mat {
    n: usize,
    data: Vec<f64>,
}

impl Mat {
    fn new(n: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), n * n);
        Mat { n, data }
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] = v;
    }

    fn mat_vec(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut out = vec![0.0; n];
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                acc += self.get(i, j) * x[j];
            }
            out[i] = acc;
        }
        out
    }

    /// Transpuesta aplicada a un vector: W^T x
    fn mat_t_vec(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut out = vec![0.0; n];
        for j in 0..n {
            let mut acc = 0.0;
            for i in 0..n {
                acc += self.get(i, j) * x[i];
            }
            out[j] = acc;
        }
        out
    }

    fn scale(&mut self, s: f64) {
        for v in self.data.iter_mut() {
            *v *= s;
        }
    }
}

fn vec_norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn vec_normalize(v: &mut [f64]) {
    let nrm = vec_norm2(v);
    if nrm > 1e-14 {
        for x in v.iter_mut() {
            *x /= nrm;
        }
    }
}

fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Estima la norma espectral (mayor valor singular) de W mediante power iteration
/// sobre el operador W^T W (metodo estandar, usado p.ej. en Spectral Normalization
/// para GANs, Miyato et al. 2018). sigma_max(W) = sqrt(lambda_max(W^T W)).
fn spectral_norm_power_iteration(w: &Mat, iters: usize, seed: u64) -> f64 {
    let n = w.n;
    // generador congruencial simple, determinista, sin dependencias externas
    let mut state = seed.max(1);
    let mut u: Vec<f64> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64 / u32::MAX as f64) - 0.5
        })
        .collect();
    vec_normalize(&mut u);

    for _ in 0..iters {
        let v = w.mat_vec(&u); // v = W u
        let mut wtv = w.mat_t_vec(&v); // W^T (W u)
        vec_normalize(&mut wtv);
        u = wtv;
    }
    // Rayleigh quotient: sigma_max^2 ~ u^T (W^T W) u
    let wu = w.mat_vec(&u);
    vec_norm2(&wu) // ||W u|| con ||u||=1  approx sigma_max
}

/// Devuelve (sigma_antes, sigma_despues). sigma_despues = min(sigma_antes, lambda_target)
/// por construccion (si se reescala, queda exactamente en lambda_target salvo error
/// numerico de la power iteration).
fn enforce_spectral_bound(w: &mut Mat, lambda_target: f64, power_iters: usize) -> (f64, f64) {
    let sigma_before = spectral_norm_power_iteration(w, power_iters, 42);
    if sigma_before > lambda_target {
        w.scale(lambda_target / sigma_before);
    }
    let sigma_after = spectral_norm_power_iteration(w, power_iters, 43);
    (sigma_before, sigma_after)
}

fn tanh_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| v.tanh()).collect()
}

/// Resultado de correr el operador FIXPOINT.
struct FixpointResult {
    x_star: Vec<f64>,
    iterations: usize,
    converged: bool,
    sigma_before: f64,
    sigma_after: f64,
    error_trace: Vec<f64>, // ||x_{k+1} - x_k|| en cada paso, para verificar tasa geometrica
}

/// Instruccion FIXPOINT de la VM abstracta.
/// T(x) = tanh(W x + b)   (tanh es 1-Lipschitz => Lip(T) <= sigma_max(W))
/// Si sigma_max(W) < 1, T es una contraccion (Banach) sobre R^n con esa metrica,
/// y la iteracion x_{k+1} = T(x_k) converge a un unico punto fijo x*.
fn fixpoint_op(
    mut w: Mat,
    b: Vec<f64>,
    x0: Vec<f64>,
    lambda_target: f64,
    tol: f64,
    max_iter: usize,
) -> FixpointResult {
    let (sigma_before, sigma_after) = enforce_spectral_bound(&mut w, lambda_target, 50);

    let mut x = x0;
    let mut error_trace = Vec::with_capacity(max_iter);
    let mut converged = false;
    let mut iterations = 0;

    for k in 0..max_iter {
        let wx = w.mat_vec(&x);
        let pre = wx.iter().zip(b.iter()).map(|(a, c)| a + c).collect::<Vec<_>>();
        let x_next = tanh_vec(&pre);
        let err = vec_norm2(&vec_sub(&x_next, &x));
        error_trace.push(err);
        x = x_next;
        iterations = k + 1;
        if err < tol {
            converged = true;
            break;
        }
    }

    FixpointResult {
        x_star: x,
        iterations,
        converged,
        sigma_before,
        sigma_after,
        error_trace,
    }
}

fn main() {
    // Ejemplo: n=4, W fijada explicitamente para reproducibilidad total.
    let n = 4;
    let w = Mat::new(
        n,
        vec![
            0.9, 0.4, -0.3, 0.2, 0.1, -0.8, 0.5, 0.3, -0.2, 0.6, 0.7, -0.4, 0.3, -0.1, 0.2, 0.85,
        ],
    );
    let b = vec![0.05, -0.02, 0.01, 0.0];
    let x0 = vec![1.0, 1.0, 1.0, 1.0];

    let lambda_target = 0.9; // lambda < 1 exigido por el enunciado
    let tol = 1e-10;
    let max_iter = 500;

    let res = fixpoint_op(w, b, x0, lambda_target, tol, max_iter);

    println!("sigma_max(W) antes de normalizar : {:.6}", res.sigma_before);
    println!("sigma_max(W) despues de normalizar: {:.6}", res.sigma_after);
    println!("lambda objetivo (contraccion)     : {:.6}", lambda_target);
    println!("convergio: {}", res.converged);
    println!("iteraciones: {}", res.iterations);
    println!("x*: {:?}", res.x_star);

    // Verificacion empirica de tasa geometrica: err_{k+1} / err_k ~ lambda
    println!("\n k | error ||x_(k+1)-x_k||_2 | ratio_k = err_(k+1)/err_k");
    for k in 1..res.error_trace.len().min(12) {
        let ratio = res.error_trace[k] / res.error_trace[k - 1];
        println!("{:2} | {:.10e} | {:.6}", k, res.error_trace[k], ratio);
    }
}
