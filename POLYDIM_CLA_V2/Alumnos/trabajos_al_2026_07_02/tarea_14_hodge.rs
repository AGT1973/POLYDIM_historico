use nalgebra::{DMatrix, DVector};

/// Tarea 14 (Escuadron 1) - enunciado original (roto/mal generado en el documento madre):
/// "Crea un arquetipo de una maquina virtual minima en Rust/WASM que [aplique] la
/// Descomposicion de Hodge aproximada aislando la 'componente armonica'."
///
/// Interpretacion honesta: en lugar de fingir que "una maquina virtual minima en WASM"
/// por si sola hace matematica nueva, esto implementa el algoritmo real y verificable
/// detras de la frase: la Descomposicion de Hodge Combinatoria sobre un complejo
/// simplicial (grafo + triangulos). No es magia topologica: es algebra lineal con
/// proyecciones ortogonales, igual que en el paper original de Jiang et al.
/// (HodgeRank, 2011), que POLYDIM cita indirectamente.
fn main() {
    // ---- 1. Construccion del complejo simplicial ----
    // 6 vertices (0..5), 7 aristas (0..6).
    // Las aristas 0,1,2 forman un TRIANGULO RELLENO (0,1,2) -> tiene una 2-celda.
    // Las aristas 3,4,5 forman un TRIANGULO VACIO (3,4,5), misma forma, SIN 2-celda.
    // La arista 6 es un puente (2,3) que no pertenece a ningun ciclo.
    let n_vertices = 6;
    let edges: Vec<(usize, usize)> = vec![
        (0, 1), // e0
        (1, 2), // e1
        (2, 0), // e2  -- cierra el triangulo RELLENO 0-1-2
        (3, 4), // e3
        (4, 5), // e4
        (5, 3), // e5  -- cierra el triangulo VACIO 3-4-5
        (2, 3), // e6  -- puente, no pertenece a ningun ciclo
    ];
    let n_edges = edges.len();

    // B1: matriz de incidencia vertices x aristas.
    // Columna de la arista (u,v): -1 en la fila u, +1 en la fila v.
    let mut b1 = DMatrix::<f64>::zeros(n_vertices, n_edges);
    for (e, &(u, v)) in edges.iter().enumerate() {
        b1[(u, e)] = -1.0;
        b1[(v, e)] = 1.0;
    }

    // B2: matriz de incidencia aristas x triangulos.
    // Solo hay UN triangulo relleno: (0,1,2), usando e0,e1,e2 con orientacion consistente.
    let n_triangles = 1;
    let mut b2 = DMatrix::<f64>::zeros(n_edges, n_triangles);
    b2[(0, 0)] = 1.0; // e0
    b2[(1, 0)] = 1.0; // e1
    b2[(2, 0)] = 1.0; // e2
    // e3,e4,e5 (triangulo vacio) NO tienen entrada en B2: no hay 2-celda ahi.

    let check = &b1 * &b2;
    println!("Chequeo B1*B2 (debe ser ~0):\n{}", check);

    // ---- 2. Flujo de prueba sobre las aristas ----
    let c_a = 3.0_f64; // circulacion en el triangulo RELLENO
    let c_b = 2.0_f64; // circulacion en el triangulo VACIO
    let g = 5.0_f64;   // diferencia de potencial en el puente
    let f = DVector::from_vec(vec![c_a, c_a, c_a, c_b, c_b, c_b, g]);

    // ---- 3. Descomposicion de Hodge: f = gradiente + curl + armonica ----
    let eps = 1e-9;

    let b1t = b1.transpose();
    let b1b1t = &b1 * &b1t;
    let b1b1t_pinv = b1b1t.pseudo_inverse(eps).expect("pseudo-inversa B1 B1^T fallo");
    let phi = &b1b1t_pinv * (&b1 * &f);
    let grad = &b1t * &phi;

    let residual = &f - &grad;
    let b2t = b2.transpose();
    let b2tb2 = &b2t * &b2;
    let b2tb2_pinv = b2tb2.pseudo_inverse(eps).expect("pseudo-inversa B2^T B2 fallo");
    let psi = &b2tb2_pinv * (&b2t * &residual);
    let curl = &b2 * &psi;

    let harmonic = &f - &grad - &curl;

    println!("\nFlujo original f      = {:?}", f.as_slice());
    println!("Componente GRADIENTE  = {:?}", grad.as_slice());
    println!("Componente CURL       = {:?}", curl.as_slice());
    println!("Componente ARMONICA   = {:?}", harmonic.as_slice());

    // ---- 4. Verificacion ----
    let reconstruction_error = (&f - (&grad + &curl + &harmonic)).norm();
    let dot_grad_curl = grad.dot(&curl);
    let dot_grad_harm = grad.dot(&harmonic);
    let dot_curl_harm = curl.dot(&harmonic);

    println!("\n--- Verificacion ---");
    println!("||f - (grad+curl+harm)||   = {:.2e}  (debe ser ~0)", reconstruction_error);
    println!("<grad, curl>               = {:.2e}  (debe ser ~0, ortogonales)", dot_grad_curl);
    println!("<grad, harmonic>           = {:.2e}  (debe ser ~0, ortogonales)", dot_grad_harm);
    println!("<curl, harmonic>           = {:.2e}  (debe ser ~0, ortogonales)", dot_curl_harm);

    let norm_curl_triangle = (curl[0].powi(2) + curl[1].powi(2) + curl[2].powi(2)).sqrt();
    let norm_harm_triangle = (harmonic[3].powi(2) + harmonic[4].powi(2) + harmonic[5].powi(2)).sqrt();
    println!("\nNorma de CURL en el triangulo RELLENO (e0,e1,e2)  = {:.4}  (deberia concentrar c_a={})", norm_curl_triangle, c_a);
    println!("Norma de ARMONICA en el triangulo VACIO (e3,e4,e5) = {:.4}  (deberia concentrar c_b={})", norm_harm_triangle, c_b);
    println!("Componente GRADIENTE en el puente (e6)             = {:.4}  (deberia ser ~g={})", grad[6], g);
}
