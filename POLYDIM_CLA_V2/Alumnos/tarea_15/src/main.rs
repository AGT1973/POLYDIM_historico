// tarea_15 — Arquetipo de Máquina Virtual mínima (Rust / WASM)
// -------------------------------------------------------------
// Modelo: sistema lógico de restricciones de pertenencia (membership
// constraints) sobre un retículo finito, que colapsa de forma estable
// (converge a un único punto fijo, independiente del orden de evaluación)
// por el Teorema de Knaster–Tarski aplicado a operadores monótonos sobre
// retículos completos finitos.
//
// Cada variable del sistema es un subconjunto de un universo finito
// U = {0, 1, ..., 63}, representado como un bitset en un u64 (bit i = 1
// significa "el elemento i pertenece al conjunto"). El orden del retículo
// es la inclusión de conjuntos (⊆), que corresponde bit a bit al operador
// AND/OR de bits.
//
// Cada "regla de pertenencia" es una función monótona que, a partir del
// estado actual, propone un conjunto de elementos que DEBEN pertenecer a
// una variable destino. La actualización es siempre de tipo
//     A_i  <-  A_i ∪ regla_i(estado)
// (unión, nunca resta), lo cual garantiza monotonía estricta de la
// secuencia de estados y por lo tanto terminación en un retículo finito
// (Teorema del punto fijo de Kleene / Knaster–Tarski constructivo).

const N_VARS: usize = 4;
const UNIVERSE_MASK: u64 = 0xFFF; // usamos solo 12 elementos (0..11) para legibilidad

/// Estado del sistema: un vector de N_VARS subconjuntos de U, cada uno
/// codificado como bitset en un u64.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct State {
    pub sets: [u64; N_VARS],
}

impl State {
    fn bottom() -> Self {
        State { sets: [0; N_VARS] }
    }
}

/// Firma de una regla monótona: recibe el estado completo (solo lectura)
/// y devuelve el conjunto de elementos que deben añadirse a la variable
/// destino. Todas las reglas aquí son composiciones de uniones e
/// intersecciones de conjuntos existentes, o filtros por un predicado fijo
/// sobre los elementos — todas ellas son operadores monótonos: si el
/// estado de entrada crece (⊆), la salida de la regla también crece (⊆).
type Rule = fn(&State) -> u64;

struct RuleSpec {
    target: usize,
    apply: Rule,
}

// Regla 0: C ⊇ A ∩ B   (intersección de dos conjuntos crecientes es creciente)
fn rule_intersection(s: &State) -> u64 {
    s.sets[0] & s.sets[1]
}

// Regla 1: D ⊇ C ∪ A   (unión es trivialmente monótona)
fn rule_union(s: &State) -> u64 {
    s.sets[2] | s.sets[0]
}

// Regla 2: B ⊇ { x en D : x es par }  (filtrar por un predicado fijo es
// monótono: al crecer D solo pueden aparecer más elementos que cumplan
// el predicado, nunca menos)
fn rule_even_filter(s: &State) -> u64 {
    let mut out = 0u64;
    let mut d = s.sets[3];
    while d != 0 {
        let bit = d.trailing_zeros();
        if bit % 2 == 0 {
            out |= 1 << bit;
        }
        d &= d - 1; // apaga el bit más bajo
    }
    out
}

const RULES: [RuleSpec; 3] = [
    RuleSpec { target: 2, apply: rule_intersection },
    RuleSpec { target: 3, apply: rule_union },
    RuleSpec { target: 1, apply: rule_even_filter },
];

/// Aplica el conjunto de reglas en el orden `order` una sola vez
/// (una "pasada" / sweep). Devuelve `true` si algún componente del
/// estado cambió.
fn sweep(state: &mut State, order: &[usize]) -> bool {
    let mut changed = false;
    for &idx in order {
        let spec = &RULES[idx];
        let add = (spec.apply)(state) & UNIVERSE_MASK;
        let before = state.sets[spec.target];
        let after = before | add;
        if after != before {
            state.sets[spec.target] = after;
            changed = true;
        }
    }
    changed
}

/// Itera `sweep` hasta que el estado deje de cambiar (punto fijo).
/// Termina siempre: cada componente es un subconjunto de un universo
/// finito y la secuencia de estados es monótonamente creciente (⊆),
/// así que en a lo sumo |U| * N_VARS pasadas se alcanza el punto fijo
/// (cota de Knaster–Tarski constructivo / cadena ascendente finita).
pub fn run_to_fixpoint(init: State, order: &[usize]) -> (State, u32) {
    let mut state = init;
    let mut sweeps = 0u32;
    loop {
        let changed = sweep(&mut state, order);
        sweeps += 1;
        if !changed {
            break;
        }
        if sweeps > 10_000 {
            panic!("no convergio dentro del limite de seguridad");
        }
    }
    (state, sweeps)
}

/// Punto de entrada exportable a WASM (wasm32-unknown-unknown).
/// Ejecuta el sistema desde un estado inicial fijo con el orden de
/// reglas [0,1,2] y devuelve el estado final empaquetado en un único
/// u64 (solo el componente D, sets[3], a modo de resultado observable
/// simple desde JS: instanciar con wasm-bindgen/wasm-pack si se desea
/// una interfaz más rica).
#[no_mangle]
pub extern "C" fn wasm_run_demo() -> u64 {
    let mut init = State::bottom();
    init.sets[0] = 0b0000_0111; // A = {0,1,2}
    let (final_state, _sweeps) = run_to_fixpoint(init, &[0, 1, 2]);
    final_state.sets[3]
}

// --------------------------------------------------------------------
// Comprobación nativa (no se ejecuta como parte de la entrega, ver
// instrucciones en el .md: el alumno la corre él mismo con `cargo run`).
// --------------------------------------------------------------------
fn set_to_vec(mask: u64) -> Vec<u32> {
    let mut v = Vec::new();
    for i in 0..64 {
        if mask & (1 << i) != 0 {
            v.push(i);
        }
    }
    v
}

fn main() {
    let mut init = State::bottom();
    init.sets[0] = 0b0000_0111; // A = {0,1,2}

    // Cuatro órdenes de evaluación distintos, incluyendo permutaciones
    // que intercalan reglas "hacia atrás" respecto de su dependencia
    // declarada, para estresar la propiedad de independencia de orden.
    let orders: [[usize; 3]; 4] = [
        [0, 1, 2],
        [2, 1, 0],
        [1, 0, 2],
        [2, 0, 1],
    ];

    let mut all_results = Vec::new();
    for order in orders.iter() {
        let (final_state, sweeps) = run_to_fixpoint(init, order);
        println!(
            "orden {:?} -> punto fijo A={:?} B={:?} C={:?} D={:?}  ({} pasadas)",
            order,
            set_to_vec(final_state.sets[0]),
            set_to_vec(final_state.sets[1]),
            set_to_vec(final_state.sets[2]),
            set_to_vec(final_state.sets[3]),
            sweeps
        );
        all_results.push(final_state);
    }

    let first = all_results[0];
    let all_equal = all_results.iter().all(|s| *s == first);
    println!(
        "\n¿Todas las trayectorias colapsan al MISMO punto fijo, sin importar el orden? {}",
        all_equal
    );
    assert!(all_equal, "el sistema NO fue orden-independiente: revisar monotonía de las reglas");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_and_is_order_independent() {
        let mut init = State::bottom();
        init.sets[0] = 0b0000_0111;
        let orders: [[usize; 3]; 4] = [
            [0, 1, 2],
            [2, 1, 0],
            [1, 0, 2],
            [2, 0, 1],
        ];
        let results: Vec<State> = orders
            .iter()
            .map(|o| run_to_fixpoint(init, o).0)
            .collect();
        for r in &results[1..] {
            assert_eq!(*r, results[0]);
        }
    }

    #[test]
    fn terminates_within_finite_bound() {
        let mut init = State::bottom();
        init.sets[0] = 0b0000_0111;
        let (_s, sweeps) = run_to_fixpoint(init, &[0, 1, 2]);
        // cota grosera: nunca debería necesitar más pasadas que
        // (numero de variables) * (tamaño del universo)
        assert!(sweeps as usize <= N_VARS * 12);
    }
}
