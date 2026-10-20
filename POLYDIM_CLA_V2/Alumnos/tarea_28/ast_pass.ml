(* Tarea 28 - Pasada de optimizacion sobre un AST de expresiones del
   Algebra de Clifford Cl(3,0), truncada.

   Un multivector denso se representa como un float array de longitud 8,
   indexado por bitmask sobre {e1,e2,e3} (ver verificacion_geometric_product.py
   para la correspondencia exacta de indices).

   La pasada:
     1) hace constant folding del producto geometrico (GProd) cuando ambos
        operandos son constantes conocidas en tiempo de compilacion;
     2) elide el operador explicito de truncamiento TruncEven SOLO cuando
        puede demostrar estaticamente (via el grade_tag) que ambos operandos
        del producto ya viven en la subalgebra par {1, e12, e13, e23}, que es
        cerrada bajo el producto geometrico;
     3) en cualquier otro caso, NO elide el truncamiento: lo conserva y emite
        un diagnostico, en vez de asumir sin evidencia que el truncamiento
        "grado <= 2" es seguro (no lo es en general: ver el contraejemplo
        e12 * e3 = e123 documentado en el .md de entrega). *)

let popcount x =
  let rec go x acc = if x = 0 then acc else go (x lsr 1) (acc + (x land 1)) in
  go x 0

(* Signo del producto geometrico entre dos blades base a, b, para
   metrica euclidiana (e_i^2 = +1). Algoritmo de "bit-blade" estandar
   (Dorst, Fontijne, Mann, "Geometric Algebra for Computer Science", cap. 21). *)
let blade_sign a b =
  let rec loop a1 cnt =
    if a1 = 0 then cnt
    else loop (a1 lsr 1) (cnt + popcount (a1 land b))
  in
  let cnt = loop (a lsr 1) 0 in
  if cnt mod 2 = 0 then 1 else -1

let blade_index a b = a lxor b

let geometric_product (a : float array) (b : float array) : float array =
  let c = Array.make 8 0.0 in
  for i = 0 to 7 do
    if a.(i) <> 0.0 then
      for j = 0 to 7 do
        if b.(j) <> 0.0 then begin
          let s = blade_sign i j in
          let k = blade_index i j in
          c.(k) <- c.(k) +. (float_of_int s) *. a.(i) *. b.(j)
        end
      done
  done;
  c

let add_mv (a : float array) (b : float array) : float array =
  Array.init 8 (fun i -> a.(i) +. b.(i))

(* Etiqueta estatica de grado: EvenGrade significa "se puede demostrar que
   este nodo vive en span{1, e12, e13, e23}". AnyGrade es el caso general
   (sin garantia). *)
type grade_tag = AnyGrade | EvenGrade

type ast =
  | Const of float array * grade_tag
  | Var of string * grade_tag
  | Add of ast * ast
  | GProd of ast * ast
  | TruncEven of ast

(* Analisis estatico de grado: composicional, sin ejecutar nada.
   El caso GProd usa el hecho demostrado en verificacion_geometric_product.py:
   producto de dos elementos pares es par. *)
let rec tag_of (a : ast) : grade_tag =
  match a with
  | Const (_, t) -> t
  | Var (_, t) -> t
  | Add (l, r) ->
    (match tag_of l, tag_of r with
     | EvenGrade, EvenGrade -> EvenGrade
     | _ -> AnyGrade)
  | GProd (l, r) ->
    (match tag_of l, tag_of r with
     | EvenGrade, EvenGrade -> EvenGrade
     | _ -> AnyGrade)
  | TruncEven inner -> ignore inner; EvenGrade  (* por construccion, tras truncar SI es par *)

let rec fold (a : ast) : ast =
  match a with
  | Const _ | Var _ -> a
  | Add (l, r) ->
    let l' = fold l and r' = fold r in
    (match l', r' with
     | Const (ca, ta), Const (cb, tb) ->
       let t = if ta = EvenGrade && tb = EvenGrade then EvenGrade else AnyGrade in
       Const (add_mv ca cb, t)
     | _ -> Add (l', r'))
  | GProd (l, r) ->
    let l' = fold l and r' = fold r in
    (match l', r' with
     | Const (ca, ta), Const (cb, tb) ->
       let t = if ta = EvenGrade && tb = EvenGrade then EvenGrade else AnyGrade in
       Const (geometric_product ca cb, t)
     | _ -> GProd (l', r'))
  | TruncEven inner ->
    let inner' = fold inner in
    if tag_of inner' = EvenGrade then
      (* Provably safe: no se puede estar descartando nada, el resultado ya
         vive en la subalgebra par. Se elimina el truncamiento redundante. *)
      inner'
    else begin
      Printf.eprintf
        "warning: TruncEven no pudo probarse seguro en este nodo; \
         puede descartar componentes de grado 1 o 3 en tiempo de ejecucion\n";
      TruncEven inner'
    end

(* --------------------------------------------------------------------- *)
(* Demostracion minima (para que el alumno la corra):                    *)
(* --------------------------------------------------------------------- *)
let mv_of_index i coeff =
  let a = Array.make 8 0.0 in
  a.(i) <- coeff; a

let () =
  (* e12 * e13, ambos pares -> el resultado debe quedar en la subalgebra
     par y el pass debe poder elidir un TruncEven que lo envuelva. *)
  let e12 = Const (mv_of_index 3 1.0, EvenGrade) in
  let e13 = Const (mv_of_index 5 1.0, EvenGrade) in
  let expr_even = TruncEven (GProd (e12, e13)) in
  (match fold expr_even with
   | Const (c, EvenGrade) ->
     Printf.printf "Caso par: TruncEven elidido. e12*e13 = [%s] (tag=EvenGrade)\n"
       (String.concat "; " (Array.to_list (Array.map string_of_float c)))
   | _ -> Printf.printf "Caso par: resultado inesperado\n");

  (* e12 * e3, e3 no es par -> el pass NO debe elidir el truncamiento y debe
     avisar por stderr. *)
  let e3 = Const (mv_of_index 4 1.0, AnyGrade) in
  let expr_leak = TruncEven (GProd (e12, e3)) in
  (match fold expr_leak with
   | TruncEven (Const (c, AnyGrade)) ->
     Printf.printf "Caso fuga: TruncEven conservado (no elidido). Valor interno = [%s]\n"
       (String.concat "; " (Array.to_list (Array.map string_of_float c)))
   | _ -> Printf.printf "Caso fuga: resultado inesperado\n")
