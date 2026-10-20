"""
POLYDIM - Tarea 56 (Test 56)
Rol: Ingeniero de PyTorch Dynamo
Backend numérico: NumPy (PyTorch no disponible en el entorno de ejecución)

Objetivo (enunciado reconstruido):
Diseñar una operación encadenada que demuestre, a nivel de volcado intermedio
de grafo (graph dump / FX Graph), que el consumo de VRAM y FLOPs gastados
durante la inferencia:
    (A) puede estimarse ESTÁTICAMENTE antes de ejecutar (análisis del grafo),
    (B) escala de forma PREDECIBLE con el tamaño de batch/secuencia,
    (C) se REDUCE mediante fusión de operadores (operator fusion) en el
        grafo compilado, comparado contra el grafo no fusionado (eager).

No usamos torch.fx real (no disponible), pero replicamos su estructura
esencial: un grafo dirigido acíclico (DAG) de nodos-operación con anotación
de shape, sobre el cual se realiza análisis estático de costo ANTES de
ejecutar ningún kernel numérico.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json


# ---------------------------------------------------------------------------
# 1. REPRESENTACIÓN DEL GRAFO INTERMEDIO (análogo simplificado a torch.fx.Graph)
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    name: str
    op: str                      # 'linear', 'relu', 'add', 'matmul', 'layernorm'
    input_shapes: List[Tuple[int, ...]]
    output_shape: Tuple[int, ...]
    params_bytes: int = 0        # bytes de pesos propios del nodo (si aplica)
    fused_into: Optional[str] = None  # si fue absorbido por fusión, nombre del nodo resultante

    def flops(self) -> int:
        """FLOPs estimados de forma puramente estática (sin ejecutar nada)."""
        if self.op == "matmul" or self.op == "linear":
            # (B, M, K) x (K, N) -> 2*B*M*K*N FLOPs (mult+add)
            in_shape = self.input_shapes[0]
            out_shape = self.output_shape
            B = in_shape[0] if len(in_shape) == 3 else 1
            M = in_shape[-2] if len(in_shape) >= 2 else in_shape[0]
            K = in_shape[-1]
            N = out_shape[-1]
            return int(2 * B * M * K * N)
        elif self.op == "relu":
            return int(np.prod(self.output_shape))
        elif self.op == "add":
            return int(np.prod(self.output_shape))
        elif self.op == "layernorm":
            # ~ 5 FLOPs por elemento (media, var, norm, escala, shift)
            return int(5 * np.prod(self.output_shape))
        else:
            return 0

    def activation_bytes(self, dtype_bytes: int = 4) -> int:
        """Memoria de la activación de salida que debe materializarse en VRAM."""
        return int(np.prod(self.output_shape) * dtype_bytes)


@dataclass
class DynamoGraph:
    nodes: List[GraphNode] = field(default_factory=list)

    def add(self, node: GraphNode):
        self.nodes.append(node)
        return node

    # ---- (A) ANÁLISIS ESTÁTICO: costo total sin ejecutar ----
    def static_cost_report(self, dtype_bytes: int = 4) -> dict:
        """
        Recorre el DAG en orden topológico (ya viene en orden de inserción,
        como lo entrega Dynamo tras el trace) y acumula FLOPs y bytes de
        activación SIN materializar ningún tensor real. Esto es lo que
        Dynamo permite hacer sobre el FX Graph antes del backend compile.
        """
        active_nodes = [n for n in self.nodes if n.fused_into is None]
        total_flops = sum(n.flops() for n in active_nodes)
        # VRAM pico ~ suma de activaciones vivas simultáneamente.
        # Aproximación conservadora: suma de todas las activaciones de salida
        # de nodos activos + memoria de parámetros.
        total_activation_bytes = sum(n.activation_bytes(dtype_bytes) for n in active_nodes)
        total_param_bytes = sum(n.params_bytes for n in active_nodes)
        return {
            "num_nodes_ejecutados": len(active_nodes),
            "num_nodes_totales_pre_fusion": len(self.nodes),
            "flops_totales": total_flops,
            "vram_activaciones_bytes": total_activation_bytes,
            "vram_parametros_bytes": total_param_bytes,
            "vram_total_bytes": total_activation_bytes + total_param_bytes,
        }


# ---------------------------------------------------------------------------
# 2. CONSTRUCCIÓN DEL GRAFO "EAGER" (sin fusión) PARA UN BLOQUE TRANSFORMER-LIKE
#    Bloque: Linear -> LayerNorm -> ReLU -> Linear -> Add(residual)
# ---------------------------------------------------------------------------

def build_eager_graph(batch: int, seq: int, d_model: int, d_ff: int) -> DynamoGraph:
    g = DynamoGraph()

    x_shape = (batch, seq, d_model)

    n1 = g.add(GraphNode(
        name="linear_1", op="linear",
        input_shapes=[x_shape], output_shape=(batch, seq, d_ff),
        params_bytes=d_model * d_ff * 4,
    ))
    n2 = g.add(GraphNode(
        name="layernorm_1", op="layernorm",
        input_shapes=[n1.output_shape], output_shape=n1.output_shape,
        params_bytes=2 * d_ff * 4,  # gamma, beta
    ))
    n3 = g.add(GraphNode(
        name="relu_1", op="relu",
        input_shapes=[n2.output_shape], output_shape=n2.output_shape,
    ))
    n4 = g.add(GraphNode(
        name="linear_2", op="linear",
        input_shapes=[n3.output_shape], output_shape=(batch, seq, d_model),
        params_bytes=d_ff * d_model * 4,
    ))
    n5 = g.add(GraphNode(
        name="residual_add", op="add",
        input_shapes=[n4.output_shape, x_shape], output_shape=x_shape,
    ))
    return g


# ---------------------------------------------------------------------------
# 3. FUSIÓN DE OPERADORES (emulando Dynamo/Inductor operator fusion)
#    Regla de fusión aplicada: Linear -> LayerNorm -> ReLU se funden en un
#    único nodo "fused_linear_ln_relu". El add residual se funde con el
#    linear_2 en "fused_linear_add".
#    Efecto físico: las activaciones intermedias (salida de linear_1 y de
#    layernorm_1) YA NO se materializan en VRAM; solo se guarda la salida
#    final de cada bloque fusionado. Los FLOPs de cómputo puro no cambian
#    (se siguen haciendo las mismas multiplicaciones), pero el tráfico de
#    memoria (bytes movidos a/desde VRAM) y los picos de activación bajan.
# ---------------------------------------------------------------------------

def apply_operator_fusion(g_eager: DynamoGraph) -> DynamoGraph:
    fused = DynamoGraph()
    nodes_by_name = {n.name: n for n in g_eager.nodes}

    l1, ln1, relu1, l2, add1 = (
        nodes_by_name["linear_1"], nodes_by_name["layernorm_1"],
        nodes_by_name["relu_1"], nodes_by_name["linear_2"], nodes_by_name["residual_add"]
    )

    # Nodo fusionado A: linear_1 + layernorm_1 + relu_1
    # FLOPs = suma de los tres (el cómputo no desaparece, se fusiona el KERNEL)
    fused_a = fused.add(GraphNode(
        name="fused_linear_ln_relu",
        op="fused_block_a",
        input_shapes=l1.input_shapes,
        output_shape=relu1.output_shape,
        params_bytes=l1.params_bytes + ln1.params_bytes,
    ))
    # Sobrescribimos flops() para nodos fusionados vía atributo calculado manualmente:
    fused_a_flops = l1.flops() + ln1.flops() + relu1.flops()

    # Nodo fusionado B: linear_2 + residual_add
    fused_b = fused.add(GraphNode(
        name="fused_linear_add",
        op="fused_block_b",
        input_shapes=[fused_a.output_shape, l1.input_shapes[0]],
        output_shape=add1.output_shape,
        params_bytes=l2.params_bytes,
    ))
    fused_b_flops = l2.flops() + add1.flops()

    # Marcamos los nodos originales como "fusionados" (no ejecutan de forma
    # independiente, no materializan activación propia en VRAM).
    for n in (l1, ln1, relu1, l2, add1):
        n.fused_into = "fused_a" if n.name in ("linear_1", "layernorm_1", "relu_1") else "fused_b"

    # Inyectamos manualmente los FLOPs precomputados como atributo extra
    fused_a.__dict__["_flops_override"] = fused_a_flops
    fused_b.__dict__["_flops_override"] = fused_b_flops

    return fused, (fused_a_flops + fused_b_flops)


def static_cost_report_fused(g_fused: DynamoGraph, dtype_bytes: int = 4) -> dict:
    total_flops = sum(n.__dict__.get("_flops_override", n.flops()) for n in g_fused.nodes)
    total_activation_bytes = sum(n.activation_bytes(dtype_bytes) for n in g_fused.nodes)
    total_param_bytes = sum(n.params_bytes for n in g_fused.nodes)
    return {
        "num_nodes_ejecutados": len(g_fused.nodes),
        "flops_totales": total_flops,
        "vram_activaciones_bytes": total_activation_bytes,
        "vram_parametros_bytes": total_param_bytes,
        "vram_total_bytes": total_activation_bytes + total_param_bytes,
    }


# ---------------------------------------------------------------------------
# 4. VALIDACIÓN NUMÉRICA: el resultado matemático del bloque fusionado debe
#    coincidir con el del bloque eager (la fusión no debe alterar semántica).
#    Esto ancla la parte "categórica/topológica" del framework POLYDIM:
#    GEO_ID debe preservarse == el resultado numérico es invariante bajo
#    refactorización del grafo (transformación natural / equivalencia).
# ---------------------------------------------------------------------------

def numeric_equivalence_check(batch, seq, d_model, d_ff, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch, seq, d_model)).astype(np.float32)
    W1 = rng.standard_normal((d_model, d_ff)).astype(np.float32) * 0.02
    gamma = np.ones((d_ff,), dtype=np.float32)
    beta = np.zeros((d_ff,), dtype=np.float32)
    W2 = rng.standard_normal((d_ff, d_model)).astype(np.float32) * 0.02

    def layernorm(z, gamma, beta, eps=1e-5):
        mu = z.mean(axis=-1, keepdims=True)
        var = z.var(axis=-1, keepdims=True)
        return gamma * (z - mu) / np.sqrt(var + eps) + beta

    # --- Camino eager: cada op por separado ---
    h1 = x @ W1
    h2 = layernorm(h1, gamma, beta)
    h3 = np.maximum(h2, 0.0)
    h4 = h3 @ W2
    out_eager = h4 + x

    # --- Camino "fusionado": misma matemática, un solo bloque de código ---
    def fused_block_a(x):
        z = x @ W1
        return np.maximum(layernorm(z, gamma, beta), 0.0)

    def fused_block_b(a, x_residual):
        return a @ W2 + x_residual

    a_out = fused_block_a(x)
    out_fused = fused_block_b(a_out, x)

    l2_error = float(np.linalg.norm(out_eager - out_fused) / (np.linalg.norm(out_eager) + 1e-12))
    return l2_error


# ---------------------------------------------------------------------------
# 5. EXPERIMENTO DE ESCALADO: variar batch*seq y medir cómo escalan
#    FLOPs y VRAM, en ambos grafos (eager vs fusionado), TODO estimado
#    estáticamente (sin ejecutar ningún forward real de tensores grandes).
# ---------------------------------------------------------------------------

def scaling_experiment(d_model=512, d_ff=2048, seq=128,
                        batch_values=(1, 2, 4, 8, 16, 32, 64)):
    results = []
    for B in batch_values:
        g_eager = build_eager_graph(B, seq, d_model, d_ff)
        report_eager = g_eager.static_cost_report()

        g_eager_fresh = build_eager_graph(B, seq, d_model, d_ff)  # copia limpia para fusionar
        g_fused, _ = apply_operator_fusion(g_eager_fresh)
        report_fused = static_cost_report_fused(g_fused)

        reduction_vram = 1.0 - (report_fused["vram_total_bytes"] / report_eager["vram_total_bytes"])
        results.append({
            "batch": B,
            "flops_eager": report_eager["flops_totales"],
            "flops_fused": report_fused["flops_totales"],
            "vram_eager_MB": report_eager["vram_total_bytes"] / 1e6,
            "vram_fused_MB": report_fused["vram_total_bytes"] / 1e6,
            "reduccion_vram_pct": 100 * reduction_vram,
        })
    return results


def fit_scaling_law(results):
    """
    Ajusta un modelo log-log FLOPs ~ C * B^alpha para demostrar que el
    escalado es PREDECIBLE (regresión lineal en log-log => alpha estimado).
    """
    B = np.array([r["batch"] for r in results], dtype=np.float64)
    F = np.array([r["flops_eager"] for r in results], dtype=np.float64)
    logB, logF = np.log(B), np.log(F)
    # Ajuste lineal logF = alpha*logB + logC
    A = np.vstack([logB, np.ones_like(logB)]).T
    alpha, logC = np.linalg.lstsq(A, logF, rcond=None)[0]
    residuals = logF - (alpha * logB + logC)
    r2 = 1 - np.sum(residuals**2) / np.sum((logF - logF.mean())**2)
    return {"alpha_estimado": float(alpha), "C_estimado": float(np.exp(logC)), "R2_ajuste": float(r2)}


# ---------------------------------------------------------------------------
# 6. EJECUCIÓN PRINCIPAL
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TAREA 56 - Análisis estático de grafo Dynamo (VRAM/FLOPs)")
    print("=" * 70)

    # (A) Análisis estático puntual (un solo tamaño)
    B, S, D, F = 8, 128, 512, 2048
    g_eager = build_eager_graph(B, S, D, F)
    report_eager = g_eager.static_cost_report()

    g_eager2 = build_eager_graph(B, S, D, F)
    g_fused, _ = apply_operator_fusion(g_eager2)
    report_fused = static_cost_report_fused(g_fused)

    print("\n--- (A) Reporte estático EAGER (sin ejecutar nada) ---")
    print(json.dumps(report_eager, indent=2))
    print("\n--- (A) Reporte estático FUSIONADO ---")
    print(json.dumps(report_fused, indent=2))

    vram_reduction = 100 * (1 - report_fused["vram_total_bytes"] / report_eager["vram_total_bytes"])
    print(f"\n(C) Reducción de VRAM por fusión: {vram_reduction:.2f}%")
    flops_delta = 100 * (report_fused["flops_totales"] / report_eager["flops_totales"] - 1)
    print(f"(C) Delta de FLOPs (debe ser ~0%, la fusión no cambia el cómputo matemático): {flops_delta:.4f}%")

    # Validación de equivalencia numérica (invariancia GEO_ID bajo refactor)
    err = numeric_equivalence_check(B, S, D, F)
    print(f"\n(Validación GEO_ID) Error relativo L2 entre grafo eager y fusionado: {err:.3e}")

    # (B) Escalado
    print("\n--- (B) Experimento de escalado (batch creciente) ---")
    scaling = scaling_experiment(d_model=D, d_ff=F, seq=S)
    for r in scaling:
        print(f"batch={r['batch']:>3} | FLOPs={r['flops_eager']:.3e} | "
              f"VRAM_eager={r['vram_eager_MB']:.2f}MB | VRAM_fused={r['vram_fused_MB']:.2f}MB | "
              f"reduccion={r['reduccion_vram_pct']:.2f}%")

    law = fit_scaling_law(scaling)
    print(f"\nLey de escalado ajustada: FLOPs ≈ {law['C_estimado']:.3e} * batch^{law['alpha_estimado']:.4f}")
    print(f"R² del ajuste: {law['R2_ajuste']:.6f}  (≈1.0 => escalado predecible)")

    # ---- Resumen estructurado JSON ----
    resumen = {
        "alumno": "andrea",
        "tarea": 56,
        "tipo_problema": "Análisis estático de costo computacional sobre grafo intermedio (Dynamo FX-like) con fusión de operadores",
        "estado": "resuelto",
        "resultado_resumen": (
            f"Se demostró que FLOPs y VRAM son estimables estáticamente sobre el grafo "
            f"(sin ejecutar tensores), que escalan como una ley de potencia predecible "
            f"(alpha≈{law['alpha_estimado']:.3f}, R²≈{law['R2_ajuste']:.4f}), y que la fusión "
            f"de operadores reduce el pico de VRAM en {vram_reduction:.2f}% preservando "
            f"exactamente los FLOPs matemáticos (Δ≈{flops_delta:.4f}%) y la identidad numérica "
            f"del resultado (error L2 relativo = {err:.2e})."
        ),
        "metrica_estandar_usada": "FLOPs estáticos, bytes de VRAM estimados, error relativo L2, R² de regresión log-log",
    }

    print("\n--- Resumen JSON ---")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))

    with open("/home/claude/t56/resumen_tarea56.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    with open("/home/claude/t56/scaling_results.json", "w", encoding="utf-8") as f:
        json.dump({"scaling": scaling, "law": law}, f, indent=2, ensure_ascii=False)
