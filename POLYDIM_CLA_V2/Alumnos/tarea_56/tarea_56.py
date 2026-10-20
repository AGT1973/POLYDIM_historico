"""
tarea_56.py
Ingeniero de PyTorch Dynamo -- escalado asintotico de FLOPs y consumo
de VRAM/memoria en una cadena de bloques de atencion, usando el
volcado del grafo FX intermedio que produce Dynamo.

Requiere: torch >= 2.1 (para torch.utils.flop_counter.FlopCounterMode
y torch._dynamo.explain). Corre en CPU o GPU; en CPU la medicion de
memoria usa tracemalloc como aproximacion (ver notas en tarea_56.md).

Instalar dependencias:
    pip install torch numpy

Ejecutar:
    python tarea_56.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch._dynamo as dynamo
from torch.utils.flop_counter import FlopCounterMode


class AttentionBlock(nn.Module):
    """Un bloque estandar de self-attention + MLP (estilo Transformer)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, N, D = x.shape
        h = self.n_heads
        qkv = self.qkv(self.ln1(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, h, D // h).transpose(1, 2)
        k = k.view(B, N, h, D // h).transpose(1, 2)
        v = v.view(B, N, h, D // h).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (D // h) ** 0.5
        attn = scores.softmax(dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.out_proj(ctx)
        x = x + self.mlp(self.ln2(x))
        return x


class ChainedModel(nn.Module):
    """Cadena de L bloques de atencion, encadenados secuencialmente."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AttentionBlock(d_model, n_heads) for _ in range(n_layers)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def theoretical_flops(n: int, d_model: int, n_layers: int) -> float:
    """FLOPs teoricos por pasada de inferencia (orden dominante):
    4 n d^2 (proyecciones qkv + salida) + 2 n^2 d (scores + contexto)
    + 8 n d^2 (MLP con expansion x4), multiplicado por n_layers.
    """
    per_layer = 12 * n * d_model**2 + 2 * n**2 * d_model
    return n_layers * per_layer


def dump_intermediate_graph(model: nn.Module, x: torch.Tensor):
    """Vuelca el grafo FX intermedio que produce Dynamo al trazar el
    modelo, para inspeccionar como se fusiona/parte la cadena de
    operaciones y cuantos graph breaks ocurren."""
    explanation = dynamo.explain(model)(x)
    print(
        f"  -> graph_count={explanation.graph_count}, "
        f"graph_break_count={explanation.graph_break_count}"
    )
    if explanation.graphs:
        g = explanation.graphs[0].graph
        print("  -> primeras 10 filas del grafo FX intermedio:")
        for line in str(g).splitlines()[:10]:
            print("     " + line)
    return explanation


def measure_flops(model: nn.Module, x: torch.Tensor) -> float:
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            model(x)
    return fc.get_total_flops()


def measure_peak_memory(model: nn.Module, x: torch.Tensor, device: torch.device) -> int:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    else:
        import tracemalloc

        tracemalloc.start()
        with torch.no_grad():
            model(x)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model, n_heads, n_layers = 256, 8, 4
    seq_lengths = [128, 256, 512, 1024, 2048]

    model = ChainedModel(d_model, n_heads, n_layers).to(device).eval()

    print(f"Dispositivo: {device}")
    print("Volcado del grafo intermedio (Dynamo) para n=128:")
    x0 = torch.randn(1, 128, d_model, device=device)
    dump_intermediate_graph(model, x0)
    print()

    results = []
    for n in seq_lengths:
        x = torch.randn(1, n, d_model, device=device)
        flops = measure_flops(model, x)
        mem = measure_peak_memory(model, x, device)
        theo = theoretical_flops(n, d_model, n_layers)
        results.append((n, flops, theo, mem))
        print(
            f"n={n:5d}  FLOPs_medidos={flops:.3e}  "
            f"FLOPs_teoricos={theo:.3e}  memoria_pico(bytes)={mem}"
        )

    ns = np.array([r[0] for r in results], dtype=float)
    flops_arr = np.array([r[1] for r in results], dtype=float)
    mem_arr = np.array([r[3] for r in results], dtype=float)

    # Ajuste log-log: la pendiente es el exponente empirico de escalado.
    flops_slope, _ = np.polyfit(np.log(ns), np.log(flops_arr), 1)
    mem_slope, _ = np.polyfit(np.log(ns), np.log(mem_arr), 1)

    print()
    print(f"Exponente empirico FLOPs vs n (pendiente log-log): {flops_slope:.3f}")
    print(f"Exponente empirico memoria vs n (pendiente log-log): {mem_slope:.3f}")
    print("Teorico esperado (n >> d, dominado por atencion): ~2.0 en ambos casos")


if __name__ == "__main__":
    run_experiment()
