"""
Tarea 16 - Kernel Triton ultra-optimizado para el paso costoso de la
descomposicion de Hodge: el producto Laplaciano-vector L0 @ s.

IMPORTANTE (leer antes de correr):
-----------------------------------
Este archivo REQUIERE una GPU NVIDIA con CUDA y el paquete `triton`
instalado (`pip install triton`, solo funciona sobre CUDA). No puede
ejecutarse en un entorno sin GPU. Por eso NO es el archivo principal de
validacion: la logica matematica ya fue comprobada de forma independiente
de Triton en `hodge_reference.py` / `validar_hodge.py` (que corren en
cualquier maquina, con NumPy/SciPy puro).

Por que el kernel ataca justamente esta operacion
--------------------------------------------------
Aislar la componente armonica requiere resolver L0 s = B^T phi. Para
grafos grandes esto se hace con metodos iterativos (ej. Conjugate
Gradient), donde el costo dominante por iteracion es el producto
Laplaciano-vector L0 @ s = (D - A) @ s, que en formato de lista de
aristas (edge list) es un patron scatter-add:

    out[i] += s[i] - s[j]   para cada arista (i, j)
    out[j] += s[j] - s[i]

Esto es memory-bound (poco computo por byte leido), exactamente el tipo
de operacion que se beneficia de un kernel Triton a medida en vez de
depender de un framework generico: se puede fusionar la lectura de
`s[i]`, `s[j]` y la escritura atomica en un solo kernel, evitando
materializar la matriz dispersa completa.

Diseno del kernel
------------------
- Un programa (bloque) procesa BLOCK_SIZE aristas.
- Cada arista contribuye con signo opuesto a sus dos vertices.
- Se usa `tl.atomic_add` porque varias aristas pueden escribir sobre el
  mismo vertice en paralelo (conflicto de escritura).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _laplacian_edge_kernel(
    src_ptr, dst_ptr,      # indices de vertices de cada arista (int32)
    s_ptr,                 # vector de entrada, tamaño n_nodes
    out_ptr,               # vector de salida (acumulador), tamaño n_nodes
    n_edges,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_edges

    i = tl.load(src_ptr + offsets, mask=mask, other=0)
    j = tl.load(dst_ptr + offsets, mask=mask, other=0)

    s_i = tl.load(s_ptr + i, mask=mask, other=0.0)
    s_j = tl.load(s_ptr + j, mask=mask, other=0.0)

    diff = s_i - s_j  # contribucion de (i,j) al Laplaciano: (D - A) s

    tl.atomic_add(out_ptr + i, diff, mask=mask)
    tl.atomic_add(out_ptr + j, -diff, mask=mask)


def laplacian_matvec_triton(
    src: torch.Tensor,
    dst: torch.Tensor,
    s: torch.Tensor,
    n_nodes: int,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    Calcula out = L0 @ s = (D - A) @ s usando el kernel Triton de arriba.
    src, dst: tensores int32 de tamaño n_edges (indices de cada arista).
    s: tensor float32 de tamaño n_nodes.
    Todos los tensores deben estar en GPU (device='cuda').
    """
    assert src.is_cuda and dst.is_cuda and s.is_cuda, "Este kernel requiere tensores en GPU (CUDA)"
    n_edges = src.shape[0]
    out = torch.zeros(n_nodes, device=s.device, dtype=torch.float32)
    grid = (triton.cdiv(n_edges, block_size),)
    _laplacian_edge_kernel[grid](
        src, dst, s, out, n_edges, BLOCK_SIZE=block_size
    )
    return out


def conjugate_gradient_hodge(
    src: torch.Tensor,
    dst: torch.Tensor,
    rhs: torch.Tensor,
    n_nodes: int,
    ref_node: int = 0,
    n_iters: int = 200,
    tol: float = 1e-8,
) -> torch.Tensor:
    """
    Resuelve L0 s = rhs (con el gauge s[ref_node]=0 impuesto por
    proyeccion en cada paso) usando CG, llamando al kernel Triton para
    cada producto Laplaciano-vector. Pensado para grafos grandes donde
    la version densa/SciPy de hodge_reference.py deja de ser practica.
    """
    device = rhs.device
    s = torch.zeros(n_nodes, device=device, dtype=torch.float32)

    def apply_L(v):
        v = v.clone()
        v[ref_node] = 0.0
        out = laplacian_matvec_triton(src, dst, v, n_nodes)
        out[ref_node] = 0.0
        return out

    r = rhs.clone()
    r[ref_node] = 0.0
    p = r.clone()
    rs_old = torch.dot(r, r)

    for _ in range(n_iters):
        Ap = apply_L(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        s = s + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return s


if __name__ == "__main__":
    # Ejemplo minimo de uso (requiere GPU CUDA disponible)
    if not torch.cuda.is_available():
        raise SystemExit(
            "No hay GPU CUDA disponible en este entorno. "
            "Corre este archivo en una maquina con GPU NVIDIA + triton instalado."
        )

    device = "cuda"
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (3, 5)]
    n_nodes = 6
    src = torch.tensor([e[0] for e in edges], dtype=torch.int32, device=device)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.int32, device=device)

    torch.manual_seed(16)
    phi = torch.randn(len(edges), device=device)

    # B^T phi calculado directamente (para comparar contra el kernel)
    rhs = torch.zeros(n_nodes, device=device)
    rhs.index_add_(0, dst.long(), phi)
    rhs.index_add_(0, src.long(), -phi)

    s = conjugate_gradient_hodge(src, dst, rhs, n_nodes)
    print("Potencial s recuperado:", s.cpu().numpy())
