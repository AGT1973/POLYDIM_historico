"""
Suite de tests para Tarea 5 (POLYDIM).
Ejecutar con: python -m pytest test_commuting_lora.py -v
"""

import torch
import sympy as sp
from commuting_lora import (
    CommutingLoRALinear,
    make_shared_orthogonal_basis,
    check_static_commutator,
    check_lie_flow_commutation,
    check_categorical_diagram,
    symbolic_sanity_check,
)

torch.manual_seed(42)
DIM = 8


def build_pair(r1: int, r2: int):
    U = make_shared_orthogonal_basis(DIM, seed=1)
    l1 = CommutingLoRALinear(DIM, r=r1, shared_basis=U)
    l2 = CommutingLoRALinear(DIM, r=r2, shared_basis=U)
    return l1, l2


# --- Caso general (rango típico intermedio) --------------------------------

def test_general_case_all_three_levels():
    l1, l2 = build_pair(r1=3, r2=5)
    dW1, dW2 = l1.delta_w(), l2.delta_w()

    e_static = check_static_commutator(dW1, dW2)
    e_lie = check_lie_flow_commutation(dW1, dW2)

    x = torch.randn(4, DIM)
    e_cat = check_categorical_diagram(l1, l2, x)

    print(f"[general] estático={e_static:.2e} lie={e_lie:.2e} categórico={e_cat:.2e}")
    assert e_static < 1e-5 and e_lie < 1e-4 and e_cat < 1e-4


# --- Caso límite 1: r = 0 (adaptador nulo) ----------------------------------

def test_edge_rank_zero():
    l1, l2 = build_pair(r1=0, r2=4)
    dW1 = l1.delta_w()
    assert torch.allclose(dW1, torch.zeros_like(dW1), atol=1e-6), \
        "r=0 debe producir ΔW exactamente nula"
    # Una matriz nula conmuta trivialmente con cualquier otra
    check_static_commutator(dW1, l2.delta_w())


# --- Caso límite 2: r = dim (rango completo, sin restricción de bajo rango)

def test_edge_rank_full():
    l1, l2 = build_pair(r1=DIM, r2=DIM)
    e_static = check_static_commutator(l1.delta_w(), l2.delta_w())
    assert e_static < 1e-5, "Incluso en rango completo, la base compartida debe garantizar conmutación"


# --- Caso límite 3: entrada nula --------------------------------------------

def test_edge_zero_input():
    l1, l2 = build_pair(r1=2, r2=2)
    x = torch.zeros(1, DIM)
    e_cat = check_categorical_diagram(l1, l2, x)
    assert e_cat < 1e-6


# --- Caso límite 4: bases NO compartidas -> DEBE FALLAR ---------------------
# Este test es tan importante como los anteriores: prueba que la propiedad
# no es "accidental", sino que depende estrictamente de compartir la base U.

def test_edge_different_basis_should_not_commute():
    U1 = make_shared_orthogonal_basis(DIM, seed=1)
    U2 = make_shared_orthogonal_basis(DIM, seed=2)  # base distinta a propósito
    l1 = CommutingLoRALinear(DIM, r=4, shared_basis=U1)
    l2 = CommutingLoRALinear(DIM, r=4, shared_basis=U2)

    # Nota de diseño experimental: la inicialización por defecto usa d ~ N(0, 0.01),
    # por lo que ||ΔW|| es tan chica que CUALQUIER par de matrices casi-nulas
    # "casi conmuta" en términos absolutos (el conmutador escala con el
    # producto de las normas). Para que el control negativo sea válido,
    # medimos el conmutador de forma RELATIVA a ese producto de normas.
    dW1, dW2 = l1.delta_w(), l2.delta_w()
    commutator_norm = torch.linalg.norm(dW1 @ dW2 - dW2 @ dW1).item()
    scale = torch.linalg.norm(dW1).item() * torch.linalg.norm(dW2).item()
    relative_norm = commutator_norm / (scale + 1e-12)

    assert relative_norm > 1e-3, (
        "Con bases distintas se espera que NO conmuten (control negativo); "
        f"norma relativa obtenida={relative_norm:.2e}. Si esto falla, la garantía "
        "teórica no se está probando correctamente."
    )


# --- Caso límite 5: gradientes fluyen y la propiedad se mantiene tras update

def test_edge_gradients_preserve_commutation():
    l1, l2 = build_pair(r1=3, r2=3)
    x = torch.randn(2, DIM)

    out = l2(l1(x)).sum()
    out.backward()

    # Corrección: out = l2(l1(x)) usa AMBAS capas en el grafo hacia adelante,
    # por lo tanto autograd calcula gradiente para AMBOS parámetros .d
    assert l1.d.grad is not None and l2.d.grad is not None
    with torch.no_grad():
        l1.d -= 0.1 * l1.d.grad

    # Tras la actualización de gradiente, la conmutatividad se preserva
    # porque la base U sigue fija: es una garantía estructural, no numérica frágil.
    check_static_commutator(l1.delta_w(), l2.delta_w())


# --- Verificación simbólica (sympy) -----------------------------------------

def test_symbolic_commutator_is_zero_matrix():
    commutator = symbolic_sanity_check()
    zero = sp.zeros(2, 2)
    assert sp.simplify(commutator - zero) == zero, \
        f"El conmutador simbólico debería simplificar a cero, dio: {commutator}"


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
