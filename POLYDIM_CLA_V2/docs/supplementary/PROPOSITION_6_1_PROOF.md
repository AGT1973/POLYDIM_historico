# POLYDIM: Flutter Algebraic Isomorphism (Proposition 6.1 Proof)
**POPL 2027 Supplementary Material**

**Proposition 6.1.** The mapping $\phi: \mathcal{T} \to \mathcal{F}$ is a strict monoidal isomorphism between the POLYDIM transformation category ($\mathcal{T}$, `COMPOSE`, `MIX`) and the Flutter widget category ($\mathcal{F}$, `Column`, `Row`).

**Proof (Constructive):**

Let $\mathcal{T}$ be the category where objects are semantic geometries $R^N$ and morphisms are continuous linear operators $T: R^N \to R^N$. 
Let $\mathcal{F}$ be the category of Flutter UI trees where morphisms are Widget combinators.

We define the mapping $\phi$:
1. Base state: $\phi(\text{id}_{\mathcal{T}}) = \text{id}_{\mathcal{F}}$ (The empty widget `SizedBox.shrink()`).
2. Sequential Application (`COMPOSE`):
   $\phi(\text{COMPOSE}(T_1, T_2)) = \text{Column}(children: [\phi(T_1), \phi(T_2)])$
3. Superposition (`MIX`):
   $\phi(\text{MIX}(\alpha, T_1, \beta, T_2)) = \text{Row}(children: [\text{Opacity}(\alpha, \phi(T_1)), \text{Opacity}(\beta, \phi(T_2))])$
4. Recurrence (`FIXPOINT`):
   $\phi(\text{FIXPOINT}(T, \epsilon)) = \text{StatefulWidget}(\phi(T), \text{convergence\_threshold}: \epsilon)$

**Preservation of Monoidal Structure:**
For $(\mathcal{T}, \text{MIX}, 0)$, the tensor product is `MIX`.
$\phi(T_1 \otimes_{\mathcal{T}} T_2) = \text{Row}(\phi(T_1), \phi(T_2)) = \phi(T_1) \otimes_{\mathcal{F}} \phi(T_2)$.

Since $\phi$ is bijective on objects (via `PROJECT(DIM_FLUTTER)`) and preserves the monoidal product and composition, it is a strict monoidal functor, establishing the algebraic isomorphism. $\blacksquare$
