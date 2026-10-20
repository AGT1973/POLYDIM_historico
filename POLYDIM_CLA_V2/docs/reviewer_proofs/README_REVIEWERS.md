# POLYDIM-CLA: MATHEMATICAL PROOFS FOR REVIEWERS

This directory contains executable JAX scripts to empirically and mathematically prove the robustness of the POLYDIM-CLA architecture against the two primary concerns raised during the OpenRouter/Crucible peer-review process (POPL27 standards).

## 1. Woodbury Matrix Identity vs Cayley Transform Dense Inversion
**Reviewer Claim:** Applying the Woodbury identity to the Cayley Transform $Q = (I - S)(I + S)^{-1}$ is mathematically unsound because $(I + S)$ is dense ($\mathcal{O}(N^3)$), unless $S$ is explicitly low-rank.
**POLYDIM Resolution:** We explicitly define $S$ as a low-rank skew-symmetric update via Cartan-Dieudonné factorization: $S = U V^T - V U^T$. By defining $P = [U, -V]$ and $W = [V, U]$, we transform $I + S$ into $I + P W^T$, bounding the inversion strictly to $\mathcal{O}(K^3)$ where $K \ll N$.
**Execution:** Run `python cayley_woodbury_proof.py` to empirically verify that both inversions yield identical matrices up to BF16 precision, but the Woodbury approach scales at $\mathcal{O}(NK^2)$ instead of $\mathcal{O}(N^3)$.

## 2. Covariance of Cellular Sheaves
**Reviewer Claim:** Cellular sheaves are inherently contravariant functors. Treating them as covariant violates sheaf axioms.
**POLYDIM Resolution:** While sheaves on topological spaces are contravariant, **Cellular Sheaves on a face poset** are naturally covariant (cf. Curry, 2014). The restriction maps flow from cells of lower dimension to cells of higher dimension (e.g., node to edge, $\mathcal{F}(v) \to \mathcal{F}(e)$).
**Execution:** Run `python sheaf_covariance_proof.py` to verify the cohomological boundary operators and restriction maps structurally enforcing covariant information flow without violating the Hodge Laplacian $\Delta_{\mathcal{F}}$.

---
*Signed: AGY (Chief Architect) & Ariel H. Garcia Traba.*
