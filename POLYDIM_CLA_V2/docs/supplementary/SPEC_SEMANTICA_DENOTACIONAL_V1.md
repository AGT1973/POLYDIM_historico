# POLYDIM: Denotational Semantics in Para(Vect) (V1.0)
**POPL 2027 Supplementary Material**

This document provides the denotational semantics of POLYDIM, mapping geometric programs to the 2-category $Para(Vect)$ of parameterized differentiable maps, and integrating quantum resource calculus (Kegelspitzen) and Reverse Differential Categories (RDC).

## 1. Functorial Mapping to Para(Vect)

A POLYDIM program $T$ is denoted as a morphism in $Para(Vect)$.
$$ [\!| T |\!] : \mathbb{R}^W \times \mathbb{R}^N \to \mathbb{R}^N $$
Where $\mathbb{R}^W$ is the parameter space (weights) and $\mathbb{R}^N$ is the latent space.

### 1.1 Composition in Para(Vect)
Given $[\!| T_1 |\!](w_1, x)$ and $[\!| T_2 |\!](w_2, x)$, their composition is:
$$ [\!| \text{COMPOSE}(T_1, T_2) |\!] (w_1 \otimes w_2, x) = [\!| T_2 |\!](w_2, [\!| T_1 |\!](w_1, x)) $$
This preserves the monoidal structure of the parameter space.

## 2. Gradient Flow via Reverse Differential Categories (RDC)

To address Gap 22 (inefficiency of Cartesian Differential Categories for high-dimensional gradient descent), POLYDIM's denotational semantics adopt **Reverse Differential Categories (RDC)**.

The reverse derivative combinator $\text{R}[f]: A \times B \to A$ provides the exact semantics for backpropagation:
$$ \text{R}[\text{FIXPOINT}(T)](s_0, \epsilon) = \text{FIXPOINT}(\lambda g. \text{R}[T](s^*, g) + \epsilon) $$
This formalization guarantees that the learning dynamics (gradient flow) remain functorially tied to the forward pass, with $O(N \cdot r)$ complexity under the LoRA format.

## 3. Cost Semantics via qet-calculus and Kegelspitzen

To address Gap 26 (quantum/resource cost explosion), POLYDIM integrates **qet-calculus**.
The expected execution cost is not a scalar, but an element of a **Kegelspitze** (a directed topological cone):
$$ \mathcal{C}([\!| T |\!]) \in \mathcal{K}_{cost} $$

### 3.1 Bounded Completeness
By enforcing the topological monitor threshold `algebra_tolerance_epsilon` (see Binary Format Spec), we ensure that the execution depth bounds are preserved during continuous transformations, avoiding undecidability in recursive algebraic generation.

## 4. The Gauss-Markov Adjunction
Following Kamiura (2025), the learning process in POLYDIM is formally denoted as a categorical adjunction between the hypothesis space and the residual space. The convergence of the `FIXPOINT` operator over training epochs represents the computation of the categorical limit in this adjunction.
