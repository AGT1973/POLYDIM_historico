# GPU Hardware Limits: Underflow, Scaling, and FP8/FP16 Arithmetic

## 1. Overview
Understanding the exact hardware limits of GPUs (like NVIDIA H100/B200) is crucial for translating continuous theoretical models (like those involving Absolute Continuity) into executable numerical models.

## 2. Low-Precision FP8 Formats on GPUs

Modern GPUs implement two primary FP8 formats, each tailored for different phases of numerical computation:
*   **E4M3 (4 exponent bits, 3 mantissa bits)**
    *   Optimized for: Forward pass (activations and weights).
    *   Max value: 448.0
    *   Min positive normal: $2^{-6} = 0.015625$
    *   Subnormals supported: Yes, down to $2^{-9} \approx 0.00195$
*   **E5M2 (5 exponent bits, 2 mantissa bits)**
    *   Optimized for: Backward pass (gradients/derivatives).
    *   Max value: 57344.0
    *   Min positive normal: $2^{-14} \approx 6.1 \times 10^{-5}$
    *   Subnormals supported: Yes, down to $2^{-16} \approx 1.5 \times 10^{-5}$

## 3. The Underflow Crisis in Radon-Nikodym Evaluation
In probability theory, the Radon-Nikodym derivative $f(x) = \frac{p(x)}{q(x)}$ is commonly used in Importance Sampling and Kullback-Leibler Divergence.
*   **Tail Probabilities**: If evaluating the tails of a Gaussian distribution, $p(x)$ decays exponentially $\propto e^{-x^2}$.
*   **Hardware Flush-to-Zero (FTZ)**: When $f(x)$ drops below $1.5 \times 10^{-5}$ (in E5M2), the GPU hardware FTZ mechanism zeroes out the value.
*   **Consequence**: The algorithm completely loses the gradient signal from the tails, meaning the optimizer assumes the regions have zero probability. This violates the absolute continuity assumption required for convergence proofs of algorithms like Stochastic Gradient Langevin Dynamics (SGLD) or normalizing flows.

## 4. Hardware-Level Scaling Solutions
To map the infinite range of theoretical derivatives into the finite, narrow band of FP8:
1.  **Tensor Core Scaling**: NVIDIA Hopper architecture introduces tensor cores that accept scaling factors directly at the hardware level. The mathematical operation becomes $C = (A \cdot S_A) \times (B \cdot S_B) + C$.
2.  **Delayed Scaling**: Calculating the maximum absolute value of a tensor, delaying the scaling update by one step to overlap computation and scaling factor communication across GPU clusters, effectively predicting the necessary scaling window for the continuous derivative based on its local Lipschitz continuity.
