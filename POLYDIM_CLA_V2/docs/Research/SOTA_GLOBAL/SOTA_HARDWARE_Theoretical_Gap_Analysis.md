# Gap Analysis: Absolute Continuity vs. Low-Precision Arithmetic

## 1. Introduction
This document analyzes the fundamental theoretical gap between the continuous mathematics of measure theory—specifically Absolute Continuity and Radon-Nikodym derivatives—and the discrete, highly constrained numerical representations used in modern deep learning hardware, namely FP8 and FP16 on GPUs.

## 2. Theoretical Foundations
### 2.1 Absolute Continuity and Radon-Nikodym Derivatives
In measure theory, a measure $\mu$ is absolutely continuous with respect to another measure $\nu$ (denoted $\mu \ll \nu$) if $\nu(A) = 0$ implies $\mu(A) = 0$ for every measurable set $A$. The Radon-Nikodym theorem states that if $\mu \ll \nu$ and both are $\sigma$-finite, there exists a measurable function $f$ (the Radon-Nikodym derivative, $f = \frac{d\mu}{d\nu}$) such that:
$$ \mu(A) = \int_A f d\nu $$
The density $f$ can take arbitrarily small positive values (e.g., in the tails of probability distributions) and arbitrarily large values.

### 2.2 Low-Precision Arithmetic Limits (FP16 / FP8)
Modern GPUs (like NVIDIA Hopper architecture) heavily rely on FP16, BF16, and FP8 formats for peak throughput. 
*   **FP16**: 1 sign bit, 5 exponent bits, 10 mantissa bits. Dynamic range: $\approx 5.96 \times 10^{-8}$ to $6.55 \times 10^4$.
*   **FP8 (E4M3)**: 1 sign, 4 exponent, 3 mantissa. Max value: 448. Min positive normal: $2^{-6} = 0.015625$.
*   **FP8 (E5M2)**: 1 sign, 5 exponent, 2 mantissa. Max value: 57344. Min positive normal: $2^{-14} \approx 6.1 \times 10^{-5}$.

## 3. The Theoretical Gap
### 3.1 Breakdown of Absolute Continuity
When representing a Radon-Nikodym derivative $f$ in low precision $\tilde{f}$, values below the underflow limit (e.g., $6.1 \times 10^{-5}$ in E5M2) are flushed to zero. 
This creates a critical violation of absolute continuity:
There can exist sets $A$ where $\nu(A) > 0$ and theoretically $\mu(A) > 0$, but because $\forall x \in A, f(x) < \text{MIN\_NORMAL}$, the computed derivative $\tilde{f}(x) = 0$. Consequently, the computed measure $\tilde{\mu}(A) = 0$. The computed measure is no longer absolutely continuous with respect to the theoretical measure space in the tails.

### 3.2 Quantization Noise and Non-Smoothness
The Radon-Nikodym derivative is approximated by a step function due to the limited mantissa bits (only 2 or 3 in FP8). This completely destroys the smoothness of gradients when computing divergences (like Kullback-Leibler divergence, which fundamentally relies on the log of the Radon-Nikodym derivative). The log function severely amplifies quantization errors near zero.
