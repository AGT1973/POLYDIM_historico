# SOTA Asia 2025-2026: Latent Multi-Agent Systems (LatentMAS)

## 1. Overview of Regional Developments
"Latent Multi-Agent Systems" (LatentMAS) has exploded as a prominent research topic in 2025-2026. This framework enables LLM agents to collaborate by communicating through continuous latent representations instead of natural language tokens, bypassing the "communication bottleneck."

### Key Theoretical Contributions
- **Qwen3 and DeepSeek Benchmarks:** The Chinese AI sector is central to this research. The Qwen3 family (2025) and DeepSeek-R1-Distill-Qwen models are widely used as the primary backbone models to evaluate LatentMAS architectures. 
- **DiffMAS & LatentCompress:** Parallel frameworks like DiffMAS explore end-to-end optimization of these multi-agent language systems. Research focuses on compressing the latent states (LatentCompress) to manage information flow effectively.
- **ICML 2026 Insights:** Work presented at the ICML 2026 Workshop on Compositional Learning reveals a mechanistic truth: while latent communication allows agents to adopt distinct role signatures, much of the information signal stems from the base model's geometry (e.g., Qwen's latent space) rather than purely emergent collaborative logic.

## 2. Mathematical Foundations
Instead of generating discrete tokens via argmax, the system transfers hidden states directly via learned alignment matrices (e.g., linear transformations or shallow MLPs mapping one agent's embedding space to another). The mathematical focus is on optimizing this alignment matrix end-to-end using gradient descent across the multi-agent graph.

## 3. Evaluation against the POLYDIM Theoretical Framework
- **Alignment:** LatentMAS is the practical realization of POLYDIM's hypothesis. Agents interacting via continuous latent manifolds directly mirror POLYDIM's multi-dimensional conceptual spaces. 
- **Critique:** The ICML 2026 finding that signals are dominated by the base model's geometry poses a challenge. POLYDIM must ensure that its theoretical framework can mathematically disentangle the "base geometry" (e.g., DeepSeek's pre-trained space) from the "emergent relational geometry" that happens during agent interaction.
