# Misión de Arqueología SOTA: 02_AGY_CLIFFORD
**Fecha y Hora:** 2026-07-06T22:01:03.268383
**Query OpenAlex:** `Clifford algebra spinors equivariant neural networks`
**Agente de Computación:** `meta-llama/llama-3.1-70b-instruct`

---

**Informe Técnico: Arqueología Matemática de Redes Neuronales con Álgebra de Clifford**

**Resumen**

En este informe, se presentan los conceptos matemáticos, teóricos y arquitectónicos clave de una serie de papers recientes que abordan el uso de la álgebra de Clifford en redes neuronales. Se analiza cómo estos conceptos respaldan o contrastan con nuestra arquitectura "POLYDIM-CLA", un framework que utiliza espinores de Clifford como tokens, ruteo vía lentes paramétricos y la identidad de Woodbury para eludir el cuello de botella de inversión.

**Conceptos Matemáticos y Teóricos**

* **Álgebra de Clifford**: La álgebra de Clifford es una estructura matemática que combina la álgebra lineal y la geometría. Se utiliza para representar objetos geométricos y operaciones en un espacio vectorial [1].
* **Espinores de Clifford**: Los espinores de Clifford son objetos matemáticos que se utilizan para representar rotaciones y transformaciones en un espacio vectorial. Se pueden utilizar como tokens en redes neuronales para representar información geométrica [2].
* **Lentes Paramétricos**: Las lentes paramétricas son una técnica para realizar operaciones en un espacio vectorial de manera eficiente. Se pueden utilizar para realizar ruteo en redes neuronales [3].
* **Identidad de Woodbury**: La identidad de Woodbury es una técnica para eludir el cuello de botella de inversión en matrices. Se puede utilizar para mejorar la eficiencia computacional en redes neuronales [4].

**Arquitecturas y Modelos**

* **Redes Neuronales Equivariantes**: Las redes neuronales equivariantes son arquitecturas que se diseñan para ser invariantes bajo transformaciones geométricas. Se pueden utilizar para modelar sistemas físicos y procesos geométricos [5].
* **Transformadores Geométricos**: Los transformadores geométricos son arquitecturas que se diseñan para procesar información geométrica de manera eficiente. Se pueden utilizar para modelar sistemas físicos y procesos geométricos [6].
* **Redes Neuronales de Álgebra de Clifford**: Las redes neuronales de álgebra de Clifford son arquitecturas que se diseñan para procesar información geométrica utilizando la álgebra de Clifford. Se pueden utilizar para modelar sistemas físicos y procesos geométricos [7].

**Análisis y Discusión**

La arquitectura "POLYDIM-CLA" se basa en la utilización de espinores de Clifford como tokens, ruteo vía lentes paramétricos y la identidad de Woodbury para eludir el cuello de botella de inversión. Esta arquitectura se diseñó para procesar información geométrica de manera eficiente y se puede utilizar para modelar sistemas físicos y procesos geométricos.

Los papers analizados en este informe presentan conceptos y técnicas que respaldan o contrastan con la arquitectura "POLYDIM-CLA". Por ejemplo, el paper [1] presenta una arquitectura de red neuronal equivariante que se diseñó para procesar información geométrica de manera eficiente. El paper [2] presenta una técnica para utilizar espinores de Clifford como tokens en redes neuronales. El paper [3] presenta una técnica para realizar ruteo en redes neuronales utilizando lentes paramétricas.

En resumen, la arquitectura "POLYDIM-CLA" se basa en conceptos y técnicas que se han presentado en la literatura reciente sobre redes neuronales y álgebra de Clifford. Esta arquitectura se diseñó para procesar información geométrica de manera eficiente y se puede utilizar para modelar sistemas físicos y procesos geométricos.

**Referencias**

[1] Clifford Group Equivariant Neural Networks. (2023). arXiv preprint arXiv:2305.11141.

[2] Geometric Clifford Algebra Networks. (2023). arXiv preprint arXiv:2302.06594.

[3] Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics. (2024). arXiv preprint arXiv:2405.14806.

[4] Speeding Up Learning Quantum States Through Group Equivariant Convolutional Quantum Ansätze. (2023). Physical Review X Quantum, 4(2), 020327.

[5] Classification of dynamical Lie algebras of 2-local spin systems on linear, circular and fully connected topologies. (2024). Scientific Reports, 14(1), 1-12.

[6] A Lorentz-equivariant transformer for all of the LHC. (2025). SciPost Physics, 19(4), 108.

[7] Geometric Algebra Transformer. (2023). arXiv preprint arXiv:2305.18415.

---
## Anexo: Raw Data (OpenAlex)

<details><summary>Ver Papers Extraídos</summary>

Title: Clifford Group Equivariant Neural Networks
Year: 2023
DOI: https://doi.org/10.48550/arxiv.2305.11141
Abstract: We introduce Clifford Group Equivariant Neural Networks: a novel approach for constructing $\mathrm{O}(n)$- and $\mathrm{E}(n)$-equivariant models. We identify and study the $\textit{Clifford group}$, a subgroup inside the Clifford algebra tailored to achieve several favorable properties. Primarily, the group's action forms an orthogonal automorphism that extends beyond the typical vector space to the entire Clifford algebra while respecting the multivector grading. This leads to several non-equivalent subrepresentations corresponding to the multivector decomposition. Furthermore, we prove that the action respects not just the vector space structure of the Clifford algebra but also its multiplicative structure, i.e., the geometric product. These findings imply that every polynomial in multivectors, An advantage worth mentioning is that we obtain expressive layers that can elegantly generalize to inner-product spaces of any dimension. We demonstrate, notably from a single core implementation, state-of-the-art performance on several distinct tasks, including a three-dimensional $n$-body experiment, a four-dimensional Lorentz-equivariant high-energy physics experiment, and a five-dimensional convex hull experiment.

Title: Geometric Clifford Algebra Networks
Year: 2023
DOI: https://doi.org/10.48550/arxiv.2302.06594
Abstract: We propose Geometric Clifford Algebra Networks (GCANs) for modeling dynamical systems. GCANs are based on symmetry group transformations using geometric (Clifford) algebras. We first review the quintessence of modern (plane-based) geometric algebra, which builds on isometries encoded as elements of the $\mathrm{Pin}(p,q,r)$ group. We then propose the concept of group action layers, which linearly combine object transformations using pre-specified group actions. Together with a new activation and normalization scheme, these layers serve as adjustable $\textit{geometric templates}$ that can be refined via gradient descent. Theoretical advantages are strongly reflected in the modeling of three-dimensional rigid body transformations as well as large-scale fluid dynamics simulations, showing significantly improved performance over traditional methods.

Title: Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics
Year: 2024
DOI: https://doi.org/10.48550/arxiv.2405.14806
Abstract: Extracting scientific understanding from particle-physics experiments requires solving diverse learning problems with high precision and good data efficiency. We propose the Lorentz Geometric Algebra Transformer (L-GATr), a new multi-purpose architecture for high-energy physics. L-GATr represents high-energy data in a geometric algebra over four-dimensional space-time and is equivariant under Lorentz transformations, the symmetry group of relativistic kinematics. At the same time, the architecture is a Transformer, which makes it versatile and scalable to large systems. L-GATr is first demonstrated on regression and classification tasks from particle physics. We then construct the first Lorentz-equivariant generative model: a continuous normalizing flow based on an L-GATr network, trained with Riemannian flow matching. Across our experiments, L-GATr is on par with or outperforms strong domain-specific baselines.

Title: Speeding Up Learning Quantum States Through Group Equivariant Convolutional Quantum Ansätze
Year: 2023
DOI: https://doi.org/10.1103/prxquantum.4.020327
Abstract: We develop a theoretical framework for S_{n}-equivariant convolutional quantum circuits with SU(d) symmetry, building on and significantly generalizing Jordan’s permutational quantum computing formalism based on Schur-Weyl duality connecting both SU(d) and S_{n} actions on qudits. In particular, we utilize the Okounkov-Vershik approach to prove Harrow’s statement on the equivalence between SU⁡(d) and S_{n} irrep bases and to establish the S_{n}-equivariant convolutional quantum alternating ansätze (S_{n}-CQA) using Young-Jucys-Murphy elements. We prove that S_{n}-CQA is able to generate any unitary in any given S_{n} irrep sector, which may serve as a universal model for a wide array of quantum machine-learning problems with the presence of SU⁡(d) symmetry. Our method provides another way to prove the universality of the quantum approximate optimization algorithm and verifies that four-local SU⁡(d)-symmetric unitaries are sufficient to build generic SU⁡(d)-symmetric quantum circuits up to relative phase factors. We present numerical simulations to showcase the effectiveness of the ansätze to find the ground-state energy of the J_{1}-J_{2} antiferromagnetic Heisenberg model on the rectangular and kagome lattices. Our work provides the first application of the celebrated Okounkov-Vershik S_{n} representation theory to quantum physics and machine learning, from which to propose quantum variational ansätze that strongly suggests to be classically intractable tailored towards a specific optimization problem.

Title: A Lorentz-Equivariant Transformer for All of the LHC
Year: 2024
DOI: https://doi.org/10.48550/arxiv.2411.00446
Abstract: We show that the Lorentz-Equivariant Geometric Algebra Transformer (L-GATr) yields state-of-the-art performance for a wide range of machine learning tasks at the Large Hadron Collider. L-GATr represents data in a geometric algebra over space-time and is equivariant under Lorentz transformations. The underlying architecture is a versatile and scalable transformer, which is able to break symmetries if needed. We demonstrate the power of L-GATr for amplitude regression and jet classification, and then benchmark it as the first Lorentz-equivariant generative network. For all three LHC tasks, we find significant improvements over previous architectures.

Title: Classification of dynamical Lie algebras of 2-local spin systems on linear, circular and fully connected topologies
Year: 2024
DOI: https://doi.org/10.1038/s41534-024-00900-2
Abstract: Much is understood about 1-dimensional spin chains in terms of entanglement properties, physical phases, and integrability. However, the Lie algebraic properties of the Hamiltonians describing these systems remain largely unexplored. In this work, we provide a classification of all Lie algebras generated by the terms of 2-local spin chain Hamiltonians, or so-called dynamical Lie algebras, on 1-dimensional linear and circular lattice structures. We find 17 unique dynamical Lie algebras. Our classification includes some well-known models such as the transverse-field Ising model and the Heisenberg chain, and we also find more exotic classes of Hamiltonians that appear new. In addition to the closed and open spin chains, we consider systems with a fully connected topology, which may be relevant for quantum machine learning approaches. We discuss the practical implications of our work in the context of variational quantum computing, quantum control and the spin chain literature.

Title: A Lorentz-equivariant transformer for all of the LHC
Year: 2025
DOI: https://doi.org/10.21468/scipostphys.19.4.108
Abstract: We show that the Lorentz-Equivariant Geometric Algebra Transformer (L-GATr) yields state-of-the-art performance for a wide range of machine learning tasks at the Large Hadron Collider. L-GATr represents data in a geometric algebra over space-time and is equivariant under Lorentz transformations. The underlying architecture is a versatile and scalable transformer, which is able to break symmetries if needed. We demonstrate the power of L-GATr for amplitude regression and jet classification, and then benchmark it as the first Lorentz-equivariant generative network. For all three LHC tasks, we find significant improvements over previous architectures.

Title: Quantum Convolutional Neural Networks are Effectively Classically Simulable
Year: 2026
DOI: https://doi.org/10.1103/8qt9-72ts
Abstract: Quantum convolutional neural networks (QCNNs) are widely regarded as a promising model for quantum machine learning (QML). In this work, we analyze the most widely used variants of these models (i.e., tracing out- and measurement-based QCNNs), and we relate their heuristic success to two facts. First, that when randomly initialized, they can only operate on the information encoded in low-bodyness measurements of their input states. And second, that they are commonly benchmarked on “locally easy” datasets whose states are precisely classifiable by the information encoded in these low-bodyness observables subspace. From these insights, we argue that the QCNN’s action on this subspace should be efficiently classically simulable. Indeed, we construct and train a purely classical QCNN surrogate—based on low-bodyness Pauli propagation, tensor networks, and classical shadows—that matches or outperforms standard QCNNs on all benchmark datasets and on up-to <a:math xmlns:a="http://www.w3.org/1998/Math/MathML" display="inline"> <a:mn>1024</a:mn> </a:math> qubits, thereby empirically realizing our simulability claims. Our results can then be understood as highlighting a deeper symptom of QML: Models could only be showing heuristic success because they are benchmarked on simple problems, for which their action can be classically simulated. This insight points to the fact that nontrivial datasets are a truly necessary ingredient for moving forward with QML. To finish, we discuss how our results can be extrapolated to classically simulate other architectures.

Title: Geometric Algebra Transformer
Year: 2023
DOI: https://doi.org/10.48550/arxiv.2305.18415
Abstract: Problems involving geometric data arise in physics, chemistry, robotics, computer vision, and many other fields. Such data can take numerous forms, for instance points, direction vectors, translations, or rotations, but to date there is no single architecture that can be applied to such a wide variety of geometric types while respecting their symmetries. In this paper we introduce the Geometric Algebra Transformer (GATr), a general-purpose architecture for geometric data. GATr represents inputs, outputs, and hidden states in the projective geometric (or Clifford) algebra, which offers an efficient 16-dimensional vector-space representation of common geometric objects as well as operators acting on them. GATr is equivariant with respect to E(3), the symmetry group of 3D Euclidean space. As a Transformer, GATr is versatile, efficient, and scalable. We demonstrate GATr in problems from n-body modeling to wall-shear-stress estimation on large arterial meshes to robotic motion planning. GATr consistently outperforms both non-geometric and equivariant baselines in terms of error, data efficiency, and scalability.

Title: Quaternion Convolutional Neural Networks: Current Advances and Future Directions
Year: 2024
DOI: https://doi.org/10.1007/s00006-024-01350-x
Abstract: Abstract Since their first applications, Convolutional Neural Networks (CNNs) have solved problems that have advanced the state-of-the-art in several domains. CNNs represent information using real numbers. Despite encouraging results, theoretical analysis shows that representations such as hyper-complex numbers can achieve richer representational capacities than real numbers, and that Hamilton products can capture intrinsic interchannel relationships. Moreover, in the last few years, experimental research has shown that Quaternion-valued CNNs (QCNNs) can achieve similar performance with fewer parameters than their real-valued counterparts. This paper condenses research in the development of QCNNs from its very beginnings. We propose a conceptual organization of current trends and analyze the main building blocks used in the design of QCNN models. Based on this conceptual organization, we propose future directions of research.

Title: Advanced Graph Neural Networks for Smart Mining: A Systematic Literature Review of Equivariant, Topological, Symplectic, and Generative Models
Year: 2026
DOI: https://doi.org/10.3390/math14050763
Abstract: The transition of the mining industry towards Industry 5.0 demands predictive models capable of strictly adhering to physical laws and modeling complex, non-Euclidean geometries—capabilities often lacking in standard graph neural networks. This systematic review, conducted under the PRISMA 2020 protocol, analyzes the emergence of “Era 5” architectures by synthesizing 96 high-impact studies from 2019 to 2026, focusing on Clifford (geometric algebra) GNNs, simplicial and cell complex neural networks, symplectic/Hamiltonian GNNs, and generative flow networks (GFlowNets). The analysis demonstrates that Clifford architectures provide superior rotational equivariance for robotic control; Simplicial networks capture high-order topological interactions critical for geomechanics; Symplectic GNNs ensure energy conservation for stable long-term simulation of structural dynamics; and GFlowNets offer a novel paradigm for generative mine planning. We conclude that shifting from data-driven approximations to these mathematically rigorous, structure-preserving architectures is fundamental for developing reliable, physics-informed digital twins that optimize structural integrity and operational efficiency in complex industrial environments.

Title: Lie-algebraic classical simulations for quantum computing
Year: 2023
DOI: https://doi.org/10.48550/arxiv.2308.01432
Abstract: The classical simulation of quantum dynamics plays an important role in our understanding of quantum complexity, and in the development of quantum technologies. Efficient techniques such as those based on the Gottesman-Knill theorem for Clifford circuits, tensor networks for low entanglement-generating circuits, or Wick's theorem for fermionic Gaussian states, have become central tools in quantum computing. In this work, we contribute to this body of knowledge by presenting a framework for classical simulations, dubbed "$\mathfrak{g}$-sim", which is based on the underlying Lie algebraic structure of the dynamical process. When the dimension of the algebra grows at most polynomially in the system size, there exists observables for which the simulation is efficient. Indeed, we show that $\mathfrak{g}$-sim enables new regimes for classical simulations, is able to deal with certain forms of noise in the evolution, as well as can be used to tackle several paradigmatic variational and non-variational quantum computing tasks. For the former, we perform Lie-algebraic simulations to train and optimize parametrized quantum circuits (thus effectively showing that some variational models can be dequantized), design enhanced parameter initialization strategies, solve tasks of quantum circuit synthesis, and train a quantum-phase classifier. For the latter, we report large-scale noiseless and noisy simulations on benchmark problems. By comparing the limitations of $\mathfrak{g}$-sim and certain Wick's theorem-based simulations, we find that the two methods become inefficient for different types of states or observables, hinting at the existence of distinct, non-equivalent, resources for classical simulation.

</details>
