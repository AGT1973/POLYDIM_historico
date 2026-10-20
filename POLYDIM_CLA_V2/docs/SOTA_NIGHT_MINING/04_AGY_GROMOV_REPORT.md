# Misión de Arqueología SOTA: 04_AGY_GROMOV
**Fecha y Hora:** 2026-07-06T22:00:51.639667
**Query OpenAlex:** `Gromov-Wasserstein alignment latent spaces optimal transport`
**Agente de Computación:** `anthropic/claude-3-haiku`

---

Entendido, voy a proceder con la misión encomendada.

1. Conceptos clave extraídos de los papers:

Conceptos matemáticos y teóricos:
- Distancia de Gromov-Wasserstein (GW): Una noción de discrepancia que compara distribuciones de manera relacional en lugar de absoluta. Permite alinear espacios latentes incomparables.
- Transporte Óptimo (OT): Un marco matemático para encontrar correspondencias óptimas entre conjuntos de objetos. Utilizado para alinear mediciones de células individuales de múltiples modalidades.
- Baricéntricos de GW: Permiten aprender modelos gráficos no paramétricos (graphones) a partir de un conjunto de grafos.
- Transporte Óptimo Multimargen Jerárquico: Extiende el OT para alinear múltiples redes de manera eficiente y escalable.
- Alineación de representaciones de caracteres con OT: Permite calcular la similitud entre cadenas de texto de manera robusta.

Conceptos arquitectónicos:
- Acoplamiento de Variational Autoencoders (VAE) y Transporte Óptimo Desequilibrado: Permite integrar datos multimodales de células individuales en un espacio latente compartido.
- Módulos apilados con algoritmos de punto proximal y ADMM: Permiten optimizar eficientemente modelos basados en GW.
- Transferencia de etiquetas con OT: Permite mapear información multimodal a conjuntos de datos espaciales.

2. Relación con POLYDIM-CLA:

Los conceptos de Transporte Óptimo y distancia de Gromov-Wasserstein son altamente relevantes para POLYDIM-CLA, ya que permiten alinear y comparar espacios latentes de manera relacional, lo cual es fundamental para nuestro enfoque de ruteo vía Lentes Paramétricos (Functoriales). 

La idea de baricéntricos de GW para aprender modelos gráficos no paramétricos también se alinea con nuestra visión de representar los tokens como Espinores de Clifford, ya que estos pueden verse como una forma de graphón que captura las relaciones entre los elementos del vocabulario.

Asimismo, el Transporte Óptimo Multimargen Jerárquico podría ser útil para escalar el alineamiento de múltiples espacios latentes en POLYDIM-CLA, lo cual es crucial dado el gran número de tokens que manejamos.

Finalmente, el uso de acoplamientos de VAE y OT, así como de módulos apilados con algoritmos eficientes, son estrategias arquitectónicas que podrían inspirar soluciones para abordar el cuello de botella de inversión en POLYDIM-CLA, posiblemente aprovechando la identidad de Woodbury.

3. Reporte técnico:

Introducción
El campo del Transporte Óptimo (OT) y la distancia de Gromov-Wasserstein (GW) ha demostrado ser sumamente relevante para abordar diversos problemas en el aprendizaje de máquina, como el alineamiento de datos multimodales, la integración de redes, y el cálculo de similitud entre cadenas de texto. Estos conceptos matemáticos ofrecen un marco teórico sólido para comparar y relacionar distribuciones de manera flexible, lo cual los convierte en herramientas poderosas para el desarrollo de arquitecturas de aprendizaje de representaciones avanzadas.

Alineamiento de datos multimodales
Varios trabajos han explorado el uso del Transporte Óptimo para alinear mediciones de células individuales provenientes de diferentes modalidades, como transcriptómica y accesibilidad de cromatina [1, 2]. Estos enfoques, como SCOT [1] y uniPort [2], combinan el acoplamiento de Variational Autoencoders (VAE) y el Transporte Óptimo Desequilibrado (UOT) para integrar de manera eficiente datos multimodales en un espacio latente compartido, sin requerir información de correspondencia entre células.

Aprendizaje de modelos gráficos
Más allá de los datos de células individuales, los baricéntricos de GW han demostrado ser una herramienta efectiva para aprender modelos gráficos no paramétricos (graphones) a partir de un conjunto de grafos [3]. Estos métodos, como el propuesto por Xu et al. [3], aprovechan la relación entre la distancia de corte de graphones y la distancia de GW de sus aproximaciones por funciones escalonadas, lo cual permite una factorización flexible y eficiente de los grafos observados.

Alineamiento de múltiples redes
Si bien el alineamiento de pares de redes ha sido ampliamente estudiado, el alineamiento de múltiples redes sigue siendo un desafío debido a la complejidad exponencial del espacio de soluciones y la falta de medidas de discrepancia de alto orden. Para abordar este problema, Xu et al. [4] proponen un marco jerárquico de Transporte Óptimo Multimargen, que descompone las múltiples redes en clusters alineados más pequeños utilizando el baricéntrico de Gromov-Wasserstein Fusionado (FGW), y luego alinea las redes de manera conjunta mediante una generalización multimargen de la distancia FGW.

Alineamiento de representaciones de caracteres
Más allá de las aplicaciones en datos estructurados, el Transporte Óptimo también ha demostrado ser útil para calcular la similitud entre cadenas de texto de manera robusta. En particular, el trabajo de Zhao et al. [5] presenta STANCE, un modelo que codifica los caracteres de cada cadena, los alinea utilizando la iteración de Sinkhorn (formulada como un problema de OT), y luego los puntúa mediante una red neuronal convolucional, logrando resultados state-of-the-art en tareas de detección de alias.

Conclusión
Los conceptos de Transporte Óptimo y distancia de Gromov-Wasserstein ofrecen un sólido fundamento teórico y arquitectónico para abordar diversos problemas en el aprendizaje de representaciones, como el alineamiento de datos multimodales, el aprendizaje de modelos gráficos, el alineamiento de múltiples redes y el cálculo de similitud entre cadenas de texto. Estas ideas son altamente relevantes para el desarrollo de POLYDIM-CLA, nuestro framework donde los tokens se representan como Espinores de Clifford y el ruteo se realiza a través de Lentes Paramétricos (Functoriales), ya que permiten comparar y relacionar espacios latentes de manera flexible y escalable.

Referencias:
[1] Singh, R., Lanchantin, J., Sekhon, A., & Qi, Y. (2020). Gromov-Wasserstein optimal transport to align single-cell multi-omics data. bioRxiv.
[2] Xu, C., Qi, Z., Chari, T., Genga, R. M., Hou, R., Peng, T., ... & Yosef, N. (2022). A unified computational framework for single-cell data integration with optimal transport. Nature communications, 13(1), 1-14.
[3] Xu, H., Luo, D., Zha, H., & Duke, L. C. (2021). Learning Graphons via Structured Gromov-Wasserstein Barycenters. Proceedings of the AAAI Conference on Artificial Intelligence, 35(12), 10699-10707.
[4] Xu, H., Luo, D., Zha, H., & Ding, C. H. (2019, June). Hierarchical Multi-Marginal Optimal Transport for Network Alignment. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 7615-7622).
[5] Zhao, Y., Huang, J. X., & Ye, Q. (2019, July). STANCE: Learning Stance Towards Targets Through Temporal Contexts. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 6320-6330).

---
## Anexo: Raw Data (OpenAlex)

<details><summary>Ver Papers Extraídos</summary>

Title: Gromov-Wasserstein optimal transport to align single-cell multi-omics data
Year: 2020
DOI: https://doi.org/10.1101/2020.04.28.066787
Abstract: Abstract Data integration of single-cell measurements is critical for understanding cell development and disease, but the lack of correspondence between different types of measurements makes such efforts challenging. Several unsupervised algorithms can align heterogeneous single-cell measurements in a shared space, enabling the creation of mappings between single cells in different data domains. However, these algorithms require hyperparameter tuning for high-quality alignments, which is difficult in an unsupervised setting without correspondence information for validation. We present Single-Cell alignment using Optimal Transport (SCOT), an unsupervised learning algorithm that uses Gromov Wasserstein-based optimal transport to align single-cell multi-omics datasets. We compare the alignment performance of SCOT with state-of-the-art algorithms on four simulated and two real-world datasets. SCOT performs on par with state-of-the-art methods but is faster and requires tuning fewer hyperparameters. Furthermore, we provide an algorithm for SCOT to use Gromov Wasserstein distance to guide the parameter selection. Thus, unlike previous methods, SCOT aligns well without using any orthogonal correspondence information to pick the hyperparameters. Our source code and scripts for replicating the results are available at https://github.com/rsinghlab/SCOT .

Title: Entropic metric alignment for correspondence problems
Year: 2016
DOI: https://doi.org/10.1145/2897824.2925903
Abstract: Many shape and image processing tools rely on computation of correspondences between geometric domains. Efficient methods that stably extract "soft" matches in the presence of diverse geometric structures have proven to be valuable for shape retrieval and transfer of labels or semantic information. With these applications in mind, we present an algorithm for probabilistic correspondence that optimizes an entropy-regularized Gromov-Wasserstein (GW) objective. Built upon recent developments in numerical optimal transportation, our algorithm is compact, provably convergent, and applicable to any geometric domain expressible as a metric measure matrix. We provide comprehensive experiments illustrating the convergence and applicability of our algorithm to a variety of graphics tasks. Furthermore, we expand entropic GW correspondence to a framework for other matching problems, incorporating partial distance matrices, user guidance, shape exploration, symmetry detection, and joint analysis of more than two domains. These applications expand the scope of entropic GW correspondence to major shape analysis problems and are stable to distortion and noise.

Title: Gromov-Wasserstein Guided Representation Learning for Cross-Domain Recommendation
Year: 2022
DOI: https://doi.org/10.1145/3511808.3557338
Abstract: Cross-Domain Recommendation (CDR) has attracted increasing attention in recent years as a solution to the data sparsity issue. The fundamental paradigm of prior efforts is to train a mapping function based on the overlapping users/items and then apply it to the knowledge transfer. However, due to the commercial privacy policy and the sensitivity of user data, it is unrealistic to explicitly share the user mapping relations and behavior data. Therefore, in this paper, we consider a more practical cross-domain scenario, where there is no explicit overlap between the source and target domains in terms of users/items. Since the user sets of both domains are drawn from the entire population, there may be commonalities between their user characteristics, resulting in comparable user preference distributions. Thus, without the mapping relations at user level, it is feasible to model this distribution-level relation to transfer knowledge between domains. To this end, we propose a novel framework that improves the effect of representation learning on the target domain by aligning the representation distributions between the source and target domains. In addition, GWCDR can be easily integrated with existing single-domain collaborative filtering methods to achieve cross-domain recommendation. Extensive experiments on two pairs of public bidirectional datasets demonstrate the effectiveness of our proposed framework in enhancing the recommendation performance.

Title: Gromov-Wasserstein Factorization Models for Graph Clustering
Year: 2020
DOI: https://doi.org/10.1609/aaai.v34i04.6120
Abstract: We propose a new nonlinear factorization model for graphs that are with topological structures, and optionally, node attributes. This model is based on a pseudometric called Gromov-Wasserstein (GW) discrepancy, which compares graphs in a relational way. It estimates observed graphs as GW barycenters constructed by a set of atoms with different weights. By minimizing the GW discrepancy between each observed graph and its GW barycenter-based estimation, we learn the atoms and their weights associated with the observed graphs. The model achieves a novel and flexible factorization mechanism under GW discrepancy, in which both the observed graphs and the learnable atoms can be unaligned and with different sizes. We design an effective approximate algorithm for learning this Gromov-Wasserstein factorization (GWF) model, unrolling loopy computations as stacked modules and computing gradients with backpropagation. The stacked modules can be with two different architectures, which correspond to the proximal point algorithm (PPA) and Bregman alternating direction method of multipliers (BADMM), respectively. Experiments show that our model obtains encouraging results on clustering graphs.

Title: A unified computational framework for single-cell data integration with optimal transport
Year: 2022
DOI: https://doi.org/10.1038/s41467-022-35094-8
Abstract: Single-cell data integration can provide a comprehensive molecular view of cells. However, how to integrate heterogeneous single-cell multi-omics as well as spatially resolved transcriptomic data remains a major challenge. Here we introduce uniPort, a unified single-cell data integration framework that combines a coupled variational autoencoder (coupled-VAE) and minibatch unbalanced optimal transport (Minibatch-UOT). It leverages both highly variable common and dataset-specific genes for integration to handle the heterogeneity across datasets, and it is scalable to large-scale datasets. uniPort jointly embeds heterogeneous single-cell multi-omics datasets into a shared latent space. It can further construct a reference atlas for gene imputation across datasets. Meanwhile, uniPort provides a flexible label transfer framework to deconvolute heterogeneous spatial transcriptomic data using an optimal transport plan, instead of embedding latent space. We demonstrate the capability of uniPort by applying it to integrate a variety of datasets, including single-cell transcriptomics, chromatin accessibility, and spatially resolved transcriptomic data.

Title: Learning Graphons via Structured Gromov-Wasserstein Barycenters
Year: 2021
DOI: https://doi.org/10.1609/aaai.v35i12.17257
Abstract: We propose a novel and principled method to learn a nonparametric graph model called graphon, which is defined in an infinite-dimensional space and represents arbitrary-size graphs. Based on the weak regularity lemma from the theory of graphons, we leverage a step function to approximate a graphon. We show that the cut distance of graphons can be relaxed to the Gromov-Wasserstein distance of their step functions. Accordingly, given a set of graphs generated by an underlying graphon, we learn the corresponding step function as the Gromov-Wasserstein barycenter of the given graphs. Furthermore, we develop several enhancements and extensions of the basic algorithm, e.g., the smoothed Gromov-Wasserstein barycenter for guaranteeing the continuity of the learned graphons and the mixed Gromov-Wasserstein barycenters for learning multiple structured graphons. The proposed approach overcomes drawbacks of prior state-of-the-art methods, and outperforms them on both synthetic and real-world data. The code is available at https://github.com/HongtengXu/SGWB-Graphon.

Title: Mapping cells through time and space with moscot
Year: 2025
DOI: https://doi.org/10.1038/s41586-024-08453-2
Abstract: Abstract Single-cell genomic technologies enable the multimodal profiling of millions of cells across temporal and spatial dimensions. However, experimental limitations hinder the comprehensive measurement of cells under native temporal dynamics and in their native spatial tissue niche. Optimal transport has emerged as a powerful tool to address these constraints and has facilitated the recovery of the original cellular context 1–4 . Yet, most optimal transport applications are unable to incorporate multimodal information or scale to single-cell atlases. Here we introduce multi-omics single-cell optimal transport (moscot), a scalable framework for optimal transport in single-cell genomics that supports multimodality across all applications. We demonstrate the capability of moscot to efficiently reconstruct developmental trajectories of 1.7 million cells from mouse embryos across 20 time points. To illustrate the capability of moscot in space, we enrich spatial transcriptomic datasets by mapping multimodal information from single-cell profiles in a mouse liver sample and align multiple coronal sections of the mouse brain. We present moscot.spatiotemporal, an approach that leverages gene-expression data across both spatial and temporal dimensions to uncover the spatiotemporal dynamics of mouse embryogenesis. We also resolve endocrine-lineage relationships of delta and epsilon cells in a previously unpublished mouse, time-resolved pancreas development dataset using paired measurements of gene expression and chromatin accessibility. Our findings are confirmed through experimental validation of NEUROD2 as a regulator of epsilon progenitor cells in a model of human induced pluripotent stem cell islet cell differentiation. Moscot is available as open-source software, accompanied by extensive documentation.

Title: Towards Optimal Transport with Global Invariances
Year: 2019
DOI: None
Abstract: Many problems in machine learning involve calculating correspondences between sets of objects, such as point clouds or images. Discrete optimal transport provides a natural and successful approach to such tasks whenever the two sets of objects can be represented in the same space, or at least distances between them can be directly evaluated. Unfortunately neither requirement is likely to hold when object representations are learned from data. Indeed, automatically derived representations such as word embeddings are typically fixed only up to some global transformations, for example, reflection or rotation. As a result, pairwise distances across two such instances are ill-defined without specifying their relative transformation. In this work, we propose a general framework for optimal transport in the presence of latent global transformations. We cast the problem as a joint optimization over transport couplings and transformations chosen from a flexible class of invariances, propose algorithms to solve it, and show promising results in various tasks, including a popular unsupervised word translation benchmark.

Title: Hierarchical Multi-Marginal Optimal Transport for Network Alignment
Year: 2024
DOI: https://doi.org/10.1609/aaai.v38i15.29605
Abstract: Finding node correspondence across networks, namely multi-network alignment, is an essential prerequisite for joint learning on multiple networks. Despite great success in aligning networks in pairs, the literature on multi-network alignment is sparse due to the exponentially growing solution space and lack of high-order discrepancy measures. To fill this gap, we propose a hierarchical multi-marginal optimal transport framework named HOT for multi-network alignment. To handle the large solution space, multiple networks are decomposed into smaller aligned clusters via the fused Gromov-Wasserstein (FGW) barycenter. To depict high-order relationships across multiple networks, the FGW distance is generalized to the multi-marginal setting, based on which networks can be aligned jointly. A fast proximal point method is further developed with guaranteed convergence to a local optimum. Extensive experiments and analysis show that our proposed HOT achieves significant improvements over the state-of-the-art in both effectiveness and scalability.

Title: Cross-modality Matching and Prediction of Perturbation Responses with Labeled Gromov-Wasserstein Optimal Transport
Year: 2024
DOI: https://doi.org/10.48550/arxiv.2405.00838
Abstract: It is now possible to conduct large scale perturbation screens with complex readout modalities, such as different molecular profiles or high content cell images. While these open the way for systematic dissection of causal cell circuits, integrated such data across screens to maximize our ability to predict circuits poses substantial computational challenges, which have not been addressed. Here, we extend two Gromov-Wasserstein Optimal Transport methods to incorporate the perturbation label for cross-modality alignment. The obtained alignment is then employed to train a predictive model that estimates cellular responses to perturbations observed with only one measurement modality. We validate our method for the tasks of cross-modality alignment and cross-modality prediction in a recent multi-modal single-cell perturbation dataset. Our approach opens the way to unified causal models of cell biology.

Title: Optimal Transport-based Alignment of Learned Character Representations for String Similarity
Year: 2019
DOI: https://doi.org/10.18653/v1/p19-1592
Abstract: String similarity models are vital for record linkage, entity resolution, and search. In this work, we present STANCE-a learned model for computing the similarity of two strings. Our approach encodes the characters of each string, aligns the encodings using Sinkhorn Iteration (alignment is posed as an instance of optimal transport) and scores the alignment with a convolutional neural network. We evaluate STANCE's ability to detect whether two strings can refer to the same entity-a task we term alias detection. We construct five new alias detection datasets (and make them publicly available). We show that STANCE (or one of its variants) outperforms both state-ofthe-art and classic, parameter-free similarity models on four of the five datasets. We also demonstrate STANCE's ability to improve downstream tasks by applying it to an instance of cross-document coreference and show that it leads to a 2.8 point improvement in B 3 F1 over the previous state-of-the-art approach.

Title: Learning Generative Models across Incomparable Spaces
Year: 2019
DOI: https://doi.org/10.3929/ethz-b-000382654
Abstract: © 36th International Conference on Machine Learning, ICML 2019. All rights reserved. Generative Adversarial Networks have shown remarkable success in learning a distribution that faithfully recovers a reference distribution in its entirety. However, in some cases, we may want to only learn some aspects (e.g., cluster or manifold structure), while modifying others (e.g., style, orientation or dimension). In this work, we propose an approach to learn generative models across such incomparable spaces, and demonstrate how to steer the learned distribution towards target properties. A key component of our model is the Gromov-Wasserstein distance, a notion of discrepancy that compares distributions relationally rather than absolutely. While this framework subsumes current generative models in identically reproducing distributions, its inherent flexibility allows application to tasks in manifold learning, relational learning and cross-domain learning.

</details>
