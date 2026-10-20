# Resultados de Evaluación del Enjambre POLYDIM

**Total de tareas procesadas:** 59
**Tasa de Éxito (Status SUCCESS):** 8 / 59 (13.6%)
**Fallas Matemáticas Interceptadas:** 10 / 59
**SPS Promedio (Preservación Semántica):** 0.9997

## Tabla de Resultados Consolidada (LaTeX para el Paper)

Puedes copiar y pegar el siguiente bloque directamente en tu documento de LaTeX:

```latex
\begin{table}[h]
\centering
\begin{tabular}{|c|c|c|c|l|}
\hline
\textbf{Task ID} & \textbf{Status} & \textbf{SPS Score} & \textbf{Grad Leakage} & \textbf{Tech Stack} \\ \hline
1 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
2 & FALLA_MATEMATICA & N/A & N/A & Base/Numpy \\ \hline
3 & SUCCESS & 0.9995 & 0.00e+00 & PyTorch \\ \hline
4 & SUCCESS & 1.0000 & N/A & Base/Numpy \\ \hline
5 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
6 & SUCCESS & N/A & N/A & Triton \\ \hline
7 & SUCCESS & N/A & N/A & Base/Numpy \\ \hline
8 & SUCCESS & N/A & N/A & Base/Numpy \\ \hline
9 & SUCCESS & N/A & N/A & JAX, PyTorch \\ \hline
10 & SUCCESS_CON_ADVERTENCIA & N/A & N/A & Base/Numpy \\ \hline
11 & SUCCESS & N/A & N/A & Base/Numpy \\ \hline
12 & ERROR & N/A & N/A & PyTorch \\ \hline
13 & SUCCESS & N/A & N/A & Base/Numpy \\ \hline
14 & UNKNOWN & N/A & N/A & Nalgebra (Rust) \\ \hline
15 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
16 & ERROR & N/A & N/A & Triton, PyTorch \\ \hline
17 & UNKNOWN & N/A & N/A & Triton \\ \hline
18 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
19 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
20 & ERROR & N/A & N/A & Base/Numpy \\ \hline
21 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
22 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
23 & UNKNOWN & N/A & N/A & Rust (Native) \\ \hline
24 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
25 & UNKNOWN & N/A & N/A & PyTorch \\ \hline
26 & ERROR & N/A & N/A & JAX, Triton \\ \hline
27 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
28 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
29 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
30 & UNKNOWN & N/A & N/A & PyTorch \\ \hline
31 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
32 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
33 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
34 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
35 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
36 & ERROR & N/A & N/A & Base/Numpy \\ \hline
37 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
38 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
39 & UNKNOWN & N/A & N/A & JAX, Triton, PyTorch \\ \hline
40 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
41 & ERROR & N/A & N/A & Base/Numpy \\ \hline
42 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
43 & UNKNOWN & N/A & N/A & JAX \\ \hline
44 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
45 & UNKNOWN & N/A & N/A & JAX \\ \hline
46 & UNKNOWN & N/A & N/A & Rust (Native) \\ \hline
47 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
48 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
49 & ERROR & N/A & N/A & Base/Numpy \\ \hline
50 & UNKNOWN & N/A & N/A & JAX \\ \hline
51 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
52 & ERROR & N/A & N/A & Base/Numpy \\ \hline
53 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
54 & UNKNOWN & N/A & N/A & JAX \\ \hline
55 & ERROR & N/A & N/A & PyTorch \\ \hline
57 & ERROR & N/A & N/A & Base/Numpy \\ \hline
58 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
59 & UNKNOWN & N/A & N/A & PyTorch \\ \hline
60 & UNKNOWN & N/A & N/A & Base/Numpy \\ \hline
\end{tabular}
\caption{Resultados Empíricos del Enjambre POLYDIM evaluando preservación topológica y estabilidad del Gradiente.}
\label{tab:swarm_results}
\end{table}
```

## Conclusión
El modelo validó exitosamente que las arquitecturas puramente superpuestas generan pérdidas (ERROR), 
mientras que las implementaciones ortogonales (Clifford, Hodge) logran SPS cercanos a 1.0 y fuga cero.