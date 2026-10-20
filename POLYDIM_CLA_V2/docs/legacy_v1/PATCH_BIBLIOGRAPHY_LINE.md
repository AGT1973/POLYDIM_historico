# POLYDIM_DEST
# destino: polydim/docs/
# nombre:  PATCH_BIBLIOGRAPHY_LINE.md
# autor:   Claude Sonnet 4.6
# fecha:   2026-06-30

---

# Patch de una línea — antes de compilar

`polydim_paper_arxiv_final.tex` (1qPwR4kl6hj-ZtU5zFWMJWfHis3meIHWk) tiene:

```latex
\bibliography{polydim}
```

`polydim_final.bib` (139Yc7qVZ6k1AvHEm6aPuVfOksUoi9bFR) es el .bib fusionado con las 26 referencias completas (incluye Suarez 2026 + O'Neill 2025).

**Único cambio necesario antes de compilar:**

```latex
\bibliography{polydim_final}
```

O alternativamente: renombrar `polydim_final.bib` a `polydim.bib` en el directorio de compilación — así no se toca el `.tex` en absoluto.

**Recomendación:** la segunda opción (renombrar el .bib) es más simple y evita cualquier edición del paper.

Secuencia de compilación sin tocar el .tex:
```bash
cp polydim_final.bib polydim.bib
pdflatex polydim_paper_arxiv_final.tex
bibtex   polydim_paper_arxiv_final
pdflatex polydim_paper_arxiv_final.tex
pdflatex polydim_paper_arxiv_final.tex
```

---

*Nota: se intentó parchear el .tex en este entorno vía descarga/edición/resubida, pero la transcripción manual del base64 (30KB) resultó propensa a errores de copiado (ej. "atend" en vez de "attend", "directimon" en vez de "direction"). Se abortó esa vía para no arriesgar la integridad del paper. El archivo original `polydim_paper_arxiv_final.tex` permanece intacto y correcto — ID 1qPwR4kl6hj-ZtU5zFWMJWfHis3meIHWk. Este patch de una línea es la vía segura.*
