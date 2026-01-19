# LaTeX Paper Compilation Guide

## Files Created

1. **paper.tex** - Main LaTeX document (complete paper)
2. **references.bib** - Bibliography file with all citations
3. **neurips_2024.sty** - Required style file (see below)

## Required: NeurIPS 2024 Style File

The paper uses the NeurIPS 2024 conference format. You need to download the style file:

**Option 1: Download from NeurIPS**
```bash
wget https://media.neurips.cc/Conferences/NeurIPS2024/Styles/neurips_2024.sty
```

**Option 2: Use Alternative Conference Style**

If you prefer a different conference style, replace line 4 in paper.tex:
```latex
% Original:
\usepackage[final]{neurips_2024}

% Alternatives:
\usepackage{icml2024}      % ICML
\usepackage{aaai24}        % AAAI
\usepackage{acl2024}       % ACL
\documentclass{article}    % Plain article (remove the neurips line)
```

## Compilation Instructions

### Using pdflatex (Recommended)

```bash
cd /Users/schatta/Downloads/personal/areas_of_work/llm_interference_framework/docs

# Compile (run 3 times for references)
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

### Using latexmk (Easier, auto-handles multiple runs)

```bash
latexmk -pdf paper.tex
```

### Using Overleaf (Online, No Installation Required)

1. Go to https://www.overleaf.com
2. Create new project → Upload Project
3. Upload: `paper.tex`, `references.bib`, `neurips_2024.sty`
4. Upload all figure files from `../results/key_results/*.png`
5. Click "Recompile"

## Figures

The LaTeX file references figures with relative paths:
```latex
\includegraphics[width=\textwidth]{../results/key_results/figure_name.png}
```

**All required figures are already generated:**
- `regression_analysis_results.png` (Figure 1)
- `decay_curves_reasoning.png` (Figure 2)
- `decay_curves_size_tiers.png` (Figure 3)
- `figure4_ri_vs_pi.png` (Figure 4)

**If compiling on a different machine:**
1. Copy the entire `results/key_results/` folder
2. Maintain the relative path structure, or
3. Update paths in paper.tex to match your directory structure

## Expected Output

After successful compilation:
- **paper.pdf** - Final paper (approximately 20-25 pages)
- Includes all 4 figures, tables, and citations
- Formatted in NeurIPS conference style (two-column layout)

## Troubleshooting

### Missing neurips_2024.sty
**Error:** `File 'neurips_2024.sty' not found`

**Solution 1:** Download from NeurIPS website (see above)

**Solution 2:** Use alternative style (replace line 4 in paper.tex):
```latex
\documentclass[11pt,letterpaper]{article}
\usepackage{fullpage}
% Remove the neurips_2024 line
```

### Missing Figures
**Error:** `File '../results/key_results/figure_name.png' not found`

**Solution:** Ensure figures are in correct location or update paths:
```latex
% If figures are in same directory as paper.tex:
\includegraphics[width=\textwidth]{figure_name.png}

% If using Overleaf, upload to root and use:
\includegraphics[width=\textwidth]{./figure_name.png}
```

### Bibliography Not Showing
**Error:** Citations appear as `[?]`

**Solution:** Run bibtex and recompile:
```bash
pdflatex paper.tex
bibtex paper      # This step is critical
pdflatex paper.tex
pdflatex paper.tex
```

### Package Errors
**Error:** `! LaTeX Error: File 'xxx.sty' not found`

**Solution:** Install missing packages:
```bash
# For TeX Live:
tlmgr install <package-name>

# For MiKTeX:
mpm --install=<package-name>

# Common missing packages:
tlmgr install booktabs amsfonts nicefrac microtype
```

## Customization

### Change Title/Authors
Edit lines 18-24 in paper.tex:
```latex
\title{Your Title Here}

\author{%
  Your Name\\
  Your Institution\\
  \texttt{your.email@institution.edu} \\
}
```

### Adjust Figure Sizes
Modify `width` parameter:
```latex
% Current (full width):
\includegraphics[width=\textwidth]{figure.png}

% Half width:
\includegraphics[width=0.5\textwidth]{figure.png}

% Custom width:
\includegraphics[width=12cm]{figure.png}
```

### Add Appendix
Add after `\bibliography{references}` and before `\end{document}`:
```latex
\appendix

\section{Additional Results}
Your appendix content here...

\section{Model Details}
Complete list of 56 models tested...
```

## Paper Statistics

- **Length:** ~20-25 pages (conference format with figures)
- **Sections:** 7 main sections (Intro, Related Work, Methods, Results, Discussion, Conclusion)
- **Figures:** 4 publication-quality figures with error bars
- **Tables:** 12 tables (statistics, comparisons, model lists)
- **References:** ~20 citations (psychology, ML, LLM research)
- **Word Count:** ~10,000 words

## Next Steps After Compilation

1. **Proofread the PDF**
   - Check all figures render correctly
   - Verify tables are properly formatted
   - Review citations are complete

2. **Add Missing Information**
   - Author names and affiliations
   - Acknowledgments (if needed)
   - Funding information (if applicable)

3. **Final Checks**
   - All hypotheses clearly stated
   - All RQs answered
   - Figures referenced in text
   - No orphaned citations `[?]`

4. **Submission Preparation**
   - Check conference page limits (usually 8-10 pages + references)
   - May need to move content to appendix
   - Verify formatting matches conference requirements

## Contact

For LaTeX issues, see:
- LaTeX Stack Exchange: https://tex.stackexchange.com
- Overleaf Documentation: https://www.overleaf.com/learn

For paper content questions, refer to PAPER_DRAFT.md (markdown version with same content).
