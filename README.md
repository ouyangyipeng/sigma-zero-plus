# Perceptually-Aware $\sigma$-zero

> Improving Imperceptibility of $\ell_0$-norm Adversarial Examples through Visual Perception Constraints

## Overview

This project reproduces and extends the **$\sigma$-zero** algorithm (Cinà et al., ICLR 2025), a state-of-the-art gradient-based $\ell_0$-norm adversarial attack. We identify that the original $\sigma$-zero produces visually detectable adversarial examples and propose **Perceptually-Aware $\sigma$-zero**, which incorporates three complementary visual perception constraints:

1. **$\ell_\infty$-norm penalty** — bounds per-pixel perturbation magnitude
2. **SSIM loss** — preserves structural similarity
3. **Total Variation (TV) regularization** — encourages spatial smoothness

## Key Results

| Method | ASR (%) | L0 Median | SSIM | SSIM Δ |
|--------|---------|-----------|------|--------|
| $\sigma$-zero (original) | 100.0 | 23.0 | 0.7939 | — |
| + Multi-scale SSIM | 100.0 | 23.0 | **0.8698** | **+9.6%** |
| + TV | 100.0 | 21.0 | 0.8695 | +9.5% |
| + Full (all) | 100.0 | **18.0** | 0.8662 | +9.1% |

## Project Structure

```
.
├── mypaper/                    # Paper (SIGPLAN template)
│   ├── main.tex                # Main entry point
│   ├── references.bib          # Bibliography (20 verified references)
│   ├── sections/               # Section files
│   │   ├── 00_abstract.tex
│   │   ├── 01_introduction.tex
│   │   ├── 02_original_paper_weakness.tex
│   │   ├── 03_design.tex
│   │   ├── 04_evaluation.tex
│   │   └── 05_discussion.tex
│   ├── figures/                # Publication-quality figures
│   └── sigplan/                # SIGPLAN template files
├── sigma-zero-adversarial-attack/  # Original + modified code
│   ├── sigma_zero_perceptual.py    # Our improved implementation
│   ├── attacks.py                  # Attack registry
│   ├── plot_results.py             # Visualization scripts
│   └── configs/                    # Experiment configurations
├── original-paper/             # Original σ-zero paper PDF
├── plans/                      # Project planning documents
└── report.md                   # Initial reproduction report
```

## Reproduction

### Environment Setup

```bash
cd sigma-zero-adversarial-attack
conda env create -f env_china.yml
conda activate sigma-zero-china
```

### Run Experiments

```bash
# Reproduce original σ-zero on MNIST
python main.py --device=cpu --config=configs/config_reproduce.json

# Run extended comparison (7 methods)
python main.py --device=cpu --config=configs/config_mnist_extended.json

# Run hyperparameter search
python run_hyperparameter_search.py

# Generate figures
python plot_results.py
```

### Compile Paper

```bash
cd mypaper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Citation

```bibtex
@article{cina2025sigma,
  title   = {{$\sigma$}-zero: Gradient-based Optimization of {$\ell_0$}-norm Adversarial Examples},
  author  = {Cin{\`a}, Antonio Emanuele and Villani, Francesco and Pintor, Maura and Sch{\"o}nherr, Lea and Biggio, Battista and Pelillo, Marcello},
  journal = {arXiv preprint arXiv:2402.01879},
  year    = {2024}
}
```
