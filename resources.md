# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project on Topological Drift Detection for LLM Embedding Streams. Resources include papers, datasets, and code repositories needed for comprehensive experimentation comparing Wasserstein distance on persistence diagrams against modern two-sample tests (MMDAgg, deep kernel MMD, C2ST).

## Papers
Total papers downloaded: 13 (11 relevant, 2 wrong arXiv IDs kept for reference)

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | MMD Aggregated Two-Sample Test | Schrab et al. | 2023 | papers/schrab2021_mmd_aggregated_two_sample_test.pdf | Primary baseline, adaptive bandwidth |
| 2 | Learning Deep Kernels for Two-Sample Tests | Liu et al. | 2020 | papers/liu2020_deep_kernel_two_sample_test.pdf | Deep kernel baseline |
| 3 | Revisiting Classifier Two-Sample Tests | Lopez-Paz & Oquab | 2017 | papers/lopez-paz2017_classifier_two_sample_test.pdf | C2ST baseline |
| 4 | A Kernel Two-Sample Test | Gretton et al. | 2012 | papers/gretton2012_kernel_two_sample_test_mmd.pdf | Foundational MMD |
| 5 | Two-sample Tests for Persistence Diagrams | Krebs & Rademacher | 2024 | papers/krebs2024_two_sample_test_persistence_diagrams.pdf | TDA two-sample tests |
| 6 | Hypothesis Testing for TDA | Robinson & Turner | 2013 | papers/hypothesis_testing_topological_data_analysis.pdf | TDA hypothesis testing |
| 7 | Uncovering Drift in Textual Data | Khaki et al. | 2023 | papers/2309.03831_drift_detection.pdf | MMD drift in text embeddings |
| 8 | MMDEW: MMD on Exponential Windows | Kalinke et al. | 2025 | papers/2205.12706_mmdew_change_detection.pdf | Efficient online MMD |
| 9 | Optimized MMD for Model Criticism | Sutherland et al. | 2017 | papers/1611.04488_tda_ml.pdf | Optimized MMD framework |
| 10 | Matryoshka Representation Learning | Kusupati et al. | 2022 | papers/2205.13147_matryoshka_representation_learning.pdf | Wrong ID match, embedding learning |
| 11 | Distributional Shifts in Text via LLM Embeddings | Gupta et al. | 2023 | papers/gupta2023_distributional_shifts_text_embeddings.pdf | Clustering-based drift detection, deployed 18mo |
| 12 | Approx. Algorithms for W1 on Persistence Diagrams | Chen & Wang | 2021 | papers/chen2021_wasserstein_persistence_diagrams.pdf | Near-linear time W1 computation for PDs |
| 13 | MMD-FUSE | Biggs, Schrab, Gretton | 2023 | papers/biggs2023_mmd_fuse.pdf | NeurIPS, kernel combination without splitting |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| AG News | HuggingFace `ag_news` | 127.6K samples | Text classification (4 classes) | datasets/ag_news/ | Used in Khaki et al. for drift detection |
| 20 Newsgroups | scikit-learn | 18.8K samples | Text classification (20 classes) | datasets/20newsgroups/ | Classic NLP benchmark, diverse topics |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| DK-for-TST | github.com/fengliu90/DK-for-TST | Deep kernel MMD tests | code/deep-kernel-tst/ | PyTorch, includes baselines |
| MMDEW | github.com/FlopsKa/mmdew-change-detector | Online MMD change detection | code/mmdew-change-detector/ | Efficient streaming MMD |
| MMDAgg | github.com/antoninschrab/mmdagg | Aggregated MMD test | code/mmdagg/ | Parameter-free, adaptive bandwidth |
| Ripser.py | github.com/scikit-tda/ripser.py | Persistent homology | code/ripser/ | Fast Vietoris-Rips computation |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Started with 6 user-specified arXiv papers (3 had wrong IDs, corrected via Semantic Scholar API)
2. Found correct papers for MMDAgg, deep kernel MMD, C2ST via Semantic Scholar
3. Found TDA-specific papers (Krebs & Rademacher 2024) for persistence diagram testing
4. Searched for code repos on GitHub matching paper implementations
5. Identified key TDA libraries (ripser, giotto-tda, persim)

### Selection Criteria
- Papers directly implementing baseline methods (MMDAgg, deep kernel MMD, C2ST)
- Papers providing theoretical foundation (original MMD, TDA testing)
- Applied papers in our exact domain (text embedding drift detection)
- Code repos with working implementations of baselines

### Challenges Encountered
- 3 of 6 user-specified arXiv IDs mapped to completely unrelated papers
- Paper-finder service was not running; manual search via Semantic Scholar API required
- wget not available in environment; used curl for downloads

### Gaps and Workarounds
- No existing paper directly compares TDA vs MMD/C2ST for embedding drift -- this IS the research gap
- No public code for Krebs & Rademacher (2024) TDA testing -- will implement using ripser + persim
- No paraphrase-based drift benchmarks exist -- will create using back-translation or T5 paraphrasing

## Recommendations for Experiment Design

### 1. Primary Dataset
**AG News** -- directly comparable to Khaki et al. (2023), well-understood category structure enables controlled drift simulation.

### 2. Drift Scenarios
- **Category proportion shift**: Vary class balance between reference and target (controlled, well-understood)
- **Paraphrase-based style shift**: Use back-translation to change writing style without changing topic/semantics (realistic, tests structural drift)
- **Vocabulary drift**: Replace words with synonyms or domain-specific alternatives
- **Noise injection**: Add random perturbation to embeddings (synthetic control)

### 3. Baseline Methods
1. **MMDAgg** (Schrab et al.) -- use code/mmdagg/ directly
2. **Deep kernel MMD** (Liu et al.) -- adapt code/deep-kernel-tst/ for embedding input
3. **C2ST with MLP** (Lopez-Paz & Oquab) -- implement with PyTorch/sklearn
4. **MMD Gaussian (median heuristic)** -- simple baseline from code/deep-kernel-tst/

### 4. Proposed Method
- Compute H0 persistence diagrams using Ripser on batches of embeddings
- Compute Wasserstein-1 distance between persistence diagrams of reference and target
- Use permutation test for significance (permute sample assignments and recompute)

### 5. Embedding Models
- `all-MiniLM-L6-v2` (384-dim) -- fast, used in Khaki et al.
- `intfloat/e5-small-v2` (384-dim) -- modern E5 family
- `all-mpnet-base-v2` (768-dim) -- strong general embeddings

### 6. Evaluation Protocol
- Fix alpha = 0.05, measure test power across drift magnitudes
- Verify Type I error control (no-drift condition)
- Report power curves, computation time
- Use 200+ repetitions per condition for stable estimates
- Sample sizes: 100, 200, 500 per batch (computationally feasible for TDA)

### 7. Key Python Libraries Needed
```
sentence-transformers  # embedding computation
ripser                 # persistent homology
persim                 # persistence diagram distances (Wasserstein)
giotto-tda             # alternative TDA library
torch                  # deep kernel baseline
scikit-learn           # C2ST, utilities
scipy                  # statistical tests
POT                    # optimal transport (Wasserstein)
```
