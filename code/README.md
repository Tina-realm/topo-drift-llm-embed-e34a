# Cloned Repositories

## Repo 1: Deep Kernel Two-Sample Test (DK-for-TST)
- **URL**: https://github.com/fengliu90/DK-for-TST
- **Purpose**: Implementation of deep kernel MMD two-sample tests (Liu et al., ICML 2020)
- **Location**: code/deep-kernel-tst/
- **Key files**:
  - `utils_HD.py` -- core utilities for MMD computation and deep kernel testing
  - `utils.py` -- general utilities
  - `Deep_Kernel_*.py` -- experiment scripts for different datasets (Blob, HDGM, HIGGS)
  - `Baselines_*.py` -- baseline comparison scripts
- **Dependencies**: PyTorch, NumPy, SciPy
- **Notes**: Provides both deep kernel and baseline (C2ST, standard MMD) implementations. The `utils_HD.py` file contains the key `MMDu` function and the deep kernel architecture. Can be adapted for our embedding drift detection experiments.

## Repo 2: MMDEW Change Detector
- **URL**: https://github.com/FlopsKa/mmdew-change-detector
- **Purpose**: Efficient online MMD-based change detection using exponential windows (Kalinke et al., 2025)
- **Location**: code/mmdew-change-detector/
- **Key files**:
  - `src/` -- source code for MMDEW algorithm
  - `notebooks/` -- example notebooks
  - `pyproject.toml` -- dependencies
- **Dependencies**: NumPy, SciPy
- **Notes**: Focuses on streaming/online change detection. Could provide an alternative efficient MMD implementation. Less relevant for our batch testing setting but useful reference.

## Repo 3: MMDAgg
- **URL**: https://github.com/antoninschrab/mmdagg
- **Purpose**: Implementation of MMD Aggregated Two-Sample Test (Schrab et al., JMLR 2023)
- **Location**: code/mmdagg/
- **Key files**: See repo README for API details
- **Dependencies**: NumPy, SciPy
- **Notes**: Key baseline implementation. Provides parameter-free aggregated MMD test with adaptive bandwidth selection. Should be used directly in experiments.

## Repo 4: Ripser.py
- **URL**: https://github.com/scikit-tda/ripser.py
- **Purpose**: Fast computation of Vietoris-Rips persistent homology
- **Location**: code/ripser/
- **Key files**: Python wrapper around C++ Ripser library
- **Dependencies**: NumPy, SciPy, Cython
- **Notes**: Core TDA library for computing persistence diagrams. Can compute H0 (connected components) and higher homology groups. Essential for the topological drift detection method. Alternative: `giotto-tda` provides similar functionality with sklearn-compatible API.

## Additional Libraries (installable via pip, not cloned)

- **giotto-tda**: Alternative TDA library with sklearn API (`pip install giotto-tda`)
- **persim**: Persistence diagram distances including Wasserstein (`pip install persim`)
- **sentence-transformers**: For computing text embeddings (`pip install sentence-transformers`)
- **POT (Python Optimal Transport)**: For Wasserstein distance computation (`pip install POT`)
