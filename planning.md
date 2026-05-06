# Planning: Topological Drift Detection v3

## Motivation & Novelty Assessment

### Why This Research Matters
Distribution drift in LLM embedding streams is a critical problem for production ML systems. Current drift detectors (MMD, C2ST) rely on distributional moments or classification boundaries, which can miss structural changes that preserve centroids (e.g., cluster splitting, topology changes). Topological Data Analysis (TDA) captures shape-level features that are invariant to these transformations, potentially providing a complementary detection signal.

### Gap in Existing Work
No prior work directly compares TDA-based drift detection against strong modern two-sample tests (MMDAgg, deep-kernel MMD, MLP-C2ST) on sentence embedding streams. Most drift work uses synthetic perturbations; realistic text-level drift (paraphrasing) is understudied. Multi-encoder evaluation across embedding families is missing.

### Our Novel Contribution
1. First systematic comparison of TDA vs. strong two-sample tests on embedding streams
2. Realistic paraphrase-based drift (text-level, not vector-level)
3. Centroid-preserving geometric drift scenarios where TDA should excel
4. Multi-encoder evaluation (MiniLM, MPNet, E5)
5. Proper hold-out FPR calibration with validation windows

### Experiment Justification
- **7 drift scenarios**: Test TDA across diverse drift types, especially centroid-preserving ones
- **3 encoders**: Ensure findings generalize across embedding families
- **Strong baselines**: MMDAgg and deep-kernel MMD are state-of-the-art; beating them is meaningful
- **Text-level paraphrase drift**: Tests realistic NLP drift, not synthetic noise
- **FPR calibration**: Ensures fair comparison at matched false positive rates

## Research Question
Can Wasserstein distance on H0 persistence diagrams detect distribution drift in LLM embedding streams more sensitively than strong modern two-sample tests, especially when drift alters topological structure without shifting centroids?

## Methodology

### Approach
1. Embed text with 3 models, cache to disk
2. Create 7 drift scenarios per dataset
3. Window-based evaluation: 20+ ref windows, 20+ test windows
4. Proper 3-way split: calibration/validation/test
5. Compare TDA features vs. classical baselines via AUC

### Key Parameters
- Window size: 200 samples
- PCA dimensions: 50 (fit on reference only)
- TDA subsample: 100 points via farthest-point sampling
- Vietoris-Rips: cosine distance, max_edge_length=2.0, max_homology_dim=1
- Seeds: [42, 123, 456, 789, 1011] (reduce to 3 if time-constrained)

### Timeline (3 hours total)
- Setup + embedding: 30 min
- Drift scenarios + windowing: 20 min
- Baseline computation: 40 min
- TDA computation: 40 min
- Analysis + visualization: 30 min
- Documentation: 20 min
