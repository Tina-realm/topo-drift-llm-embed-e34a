# Literature Review: Topological Drift Detection for LLM Embedding Streams

## Research Area Overview

This research investigates whether topological data analysis (TDA), specifically Wasserstein distance on H0 persistence diagrams, can detect distribution drift in LLM embedding streams more sensitively than modern two-sample tests (MMD-aggregated, deep-kernel MMD, MLP-based C2ST). The key hypothesis is that topological methods excel when drift alters topological structure (e.g., cluster splitting, merging, hole formation) without shifting centroids -- a blind spot for mean-based statistics.

The field sits at the intersection of three areas: (1) kernel-based two-sample testing, (2) concept drift detection in NLP, and (3) topological data analysis for ML.

## Key Papers

### Paper 1: MMD Aggregated Two-Sample Test (MMDAgg)
- **Authors**: Schrab, Kim, Albert, Laurent, Guedj, Gretton
- **Year**: 2023 (JMLR)
- **Source**: arXiv:2110.15073
- **Key Contribution**: Proposes MMDAgg, which aggregates MMD tests over multiple kernel bandwidths, adapting to unknown smoothness. Controls Type I error non-asymptotically (valid even for small samples). Achieves minimax-optimal rates over Sobolev balls.
- **Methodology**: Uses permutation or wild bootstrap to compute test threshold. Aggregates over a collection of bandwidths using a multiple testing correction. Parameter-free adaptive bandwidth collection.
- **Datasets Used**: Synthetic (Gaussian mixtures), MNIST
- **Results**: Outperforms single-bandwidth MMD, median heuristic, and kernel selection approaches. Matches deep-kernel tests on image data.
- **Code Available**: Yes -- github.com/antoninschrab/mmdagg
- **Relevance**: Primary baseline. MMDAgg is the strongest "off-the-shelf" MMD test. Our experiment must show topological methods beat this on structure-altering drift.

### Paper 2: Learning Deep Kernels for Non-Parametric Two-Sample Tests
- **Authors**: Liu, Xu, Lu, Zhang, Gretton, Sutherland
- **Year**: 2020 (ICML)
- **Source**: arXiv:2002.09116
- **Key Contribution**: Uses deep neural networks to learn spatially-varying kernels for MMD tests, trained to maximize test power (MMD^2/variance ratio). Proves consistency of kernel selection.
- **Methodology**: Deep kernel k_w(x,y) = [(1-eps)*kappa(phi_w(x), phi_w(y)) + eps]*q(x,y), where phi_w is a deep net. Trained on held-out split to maximize J = MMD^2/sigma_H1. Includes C2ST as special case.
- **Datasets Used**: Blob (multimodal Gaussian), HDGM, HIGGS, MNIST, CIFAR-10
- **Results**: Deep kernels consistently outperform simple kernels and C2ST. Especially strong for complex, spatially heterogeneous distributions.
- **Code Available**: Yes -- github.com/fengliu90/DK-for-TST
- **Relevance**: Strong baseline. Deep kernels could potentially learn topological features, but our hypothesis is that explicit TDA captures structural drift more efficiently.

### Paper 3: Revisiting Classifier Two-Sample Tests (C2ST)
- **Authors**: Lopez-Paz, Oquab
- **Year**: 2017 (ICLR)
- **Source**: arXiv:1610.06545
- **Key Contribution**: Formalizes binary classifier accuracy as a two-sample test statistic. Under H0, accuracy ~ Binomial(n, 0.5). Simple, interpretable, and learns representations on-the-fly.
- **Methodology**: Train binary classifier to distinguish P from Q samples. Test statistic = classification accuracy on held-out set. Null distribution is N(0.5, 1/(4n)).
- **Datasets Used**: Synthetic, MNIST vs rotated MNIST
- **Results**: Competitive with MMD tests. Interpretable -- predictive uncertainty shows where P and Q differ.
- **Code Available**: No official repo, but trivial to implement with any classifier (MLP, random forest, etc.)
- **Relevance**: Key baseline. MLP-based C2ST is our third baseline. The hypothesis claims TDA outperforms even learned classifiers on structural drift.

### Paper 4: Two-sample Tests for Relevant Differences in Persistence Diagrams
- **Authors**: Krebs, Rademacher
- **Year**: 2024
- **Source**: arXiv:2401.10349
- **Key Contribution**: Develops mathematically rigorous two-sample tests comparing Frechet variances and independent-copy variances of persistence diagrams using Wasserstein metrics. Works with weakly dependent (L^p-m-approximable) data.
- **Methodology**: Compares Frechet variances Var(X) vs Var(Y) of persistence diagrams using self-normalized test statistics. Uses Wasserstein distance W_r on persistence diagram space. Tests for *relevant* differences (threshold Delta) rather than exact equality.
- **Datasets Used**: Theoretical/simulation
- **Results**: Proves functional CLTs for U-statistics on persistence diagrams. Shows consistency of testing procedures.
- **Code Available**: No
- **Relevance**: Directly relevant -- provides the statistical framework for comparing persistence diagram distributions. Our approach uses similar Wasserstein distances on H0 diagrams but as a drift detection statistic rather than a formal two-sample test.

### Paper 5: Uncovering Drift in Textual Data (MMD for Text Drift)
- **Authors**: Khaki, Aditya, Karnin, Ma, Pan, Chandrashekar (Amazon)
- **Year**: 2023
- **Source**: arXiv:2309.03831
- **Key Contribution**: Applied MMD-based drift detection to text embeddings in production ML systems. Shows strong correlation between MMD drift estimates and model performance degradation.
- **Methodology**: Encodes text with BERT/sentence-transformers, computes MMD between reference (training) and target (production) embedding distributions using bootstrap. Identifies high-drift samples for retraining.
- **Datasets Used**: AG News, Yelp Review (+ internal Amazon data)
- **Results**: MMD vs BCE correlation: 76.9%, MMD vs AUC: -65.2%. Drift mitigation (retraining on high-drift samples) reduced false accept rate from 73% to 60%.
- **Code Available**: No
- **Relevance**: Most directly relevant applied work. Uses the same datasets (AG News) and embedding models (MiniLM) we plan to use. Establishes MMD as a practical drift detector for embeddings -- our work extends this by comparing against topological approaches.

### Paper 6: MMDEW - MMD on Exponential Windows for Online Change Detection
- **Authors**: Kalinke, Heyden, Gntuni, Fouche, Bohm
- **Year**: 2025 (updated)
- **Source**: arXiv:2205.12706
- **Key Contribution**: Efficient online MMD computation using exponential windows. O(log^2 t) runtime per observation, O(log t) memory.
- **Methodology**: Maintains exponentially growing windows with log-size samples per window. Approximates quadratic-time MMD in streaming setting. Uses permutation tests for significance.
- **Datasets Used**: Standard change detection benchmarks (5 datasets)
- **Results**: Best F1 on 4/5 datasets vs ADWINK, WATCH, Scan B-Statistics, NEWMA, D3, IBDD.
- **Code Available**: Yes -- github.com/FlopsKa/mmdew-change-detector
- **Relevance**: Provides efficient streaming MMD baseline. Our experiment focuses on batch drift detection, but this shows MMD's effectiveness in the streaming setting.

### Paper 7: Generative Models and Model Criticism via Optimized MMD
- **Authors**: Sutherland, Tung, Strathmann, De, Ramdas, Smola, Gretton
- **Year**: 2017 (ICLR)
- **Source**: arXiv:1611.04488
- **Key Contribution**: Proposes optimizing kernel parameters to maximize MMD test power. Uses ARD kernels to identify which dimensions differ between distributions. Applied to GAN evaluation.
- **Methodology**: Maximizes test power = Phi(MMD^2/sqrt(V) - c/sqrt(mV)) over kernel parameters. Uses permutation tests with efficient block computation.
- **Results**: Optimized MMD provides more interpretable and powerful tests than standard kernels.
- **Code Available**: Referenced in paper
- **Relevance**: Foundational work on optimized MMD that underlies deep kernel approach. The ARD kernel idea is relevant for understanding which embedding dimensions are most affected by drift.

### Paper 8: A Kernel Two-Sample Test (Original MMD)
- **Authors**: Gretton, Borgwardt, Rasch, Scholkopf, Smola
- **Year**: 2012 (JMLR)
- **Source**: arXiv:0805.2368
- **Key Contribution**: The foundational MMD paper. Defines MMD as a metric on probability distributions via kernel mean embeddings in RKHS. Proves consistency and derives asymptotic null distribution.
- **Relevance**: Foundational reference for all MMD-based methods in our experiments.

## Common Methodologies

- **MMD-based tests**: Used in Papers 1, 2, 5, 6, 7, 8. All compare mean embeddings in RKHS. Key design choices: kernel selection (median heuristic, optimized, aggregated, deep), test threshold (permutation, wild bootstrap, asymptotic).
- **Classifier-based tests (C2ST)**: Paper 3. Train binary classifier, use accuracy as test statistic. Null is binomial.
- **Topological tests**: Paper 4. Compute persistence diagrams, compare via Wasserstein distance. Captures shape/structure rather than mean.

## Standard Baselines

1. **MMD with Gaussian kernel (median heuristic)** -- simplest, widely used
2. **MMDAgg** -- strongest off-the-shelf kernel test, adaptive bandwidth
3. **Deep kernel MMD** -- learned kernel, highest power on complex data
4. **C2ST (MLP)** -- classifier-based, interpretable
5. **C2ST (Random Forest)** -- non-neural variant

## Evaluation Metrics

- **Test power** (1 - Type II error rate) at fixed significance level alpha
- **Type I error rate** (should be <= alpha)
- **Detection delay** (for streaming settings)
- **AUC/ROC** for drift vs no-drift classification
- **Sensitivity to drift magnitude** (power curves)

## Datasets in the Literature

- **AG News**: Used in Paper 5 for text drift detection. 120K training, 7.6K test. 4 classes (World, Sports, Business, Sci/Tech).
- **20 Newsgroups**: Classic text classification. 11K train, 7.5K test. 20 categories.
- **MNIST/CIFAR-10**: Used in Papers 1, 2, 3 for image two-sample testing.
- **Synthetic Gaussians**: Used in Papers 1, 2, 3 for controlled experiments.
- **HIGGS**: Used in Paper 2 for high-dimensional testing.

### Paper 9: Measuring Distributional Shifts in Text via LLM Embeddings
- **Authors**: Gupta, Rastegarpanah, Iyer, Rubin, Kenthapadi
- **Year**: 2023
- **Source**: arXiv:2312.02337
- **Key Contribution**: Proposes clustering-based algorithm for measuring distributional shifts in text using LLM embeddings. Shows LLM-based embeddings have higher drift sensitivity than classical embeddings. Deployed in production for 18 months.
- **Methodology**: Cluster reference embeddings, measure how target embeddings distribute across clusters. Compares LLM embeddings vs TF-IDF/Word2Vec for drift sensitivity.
- **Relevance**: Directly relevant -- another applied drift detection system for text embeddings. Validates that embedding choice matters for drift sensitivity.

### Paper 10: Approximation Algorithms for 1-Wasserstein Distance between Persistence Diagrams
- **Authors**: Chen, Wang
- **Year**: 2021
- **Source**: arXiv:2104.07710
- **Key Contribution**: Near-linear time algorithms for computing W1 distance on persistence diagrams using randomly shifted quadtrees. Much faster than exact O(n^3) algorithms.
- **Relevance**: Critical for scalability -- our method needs efficient Wasserstein computation on persistence diagrams. These algorithms make the topological approach computationally feasible for larger sample sizes.

### Paper 11: MMD-FUSE: Learning and Combining Kernels Without Data Splitting
- **Authors**: Biggs, Schrab, Gretton
- **Year**: 2023 (NeurIPS)
- **Source**: arXiv:2306.08777
- **Key Contribution**: Proposes MMD-FUSE which combines multiple kernels via weighted soft maximum without data splitting (unlike deep kernel approach). Proves exponential concentration bounds. Avoids the power loss from train/test splitting.
- **Relevance**: Another strong MMD baseline. Could be considered as an additional comparison point.

## Gaps and Opportunities

1. **No work comparing TDA-based drift detection against modern two-sample tests on embedding streams.** Papers use either TDA or MMD/C2ST, but never compare them directly.
2. **Most drift detection work uses synthetic perturbations (Gaussian noise, rotation).** Our research proposes realistic drift via paraphrase-based style shift, which is more representative of real-world NLP drift.
3. **Topological methods have not been evaluated on high-dimensional LLM embeddings.** Krebs & Rademacher (2024) work with point clouds in R^d, not 384-768 dimensional sentence embeddings.
4. **The hypothesis that structural drift (without centroid shift) favors topological methods is untested.** This is the core research gap.
5. **Multi-model evaluation across embedding families (MiniLM, E5, MPNet) is missing.** Most papers test on one embedding model.

## Recommendations for Our Experiment

### Recommended Datasets
- **AG News** (primary): Well-established, used by Khaki et al. (2023) for drift detection in text. Multiple topical categories enable realistic drift simulation (e.g., shifting category proportions or applying paraphrasing within categories).
- **20 Newsgroups** (secondary): More categories, longer texts, different domain. Good for robustness check.

### Recommended Baselines
1. **MMDAgg** (Schrab et al., 2023) -- strongest kernel-adaptive test
2. **Deep kernel MMD** (Liu et al., 2020) -- learned kernel, high power
3. **C2ST with MLP** (Lopez-Paz & Oquab, 2017) -- classifier-based
4. **MMD with Gaussian kernel, median heuristic** -- simple baseline

### Recommended Metrics
- Test power at alpha = 0.05 across varying drift magnitudes
- Type I error rate (should be <= 0.05)
- Power curves as function of sample size and drift strength
- Computational cost (time per test)

### Recommended Embedding Models
- **all-MiniLM-L6-v2** (384-dim): Fast, compact, used in Khaki et al.
- **e5-base-v2** or **e5-small-v2** (768/384-dim): Modern E5 embeddings
- **all-mpnet-base-v2** (768-dim): Strong general-purpose embeddings

### Methodological Considerations
- Use **Ripser** or **giotto-tda** for persistent homology computation
- Focus on **H0 (connected components)** persistence diagrams as specified in hypothesis
- Compute **Wasserstein-1 distance** between H0 persistence diagrams of reference and target batches
- For fair comparison: use same sample sizes and significance levels across all methods
- Generate realistic drift via **paraphrase models** (e.g., back-translation, T5-based paraphrasing) to change style without changing topic distribution
- Also test with **category proportion shift** (simple, well-understood drift) as a control
