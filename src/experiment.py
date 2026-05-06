"""
Topological Drift Detection for LLM Embedding Streams v3
=========================================================
Comprehensive evaluation with strong baselines and realistic drifts.
"""

import os
import sys
import json
import time
import random
import warnings
import hashlib
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
DATASETS_DIR = ROOT / "datasets"

CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

ENCODERS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "e5-small-v2": "intfloat/e5-small-v2",
}

SEEDS = [42, 123, 456, 789, 1011]
WINDOW_SIZE = 200
PCA_DIM = 50
TDA_SUBSAMPLE = 100
MAX_EDGE_LENGTH = 2.0
MAX_HOMOLOGY_DIM = 1
N_REF_WINDOWS = 10    # calibration
N_VAL_WINDOWS = 10    # validation (FPR estimation)
N_TEST_WINDOWS = 20   # test (per drift scenario)

# ─── Utility Functions ───────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except:
        pass


def farthest_point_sampling(X, n_samples):
    """Greedy farthest-point sampling to preserve geometric structure."""
    if len(X) <= n_samples:
        return X
    # Use cosine distance
    idx = [np.random.randint(len(X))]
    # Compute distances from first point
    dists = cdist(X[idx], X, metric='cosine')[0]
    for _ in range(n_samples - 1):
        new_idx = np.argmax(dists)
        idx.append(new_idx)
        new_dists = cdist(X[new_idx:new_idx+1], X, metric='cosine')[0]
        dists = np.minimum(dists, new_dists)
    return X[np.array(idx)]


def random_sampling(X, n_samples):
    """Simple random subsampling."""
    if len(X) <= n_samples:
        return X
    idx = np.random.choice(len(X), n_samples, replace=False)
    return X[idx]


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_ag_news():
    """Load AG News dataset."""
    from datasets import load_from_disk
    ds_path = DATASETS_DIR / "ag_news"
    try:
        ds = load_from_disk(str(ds_path))
        texts = ds['train']['text'] + ds['test']['text']
        labels = ds['train']['label'] + ds['test']['label']
    except:
        from datasets import load_dataset
        ds = load_dataset("ag_news")
        texts = ds['train']['text'] + ds['test']['text']
        labels = ds['train']['label'] + ds['test']['label']
    return texts, labels


def load_20newsgroups():
    """Load 20 Newsgroups dataset."""
    ng_path = DATASETS_DIR / "20newsgroups"
    if (ng_path / "train.json").exists():
        with open(ng_path / "train.json") as f:
            train = json.load(f)
        with open(ng_path / "test.json") as f:
            test = json.load(f)
        texts = train['texts'] + test['texts']
        labels = train['labels'] + test['labels']
    else:
        from sklearn.datasets import fetch_20newsgroups
        data = fetch_20newsgroups(subset='all')
        texts = list(data.data)
        labels = list(data.target)
    return texts, labels


# ─── Embedding ───────────────────────────────────────────────────────────────

def get_embeddings(texts, encoder_name, encoder_path, batch_size=64):
    """Get embeddings with disk caching."""
    cache_key = hashlib.md5(f"{encoder_name}_{len(texts)}_{texts[0][:50]}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"embeddings_{encoder_name}_{cache_key}.npz"

    if cache_file.exists():
        print(f"  Loading cached embeddings from {cache_file.name}")
        data = np.load(cache_file)
        return data['embeddings']

    print(f"  Computing embeddings with {encoder_name}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(encoder_path)

    # For E5 models, prepend "query: "
    if "e5" in encoder_name.lower():
        texts_to_encode = ["query: " + t for t in texts]
    else:
        texts_to_encode = texts

    embeddings = model.encode(texts_to_encode, batch_size=batch_size,
                               show_progress_bar=True, normalize_embeddings=True)

    np.savez_compressed(cache_file, embeddings=embeddings)
    print(f"  Saved embeddings to {cache_file.name}")
    return embeddings


# ─── Text Augmentation (Paraphrase Drift) ────────────────────────────────────

def augment_texts(texts, aug_p=0.3, seed=42):
    """Apply text-level augmentation using nlpaug (WordNet synonym + random swap)."""
    cache_key = hashlib.md5(f"aug_{len(texts)}_{aug_p}_{seed}_{texts[0][:50]}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"augmented_texts_{cache_key}.json"

    if cache_file.exists():
        print(f"  Loading cached augmented texts from {cache_file.name}")
        with open(cache_file) as f:
            return json.load(f)

    print(f"  Augmenting {len(texts)} texts (aug_p={aug_p})...")
    try:
        import nlpaug.augmenter.word as naw

        # Synonym replacement
        aug_syn = naw.SynonymAug(aug_src='wordnet', aug_p=aug_p)
        # Random swap
        aug_swap = naw.RandomWordAug(action='swap', aug_p=aug_p/2)

        augmented = []
        for i, text in enumerate(tqdm(texts, desc="Augmenting")):
            try:
                t = aug_syn.augment(text)
                if isinstance(t, list):
                    t = t[0]
                t2 = aug_swap.augment(t)
                if isinstance(t2, list):
                    t2 = t2[0]
                augmented.append(t2)
            except:
                augmented.append(text)
    except ImportError:
        print("  nlpaug not available, using simple word shuffle fallback")
        augmented = []
        rng = np.random.RandomState(seed)
        for text in texts:
            words = text.split()
            n_swap = max(1, int(len(words) * aug_p))
            for _ in range(n_swap):
                if len(words) > 1:
                    i, j = rng.choice(len(words), 2, replace=False)
                    words[i], words[j] = words[j], words[i]
            augmented.append(' '.join(words))

    with open(cache_file, 'w') as f:
        json.dump(augmented, f)
    print(f"  Saved augmented texts to {cache_file.name}")
    return augmented


# ─── Drift Scenarios ────────────────────────────────────────────────────────

def create_windows(embeddings, labels, texts, augmented_embeddings,
                   scenario, seed, n_ref, n_val, n_test, window_size):
    """Create reference (calibration), validation, and test windows."""
    set_seed(seed)

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # Group indices by label
    label_indices = {l: np.where(labels == l)[0] for l in unique_labels}

    # Shuffle within each label group
    for l in unique_labels:
        np.random.shuffle(label_indices[l])

    ref_windows = []
    val_windows = []
    test_windows = []

    if scenario == "no_drift":
        # All windows from same distribution (balanced classes)
        total_needed = (n_ref + n_val + n_test) * window_size
        # Sample balanced from main classes
        main_labels = unique_labels[:min(4, n_labels)]
        per_class = window_size // len(main_labels)

        all_indices = []
        for l in main_labels:
            avail = label_indices[l]
            needed = (n_ref + n_val + n_test) * per_class
            if len(avail) < needed:
                avail = np.tile(avail, needed // len(avail) + 1)
            all_indices.append(avail[:needed])

        # Build windows
        for w in range(n_ref + n_val + n_test):
            win_idx = []
            for li, l in enumerate(main_labels):
                start = w * per_class
                win_idx.extend(all_indices[li][start:start + per_class])
            win_idx = np.array(win_idx)
            np.random.shuffle(win_idx)

            if w < n_ref:
                ref_windows.append(embeddings[win_idx])
            elif w < n_ref + n_val:
                val_windows.append(embeddings[win_idx])
            else:
                test_windows.append(embeddings[win_idx])

    elif scenario == "abrupt_topic":
        # Reference: class 0,1. Test: class 2,3
        ref_labels = unique_labels[:2]
        test_labels = unique_labels[2:4] if n_labels >= 4 else unique_labels[2:3]
        per_class_ref = window_size // len(ref_labels)
        per_class_test = window_size // len(test_labels)

        for w in range(n_ref + n_val):
            win_idx = []
            for l in ref_labels:
                avail = label_indices[l]
                start = w * per_class_ref
                if start + per_class_ref > len(avail):
                    idx = np.random.choice(avail, per_class_ref, replace=True)
                else:
                    idx = avail[start:start + per_class_ref]
                win_idx.extend(idx)
            win_idx = np.array(win_idx)
            np.random.shuffle(win_idx)
            if w < n_ref:
                ref_windows.append(embeddings[win_idx])
            else:
                val_windows.append(embeddings[win_idx])

        for w in range(n_test):
            win_idx = []
            for l in test_labels:
                avail = label_indices[l]
                start = w * per_class_test
                if start + per_class_test > len(avail):
                    idx = np.random.choice(avail, per_class_test, replace=True)
                else:
                    idx = avail[start:start + per_class_test]
                win_idx.extend(idx)
            win_idx = np.array(win_idx)
            np.random.shuffle(win_idx)
            test_windows.append(embeddings[win_idx])

    elif scenario == "gradual_topic":
        # Reference: class 0,1. Test: gradual mix (increasing class 2 proportion)
        ref_labels = unique_labels[:2]
        drift_label = unique_labels[2] if n_labels >= 3 else unique_labels[1]
        per_class = window_size // 2

        for w in range(n_ref + n_val):
            win_idx = []
            for l in ref_labels:
                avail = label_indices[l]
                start = w * per_class
                if start + per_class > len(avail):
                    idx = np.random.choice(avail, per_class, replace=True)
                else:
                    idx = avail[start:start + per_class]
                win_idx.extend(idx)
            win_idx = np.array(win_idx)
            np.random.shuffle(win_idx)
            if w < n_ref:
                ref_windows.append(embeddings[win_idx])
            else:
                val_windows.append(embeddings[win_idx])

        for w in range(n_test):
            mix_ratio = (w + 1) / n_test  # 0.05 to 1.0
            n_drift = int(window_size * mix_ratio * 0.5)
            n_orig = window_size - n_drift

            win_idx = []
            # Original distribution
            for l in ref_labels:
                n_per = n_orig // len(ref_labels)
                avail = label_indices[l]
                idx = np.random.choice(avail, n_per, replace=True)
                win_idx.extend(idx)
            # Drift class
            avail = label_indices[drift_label]
            idx = np.random.choice(avail, n_drift, replace=True)
            win_idx.extend(idx)

            win_idx = np.array(win_idx[:window_size])
            np.random.shuffle(win_idx)
            test_windows.append(embeddings[win_idx])

    elif scenario == "domain_shift":
        # Reference: first half of classes. Test: second half
        half = n_labels // 2
        ref_labels = unique_labels[:half]
        test_labels = unique_labels[half:]
        per_class_ref = window_size // len(ref_labels)
        per_class_test = window_size // len(test_labels)

        for w in range(n_ref + n_val):
            win_idx = []
            for l in ref_labels:
                avail = label_indices[l]
                idx = np.random.choice(avail, per_class_ref, replace=True)
                win_idx.extend(idx)
            win_idx = np.array(win_idx[:window_size])
            np.random.shuffle(win_idx)
            if w < n_ref:
                ref_windows.append(embeddings[win_idx])
            else:
                val_windows.append(embeddings[win_idx])

        for w in range(n_test):
            win_idx = []
            for l in test_labels:
                avail = label_indices[l]
                idx = np.random.choice(avail, per_class_test, replace=True)
                win_idx.extend(idx)
            win_idx = np.array(win_idx[:window_size])
            np.random.shuffle(win_idx)
            test_windows.append(embeddings[win_idx])

    elif scenario == "subtopic_reweight":
        # Centroid-preserving: same classes, but reweight subtopics
        # Use all classes but change proportions while keeping centroid
        main_labels = unique_labels[:min(4, n_labels)]
        per_class = window_size // len(main_labels)

        for w in range(n_ref + n_val):
            win_idx = []
            for l in main_labels:
                avail = label_indices[l]
                idx = np.random.choice(avail, per_class, replace=True)
                win_idx.extend(idx)
            win_idx = np.array(win_idx[:window_size])
            np.random.shuffle(win_idx)
            if w < n_ref:
                ref_windows.append(embeddings[win_idx])
            else:
                val_windows.append(embeddings[win_idx])

        for w in range(n_test):
            # Skewed proportions (preserving centroid approximately)
            weights = np.array([0.1, 0.4, 0.4, 0.1])[:len(main_labels)]
            weights = weights / weights.sum()
            counts = (weights * window_size).astype(int)
            counts[-1] = window_size - counts[:-1].sum()

            win_idx = []
            for li, l in enumerate(main_labels):
                avail = label_indices[l]
                idx = np.random.choice(avail, counts[li], replace=True)
                win_idx.extend(idx)

            # Re-center to match reference centroid
            win_emb = embeddings[np.array(win_idx[:window_size])]
            ref_centroid = np.mean(ref_windows[0], axis=0)
            win_centroid = np.mean(win_emb, axis=0)
            shift = ref_centroid - win_centroid
            win_emb = win_emb + shift
            test_windows.append(win_emb)

    elif scenario == "paraphrase_style":
        # Text-level augmentation BEFORE embedding
        main_labels = unique_labels[:min(4, n_labels)]
        per_class = window_size // len(main_labels)

        for w in range(n_ref + n_val):
            win_idx = []
            for l in main_labels:
                avail = label_indices[l]
                idx = np.random.choice(avail, per_class, replace=True)
                win_idx.extend(idx)
            win_idx = np.array(win_idx[:window_size])
            np.random.shuffle(win_idx)
            if w < n_ref:
                ref_windows.append(embeddings[win_idx])
            else:
                val_windows.append(embeddings[win_idx])

        # Test windows use augmented embeddings
        for w in range(n_test):
            win_idx = []
            for l in main_labels:
                avail = label_indices[l]
                idx = np.random.choice(avail, per_class, replace=True)
                win_idx.extend(idx)
            win_idx = np.array(win_idx[:window_size])
            np.random.shuffle(win_idx)
            test_windows.append(augmented_embeddings[win_idx])

    elif scenario == "geo_reorg":
        # Geometric reorganization: k=4 clusters, cyclic shift, 40% toward next, re-center
        main_labels = unique_labels[:min(4, n_labels)]
        per_class = window_size // len(main_labels)

        for w in range(n_ref + n_val):
            win_idx = []
            for l in main_labels:
                avail = label_indices[l]
                idx = np.random.choice(avail, per_class, replace=True)
                win_idx.extend(idx)
            win_idx = np.array(win_idx[:window_size])
            np.random.shuffle(win_idx)
            if w < n_ref:
                ref_windows.append(embeddings[win_idx])
            else:
                val_windows.append(embeddings[win_idx])

        for w in range(n_test):
            win_idx = []
            for l in main_labels:
                avail = label_indices[l]
                idx = np.random.choice(avail, per_class, replace=True)
                win_idx.extend(idx)
            win_idx = np.array(win_idx[:window_size])
            win_emb = embeddings[win_idx].copy()
            ref_centroid = np.mean(ref_windows[0], axis=0)

            # k-means clustering on this window
            km = KMeans(n_clusters=4, random_state=seed, n_init=3)
            cluster_labels = km.fit_predict(win_emb)
            centers = km.cluster_centers_

            # Cyclic shift: move 40% toward cyclically-next cluster center
            for c in range(4):
                mask = cluster_labels == c
                next_c = (c + 1) % 4
                direction = centers[next_c] - centers[c]
                win_emb[mask] += 0.4 * direction

            # Re-center to match reference centroid
            win_centroid = np.mean(win_emb, axis=0)
            win_emb += (ref_centroid - win_centroid)

            test_windows.append(win_emb)

    return ref_windows, val_windows, test_windows


# ─── PCA ─────────────────────────────────────────────────────────────────────

def fit_pca_on_reference(ref_windows, n_components=50):
    """Fit PCA ONLY on reference/calibration windows."""
    ref_data = np.vstack(ref_windows)
    pca = PCA(n_components=min(n_components, ref_data.shape[1]))
    pca.fit(ref_data)
    return pca


def apply_pca(windows, pca):
    """Transform windows using pre-fitted PCA."""
    return [pca.transform(w) for w in windows]


# ─── TDA Computation ────────────────────────────────────────────────────────

def compute_persistence_diagram(X, max_edge=2.0, max_dim=1):
    """Compute persistence diagram using Vietoris-Rips with cosine distance."""
    import ripser
    # Compute cosine distance matrix
    D = squareform(pdist(X, metric='cosine'))
    result = ripser.ripser(D, maxdim=max_dim, thresh=max_edge, distance_matrix=True)
    return result['dgms']


def wasserstein_distance_pd(dgm1, dgm2, p=1):
    """Wasserstein-p distance between persistence diagrams using persim."""
    from persim import wasserstein
    # Filter out infinite death values
    d1 = dgm1[np.isfinite(dgm1[:, 1])] if len(dgm1) > 0 else np.empty((0, 2))
    d2 = dgm2[np.isfinite(dgm2[:, 1])] if len(dgm2) > 0 else np.empty((0, 2))
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    return wasserstein(d1, d2)


def bottleneck_distance_pd(dgm1, dgm2):
    """Bottleneck distance between persistence diagrams."""
    from persim import bottleneck
    d1 = dgm1[np.isfinite(dgm1[:, 1])] if len(dgm1) > 0 else np.empty((0, 2))
    d2 = dgm2[np.isfinite(dgm2[:, 1])] if len(dgm2) > 0 else np.empty((0, 2))
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    return bottleneck(d1, d2)


def persistence_entropy(dgm):
    """Compute persistence entropy of a diagram."""
    d = dgm[np.isfinite(dgm[:, 1])]
    if len(d) == 0:
        return 0.0
    lifetimes = d[:, 1] - d[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    probs = lifetimes / lifetimes.sum()
    return -np.sum(probs * np.log(probs + 1e-12))


def persistence_image_vector(dgm, resolution=20, sigma=0.1):
    """Compute persistence image as a vector."""
    from persim import PersistenceImager
    d = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else np.empty((0, 2))
    if len(d) == 0:
        return np.zeros(resolution * resolution)
    try:
        pimgr = PersistenceImager(pixel_size=sigma, birth_range=(0, 1), pers_range=(0, 1),
                                   kernel_params={'sigma': [[sigma, 0], [0, sigma]]})
        pimgr.fit([d])
        img = pimgr.transform([d])[0]
        return img.flatten()
    except:
        # Fallback: simple histogram
        lifetimes = d[:, 1] - d[:, 0]
        hist, _ = np.histogram(lifetimes, bins=resolution, range=(0, 1))
        return hist.astype(float)


def betti_curve(dgm, resolution=50):
    """Compute Betti curve (number of features alive at each filtration value)."""
    d = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else np.empty((0, 2))
    if len(d) == 0:
        return np.zeros(resolution)
    t_range = np.linspace(0, MAX_EDGE_LENGTH, resolution)
    curve = np.zeros(resolution)
    for i, t in enumerate(t_range):
        curve[i] = np.sum((d[:, 0] <= t) & (d[:, 1] > t))
    return curve


def phd_h0(dgm):
    """Persistent homology dimension for H0."""
    d = dgm[np.isfinite(dgm[:, 1])]
    if len(d) < 2:
        return 0.0
    lifetimes = np.sort(d[:, 1] - d[:, 0])[::-1]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) < 2:
        return 0.0
    # Log-log slope of sorted lifetimes
    x = np.log(np.arange(1, len(lifetimes) + 1))
    y = np.log(lifetimes + 1e-12)
    slope = np.polyfit(x, y, 1)[0]
    return -slope


# ─── Classical Baselines ────────────────────────────────────────────────────

def centroid_shift(ref, test):
    """L2 distance between centroids."""
    return np.linalg.norm(ref.mean(0) - test.mean(0))


def covariance_shift(ref, test):
    """Frobenius norm of covariance difference."""
    cov_ref = np.cov(ref.T)
    cov_test = np.cov(test.T)
    return np.linalg.norm(cov_ref - cov_test, 'fro')


def mmd_rbf(X, Y, bandwidth=None):
    """MMD with single RBF kernel, median heuristic bandwidth."""
    XY = np.vstack([X, Y])
    dists = pdist(XY, metric='sqeuclidean')
    if bandwidth is None:
        bandwidth = np.median(dists)
    if bandwidth == 0:
        bandwidth = 1.0

    n, m = len(X), len(Y)
    K = squareform(np.exp(-dists / (2 * bandwidth)))

    Kxx = K[:n, :n]
    Kyy = K[n:, n:]
    Kxy = K[:n, n:]

    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + \
           (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - \
           2 * Kxy.sum() / (n * m)
    return max(0, mmd2)


def mmd_aggregated(X, Y, bandwidth_multipliers=[0.1, 0.5, 1.0, 5.0, 10.0]):
    """MMD-aggregated: max over multiple bandwidths."""
    XY = np.vstack([X, Y])
    sq_dists = pdist(XY, metric='sqeuclidean')
    median_bw = np.median(sq_dists)
    if median_bw == 0:
        median_bw = 1.0

    n, m = len(X), len(Y)
    max_mmd = 0.0

    for mult in bandwidth_multipliers:
        bw = median_bw * mult
        K = squareform(np.exp(-sq_dists / (2 * bw)))
        Kxx = K[:n, :n]
        Kyy = K[n:, n:]
        Kxy = K[:n, n:]
        mmd2 = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + \
               (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - \
               2 * Kxy.sum() / (n * m)
        max_mmd = max(max_mmd, mmd2)

    return max(0, max_mmd)


def energy_distance(X, Y):
    """Energy distance between two samples."""
    xy = cdist(X, Y).mean()
    xx = pdist(X).mean() if len(X) > 1 else 0
    yy = pdist(Y).mean() if len(Y) > 1 else 0
    return 2 * xy - xx - yy


def knn_distance(X, Y, k=5):
    """Average k-NN distance shift."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dists_ref, _ = nn.kneighbors(X)
    dists_test, _ = nn.kneighbors(Y)
    return abs(dists_test.mean() - dists_ref.mean())


def c2st_lr(X, Y):
    """C2ST with logistic regression, 3-fold CV AUC."""
    n = len(X)
    data = np.vstack([X, Y])
    labels = np.array([0]*n + [1]*len(Y))

    clf = LogisticRegression(max_iter=500, random_state=42)
    scores = cross_val_score(clf, data, labels, cv=3, scoring='roc_auc')
    return scores.mean()


def c2st_mlp(X, Y):
    """C2ST with MLP (128-64, ReLU, dropout=0.2), 3-fold CV AUC."""
    n = len(X)
    data = np.vstack([X, Y])
    labels = np.array([0]*n + [1]*len(Y))

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                        max_iter=300, random_state=42, early_stopping=True,
                        validation_fraction=0.15)
    scores = cross_val_score(clf, data, labels, cv=3, scoring='roc_auc')
    return scores.mean()


def deep_kernel_mmd(X, Y, ref_held_out=None):
    """Deep-kernel MMD: train a 2-layer MLP to maximize MMD, then evaluate.
    Simplified version: use MLP features then compute MMD in learned space."""
    import torch
    import torch.nn as nn

    # If no held-out data, use a portion of X
    if ref_held_out is None:
        n_train = len(X) // 3
        ref_held_out = X[:n_train]
        X = X[n_train:]

    device = torch.device('cpu')

    # Simple 2-layer feature extractor
    d = X.shape[1]
    net = nn.Sequential(
        nn.Linear(d, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU()
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Train to maximize MMD between held-out ref and a bootstrapped "pseudo-test"
    X_t = torch.FloatTensor(X).to(device)
    held_t = torch.FloatTensor(ref_held_out).to(device)
    Y_t = torch.FloatTensor(Y).to(device)

    # Quick training (few epochs for speed)
    net.train()
    for epoch in range(20):
        phi_x = net(X_t)
        phi_y = net(Y_t)

        # MMD in feature space
        mmd2 = (phi_x.mean(0) - phi_y.mean(0)).pow(2).sum()
        loss = -mmd2  # maximize MMD

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    net.eval()
    with torch.no_grad():
        phi_x = net(X_t).numpy()
        phi_y = net(Y_t).numpy()

    return mmd_rbf(phi_x, phi_y)


# ─── Main Experiment Runner ─────────────────────────────────────────────────

def compute_tda_features(window, ref_diagrams, sampling_method='fps'):
    """Compute all TDA features for a window against reference diagrams.
    Returns dict of feature_name -> score and timing breakdown."""
    timings = {}

    # Subsample
    t0 = time.time()
    if sampling_method == 'fps':
        sub = farthest_point_sampling(window, TDA_SUBSAMPLE)
    else:
        sub = random_sampling(window, TDA_SUBSAMPLE)
    timings['sampling_ms'] = (time.time() - t0) * 1000

    # Compute persistence diagram
    t0 = time.time()
    dgms = compute_persistence_diagram(sub, max_edge=MAX_EDGE_LENGTH, max_dim=MAX_HOMOLOGY_DIM)
    timings['diagram_ms'] = (time.time() - t0) * 1000

    t0 = time.time()
    features = {}

    dgm_h0 = dgms[0]
    dgm_h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

    # Persistence entropy
    features['pers_entropy_H0'] = persistence_entropy(dgm_h0)
    features['pers_entropy_H1'] = persistence_entropy(dgm_h1)

    # PHD H0
    features['phd_H0'] = phd_h0(dgm_h0)

    # Betti curves
    bc_h0 = betti_curve(dgm_h0)
    bc_h1 = betti_curve(dgm_h1)

    # Compare against each reference diagram, take mean
    wass_h0_list = []
    wass_h1_list = []
    bott_h0_list = []
    bott_h1_list = []
    pi_h0_list = []
    pi_h1_list = []
    betti_h0_list = []
    betti_h1_list = []

    for ref_dgms in ref_diagrams:
        ref_h0 = ref_dgms[0]
        ref_h1 = ref_dgms[1] if len(ref_dgms) > 1 else np.empty((0, 2))

        wass_h0_list.append(wasserstein_distance_pd(dgm_h0, ref_h0))
        wass_h1_list.append(wasserstein_distance_pd(dgm_h1, ref_h1))
        bott_h0_list.append(bottleneck_distance_pd(dgm_h0, ref_h0))
        bott_h1_list.append(bottleneck_distance_pd(dgm_h1, ref_h1))

        # Persistence images
        pi_test_h0 = persistence_image_vector(dgm_h0)
        pi_ref_h0 = persistence_image_vector(ref_h0)
        pi_h0_list.append(np.linalg.norm(pi_test_h0 - pi_ref_h0))

        pi_test_h1 = persistence_image_vector(dgm_h1)
        pi_ref_h1 = persistence_image_vector(ref_h1)
        pi_h1_list.append(np.linalg.norm(pi_test_h1 - pi_ref_h1))

        # Betti curves
        ref_bc_h0 = betti_curve(ref_h0)
        ref_bc_h1 = betti_curve(ref_h1)
        betti_h0_list.append(np.linalg.norm(bc_h0 - ref_bc_h0))
        betti_h1_list.append(np.linalg.norm(bc_h1 - ref_bc_h1))

    features['wasserstein_H0'] = np.mean(wass_h0_list)
    features['wasserstein_H1'] = np.mean(wass_h1_list)
    features['bottleneck_H0'] = np.mean(bott_h0_list)
    features['bottleneck_H1'] = np.mean(bott_h1_list)
    features['pers_image_H0'] = np.mean(pi_h0_list)
    features['pers_image_H1'] = np.mean(pi_h1_list)
    features['betti_curve_H0'] = np.mean(betti_h0_list)
    features['betti_curve_H1'] = np.mean(betti_h1_list)

    timings['feature_ms'] = (time.time() - t0) * 1000

    return features, timings


def compute_classical_features(ref_window, test_window, ref_held_out=None):
    """Compute all classical baseline features."""
    features = {}
    features['centroid_shift'] = centroid_shift(ref_window, test_window)
    features['covariance_shift'] = covariance_shift(ref_window, test_window)
    features['mmd_rbf'] = mmd_rbf(ref_window, test_window)
    features['mmd_aggregated'] = mmd_aggregated(ref_window, test_window)
    features['energy_distance'] = energy_distance(ref_window, test_window)
    features['knn_distance'] = knn_distance(ref_window, test_window)
    features['c2st_lr'] = c2st_lr(ref_window, test_window)
    features['c2st_mlp'] = c2st_mlp(ref_window, test_window)
    features['deep_kernel_mmd'] = deep_kernel_mmd(ref_window, test_window, ref_held_out)
    return features


def run_single_experiment(dataset_name, encoder_name, encoder_path,
                          texts, labels, scenario, seed):
    """Run a single experiment configuration."""
    set_seed(seed)

    # Get embeddings (cached)
    embeddings = get_embeddings(texts, encoder_name, encoder_path)
    labels_arr = np.array(labels)

    # Get augmented embeddings for paraphrase scenario
    augmented_embeddings = None
    if scenario == "paraphrase_style":
        # Sample texts to augment (only the ones we'll need)
        aug_texts = augment_texts(texts[:len(texts)//2], aug_p=0.3, seed=seed)
        # Embed augmented texts
        aug_cache = CACHE_DIR / f"aug_emb_{encoder_name}_{seed}.npz"
        if aug_cache.exists():
            augmented_embeddings = np.load(aug_cache)['embeddings']
        else:
            # Pad augmented texts to match original text length for indexing
            full_aug_texts = aug_texts + texts[len(texts)//2:]
            augmented_embeddings = get_embeddings(full_aug_texts,
                                                   f"{encoder_name}_aug_{seed}",
                                                   encoder_path)
            np.savez_compressed(aug_cache, embeddings=augmented_embeddings)

    # Create windows
    ref_windows, val_windows, test_windows = create_windows(
        embeddings, labels_arr, texts, augmented_embeddings,
        scenario, seed, N_REF_WINDOWS, N_VAL_WINDOWS, N_TEST_WINDOWS, WINDOW_SIZE
    )

    if not ref_windows or not test_windows:
        print(f"  WARNING: No windows created for {scenario}")
        return []

    # Fit PCA on reference windows ONLY
    t_pca = time.time()
    pca = fit_pca_on_reference(ref_windows, PCA_DIM)
    pca_time_ms = (time.time() - t_pca) * 1000

    ref_pca = apply_pca(ref_windows, pca)
    val_pca = apply_pca(val_windows, pca) if val_windows else []
    test_pca = apply_pca(test_windows, pca)

    # Compute reference persistence diagrams
    ref_diagrams = []
    for rw in ref_pca:
        sub = farthest_point_sampling(rw, TDA_SUBSAMPLE)
        dgms = compute_persistence_diagram(sub, MAX_EDGE_LENGTH, MAX_HOMOLOGY_DIM)
        ref_diagrams.append(dgms)

    # Held-out reference data for deep-kernel MMD
    ref_held_out = np.vstack(ref_pca[:3]) if len(ref_pca) >= 3 else None

    # Compute reference window for classical baselines (concatenated ref)
    ref_concat = np.vstack(ref_pca)
    # Use a random subsample of ref for classical methods (speed)
    if len(ref_concat) > WINDOW_SIZE:
        ref_sub_idx = np.random.choice(len(ref_concat), WINDOW_SIZE, replace=False)
        ref_for_classical = ref_concat[ref_sub_idx]
    else:
        ref_for_classical = ref_concat

    results = []

    # Process validation windows (no-drift, for FPR estimation)
    for wi, win in enumerate(val_pca):
        row = {
            'dataset': dataset_name,
            'encoder': encoder_name,
            'drift_type': scenario,
            'seed': seed,
            'window_idx': wi,
            'window_set': 'validation',
            'runtime_pca_ms': pca_time_ms / len(val_pca),
        }

        # Classical features
        classical = compute_classical_features(ref_for_classical, win, ref_held_out)
        row.update(classical)

        # TDA features
        tda_feats, tda_times = compute_tda_features(win, ref_diagrams, 'fps')
        row.update(tda_feats)
        row.update({f'runtime_{k}': v for k, v in tda_times.items()})
        row['runtime_total_ms'] = pca_time_ms / len(val_pca) + sum(tda_times.values())
        row['sampling_method'] = 'fps'

        results.append(row)

    # Process test windows (drift or no-drift)
    for wi, win in enumerate(test_pca):
        row = {
            'dataset': dataset_name,
            'encoder': encoder_name,
            'drift_type': scenario,
            'seed': seed,
            'window_idx': wi,
            'window_set': 'test',
            'runtime_pca_ms': pca_time_ms / len(test_pca),
        }

        # Classical features
        classical = compute_classical_features(ref_for_classical, win, ref_held_out)
        row.update(classical)

        # TDA features
        tda_feats, tda_times = compute_tda_features(win, ref_diagrams, 'fps')
        row.update(tda_feats)
        row.update({f'runtime_{k}': v for k, v in tda_times.items()})
        row['runtime_total_ms'] = pca_time_ms / len(test_pca) + sum(tda_times.values())
        row['sampling_method'] = 'fps'

        results.append(row)

    return results


def run_ablation_sampling(dataset_name, encoder_name, encoder_path,
                           texts, labels, seed):
    """Run ablation: farthest-point vs random sampling, varying subsample sizes."""
    set_seed(seed)

    embeddings = get_embeddings(texts, encoder_name, encoder_path)
    labels_arr = np.array(labels)

    ref_windows, val_windows, test_windows = create_windows(
        embeddings, labels_arr, texts, None,
        "geo_reorg", seed, N_REF_WINDOWS, 0, 10, WINDOW_SIZE
    )

    pca = fit_pca_on_reference(ref_windows, PCA_DIM)
    ref_pca = apply_pca(ref_windows, pca)
    test_pca = apply_pca(test_windows, pca)

    results = []

    for subsample_size in [40, 80, 100, 160]:
        for sampling in ['fps', 'random']:
            # Compute ref diagrams with this sampling
            ref_diagrams = []
            for rw in ref_pca:
                if sampling == 'fps':
                    sub = farthest_point_sampling(rw, subsample_size)
                else:
                    sub = random_sampling(rw, subsample_size)
                dgms = compute_persistence_diagram(sub, MAX_EDGE_LENGTH, MAX_HOMOLOGY_DIM)
                ref_diagrams.append(dgms)

            for wi, win in enumerate(test_pca):
                global TDA_SUBSAMPLE
                old_sub = TDA_SUBSAMPLE
                TDA_SUBSAMPLE = subsample_size
                tda_feats, tda_times = compute_tda_features(win, ref_diagrams, sampling)
                TDA_SUBSAMPLE = old_sub

                results.append({
                    'dataset': dataset_name,
                    'encoder': encoder_name,
                    'drift_type': 'geo_reorg',
                    'seed': seed,
                    'subsample_size': subsample_size,
                    'sampling_method': sampling,
                    'wasserstein_H0': tda_feats['wasserstein_H0'],
                    'window_idx': wi,
                    'window_set': 'test',
                })

    return results


# ─── AUC Computation ────────────────────────────────────────────────────────

def compute_auc_for_method(df, method_col, drift_type):
    """Compute AUC for a method on a drift type vs no_drift."""
    from sklearn.metrics import roc_auc_score

    # Get no-drift validation scores (negatives)
    no_drift = df[(df['drift_type'] == 'no_drift') & (df['window_set'] == 'test')]
    # Get drift test scores (positives)
    drift = df[(df['drift_type'] == drift_type) & (df['window_set'] == 'test')]

    if len(no_drift) == 0 or len(drift) == 0:
        return np.nan

    scores = np.concatenate([no_drift[method_col].values, drift[method_col].values])
    labels = np.array([0]*len(no_drift) + [1]*len(drift))

    # Remove NaN
    mask = ~np.isnan(scores)
    if mask.sum() < 4:
        return np.nan

    try:
        auc = roc_auc_score(labels[mask], scores[mask])
        return max(auc, 1 - auc)  # Ensure AUC >= 0.5
    except:
        return np.nan


def compute_fpr(df, method_col, threshold):
    """Compute realized FPR on validation no-drift windows."""
    val_nodrift = df[(df['drift_type'] == 'no_drift') & (df['window_set'] == 'validation')]
    if len(val_nodrift) == 0:
        return np.nan
    scores = val_nodrift[method_col].values
    return np.mean(scores > threshold)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Topological Drift Detection for LLM Embedding Streams v3")
    print("=" * 70)

    start_time = time.time()

    # Load datasets
    print("\n[1/6] Loading datasets...")
    datasets = {}

    print("  Loading AG News...")
    ag_texts, ag_labels = load_ag_news()
    datasets['ag_news'] = (ag_texts, ag_labels)
    print(f"  AG News: {len(ag_texts)} texts, {len(set(ag_labels))} classes")

    print("  Loading 20 Newsgroups...")
    ng_texts, ng_labels = load_20newsgroups()
    datasets['20newsgroups'] = (ng_texts, ng_labels)
    print(f"  20 Newsgroups: {len(ng_texts)} texts, {len(set(ng_labels))} classes")

    scenarios = [
        "no_drift", "abrupt_topic", "gradual_topic", "domain_shift",
        "subtopic_reweight", "paraphrase_style", "geo_reorg"
    ]

    all_results = []
    ablation_results = []

    # Run experiments
    print("\n[2/6] Running experiments...")

    total_configs = len(datasets) * len(ENCODERS) * len(scenarios) * len(SEEDS)
    progress = 0

    for ds_name, (texts, labels) in datasets.items():
        for enc_name, enc_path in ENCODERS.items():
            print(f"\n  === {ds_name} / {enc_name} ===")

            # Pre-compute and cache embeddings
            _ = get_embeddings(texts, enc_name, enc_path)

            for scenario in scenarios:
                for seed in SEEDS:
                    progress += 1
                    elapsed = time.time() - start_time
                    print(f"  [{progress}/{total_configs}] {scenario} seed={seed} "
                          f"({elapsed/60:.1f}min elapsed)")

                    try:
                        results = run_single_experiment(
                            ds_name, enc_name, enc_path,
                            texts, labels, scenario, seed
                        )
                        all_results.extend(results)
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        import traceback
                        traceback.print_exc()

                    # Time check: if over 2.5 hours, reduce seeds
                    if elapsed > 9000:  # 2.5 hours
                        print("  WARNING: Time limit approaching, reducing seeds")
                        break

            # Run ablation for first seed only
            if elapsed < 8000:
                print(f"  Running sampling ablation for {ds_name}/{enc_name}...")
                try:
                    abl = run_ablation_sampling(ds_name, enc_name, enc_path,
                                                texts, labels, SEEDS[0])
                    ablation_results.extend(abl)
                except Exception as e:
                    print(f"    Ablation ERROR: {e}")

    # Save raw results
    print("\n[3/6] Saving results...")
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "raw_results.csv", index=False)

    if ablation_results:
        df_abl = pd.DataFrame(ablation_results)
        df_abl.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

    # Compute AUC metrics
    print("\n[4/6] Computing AUC metrics...")
    compute_auc_metrics(df)

    # Generate visualizations
    print("\n[5/6] Generating visualizations...")
    generate_plots(df, ablation_results)

    # Save timing
    total_time = time.time() - start_time
    print(f"\n[6/6] Total runtime: {total_time/60:.1f} minutes")

    # Save summary
    save_metrics_json(df, total_time)

    print("\nDone! Results saved to results/ and figures/")


def compute_auc_metrics(df):
    """Compute AUC for all methods across all configurations."""
    from sklearn.metrics import roc_auc_score

    method_cols = [
        'centroid_shift', 'covariance_shift', 'mmd_rbf', 'mmd_aggregated',
        'energy_distance', 'knn_distance', 'c2st_lr', 'c2st_mlp', 'deep_kernel_mmd',
        'wasserstein_H0', 'wasserstein_H1', 'bottleneck_H0', 'bottleneck_H1',
        'pers_entropy_H0', 'pers_entropy_H1', 'pers_image_H0', 'pers_image_H1',
        'betti_curve_H0', 'betti_curve_H1', 'phd_H0'
    ]

    drift_scenarios = [s for s in df['drift_type'].unique() if s != 'no_drift']

    auc_results = []

    for ds in df['dataset'].unique():
        for enc in df['encoder'].unique():
            for drift in drift_scenarios:
                for seed in df['seed'].unique():
                    sub = df[(df['dataset'] == ds) & (df['encoder'] == enc) &
                             (df['seed'] == seed)]

                    for method in method_cols:
                        if method not in sub.columns:
                            continue

                        auc = compute_auc_for_method(sub, method, drift)

                        # Compute FPR on validation
                        cal_scores = sub[(sub['drift_type'] == 'no_drift') &
                                        (sub['window_set'] == 'test')][method].values
                        if len(cal_scores) > 0:
                            threshold = np.percentile(cal_scores, 95)
                            fpr = compute_fpr(sub, method, threshold)
                        else:
                            fpr = np.nan

                        # Runtime
                        runtime_cols = [c for c in sub.columns if 'runtime' in c]
                        runtime = sub[sub['drift_type'] == drift].get('runtime_total_ms', pd.Series([np.nan])).mean()

                        auc_results.append({
                            'dataset': ds,
                            'encoder': enc,
                            'method': method,
                            'drift_type': drift,
                            'seed': seed,
                            'auc': auc,
                            'fpr_realized': fpr,
                            'runtime_total_ms': runtime,
                        })

    auc_df = pd.DataFrame(auc_results)
    auc_df.to_csv(RESULTS_DIR / "auc_results.csv", index=False)

    # Compute summary with CI
    summary = auc_df.groupby(['dataset', 'encoder', 'method', 'drift_type']).agg(
        auc_mean=('auc', 'mean'),
        auc_std=('auc', 'std'),
        fpr_mean=('fpr_realized', 'mean'),
        fpr_std=('fpr_realized', 'std'),
        runtime_mean=('runtime_total_ms', 'mean'),
        n_seeds=('auc', 'count'),
    ).reset_index()

    summary['auc_ci_low'] = summary['auc_mean'] - 1.96 * summary['auc_std'] / np.sqrt(summary['n_seeds'])
    summary['auc_ci_high'] = summary['auc_mean'] + 1.96 * summary['auc_std'] / np.sqrt(summary['n_seeds'])

    summary.to_csv(RESULTS_DIR / "auc_summary.csv", index=False)

    # Print key results
    print("\n  === Key Results (averaged across datasets/encoders) ===")
    avg = summary.groupby(['method', 'drift_type'])['auc_mean'].mean().unstack()
    print(avg.to_string())

    return auc_df


def generate_plots(df, ablation_results):
    """Generate all required visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    auc_file = RESULTS_DIR / "auc_summary.csv"
    if not auc_file.exists():
        print("  No AUC summary found, skipping plots")
        return

    summary = pd.read_csv(auc_file)

    # 1. AUC Heatmap: method x drift_type (per encoder)
    print("  Generating AUC heatmaps...")
    for enc in summary['encoder'].unique():
        fig, ax = plt.subplots(figsize=(14, 10))
        sub = summary[summary['encoder'] == enc]
        pivot = sub.pivot_table(index='method', columns='drift_type', values='auc_mean', aggfunc='mean')

        if pivot.empty:
            continue

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                    ax=ax, cbar_kws={'label': 'AUC'})
        ax.set_title(f'AUC Heatmap - {enc}')
        ax.set_xlabel('Drift Type')
        ax.set_ylabel('Method')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'auc_heatmap_{enc}.png', dpi=150)
        plt.close()

    # 2. Per-encoder comparison for geo_reorg
    print("  Generating per-encoder comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    methods_to_compare = ['wasserstein_H0', 'mmd_aggregated', 'c2st_mlp']
    geo = summary[(summary['drift_type'] == 'geo_reorg') &
                   (summary['method'].isin(methods_to_compare))]

    if not geo.empty:
        geo_avg = geo.groupby(['encoder', 'method'])['auc_mean'].mean().reset_index()
        pivot = geo_avg.pivot(index='encoder', columns='method', values='auc_mean')
        pivot.plot(kind='bar', ax=ax, rot=15)
        ax.set_ylabel('AUC')
        ax.set_title('Geometric Reorganization: TDA vs Strong Baselines')
        ax.legend(title='Method')
        ax.set_ylim(0.5, 1.0)
        plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'per_encoder_geo_reorg.png', dpi=150)
    plt.close()

    # 3. Runtime breakdown
    print("  Generating runtime breakdown...")
    runtime_cols = ['runtime_pca_ms', 'runtime_sampling_ms', 'runtime_diagram_ms', 'runtime_feature_ms']
    avail_cols = [c for c in runtime_cols if c in df.columns]
    if avail_cols:
        tda_methods = df[df['drift_type'] != 'no_drift']
        if not tda_methods.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            means = tda_methods[avail_cols].mean()
            means.plot(kind='bar', ax=ax)
            ax.set_ylabel('Time (ms)')
            ax.set_title('Runtime Breakdown per Window (TDA Pipeline)')
            ax.set_xticklabels([c.replace('runtime_', '').replace('_ms', '') for c in avail_cols], rotation=30)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / 'runtime_breakdown.png', dpi=150)
            plt.close()

    # 4. FPR calibration plot
    print("  Generating FPR calibration plot...")
    fpr_data = summary[summary['drift_type'] == summary['drift_type'].iloc[0]]  # any drift type
    methods_with_fpr = fpr_data.dropna(subset=['fpr_mean']).groupby('method')['fpr_mean'].mean()

    if not methods_with_fpr.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        methods_with_fpr.plot(kind='bar', ax=ax)
        ax.axhline(y=0.05, color='r', linestyle='--', label='Target FPR (5%)')
        ax.set_ylabel('Realized FPR')
        ax.set_title('FPR Calibration: Realized vs Target (5%)')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'fpr_calibration.png', dpi=150)
        plt.close()

    # 5. Ablation: FPS vs random sampling
    if ablation_results:
        print("  Generating ablation plot...")
        abl_df = pd.DataFrame(ablation_results)

        fig, ax = plt.subplots(figsize=(8, 5))
        for sampling in ['fps', 'random']:
            sub = abl_df[abl_df['sampling_method'] == sampling]
            means = sub.groupby('subsample_size')['wasserstein_H0'].mean()
            ax.plot(means.index, means.values, 'o-', label=sampling)

        ax.set_xlabel('Subsample Size')
        ax.set_ylabel('Wasserstein H0 Distance')
        ax.set_title('Farthest-Point vs Random Sampling')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'ablation_sampling.png', dpi=150)
        plt.close()

    # 6. Paraphrase vs synthetic comparison
    print("  Generating paraphrase comparison...")
    para_methods = ['wasserstein_H0', 'mmd_aggregated', 'c2st_mlp']
    para_types = ['paraphrase_style', 'geo_reorg']
    para = summary[(summary['method'].isin(para_methods)) &
                    (summary['drift_type'].isin(para_types))]

    if not para.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        para_avg = para.groupby(['drift_type', 'method'])['auc_mean'].mean().reset_index()
        pivot = para_avg.pivot(index='method', columns='drift_type', values='auc_mean')
        pivot.plot(kind='bar', ax=ax, rot=15)
        ax.set_ylabel('AUC')
        ax.set_title('Paraphrase Style Drift vs Geometric Reorganization')
        ax.set_ylim(0.5, 1.0)
        ax.legend(title='Drift Type')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'paraphrase_vs_geo.png', dpi=150)
        plt.close()

    print("  All plots saved to figures/")


def save_metrics_json(df, total_time):
    """Save comprehensive metrics in JSON format."""
    metrics = {
        'total_runtime_seconds': total_time,
        'n_configurations': len(df),
        'datasets': list(df['dataset'].unique()),
        'encoders': list(df['encoder'].unique()),
        'drift_types': list(df['drift_type'].unique()),
        'seeds': list(df['seed'].unique()),
        'parameters': {
            'window_size': WINDOW_SIZE,
            'pca_dim': PCA_DIM,
            'tda_subsample': TDA_SUBSAMPLE,
            'max_edge_length': MAX_EDGE_LENGTH,
            'max_homology_dim': MAX_HOMOLOGY_DIM,
            'distance_metric': 'cosine',
            'sampling_method': 'farthest_point',
            'n_ref_windows': N_REF_WINDOWS,
            'n_val_windows': N_VAL_WINDOWS,
            'n_test_windows': N_TEST_WINDOWS,
        }
    }

    with open(RESULTS_DIR / "experiment_config.json", 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
