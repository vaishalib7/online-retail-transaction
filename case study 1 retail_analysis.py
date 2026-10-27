"""Online Retail Transaction Clustering Using Machine Learning Techniques.

Input: a CSV where each row is one basket and each non-empty cell is a product slot.
Place the CSV in this folder or in data/; set DATA_FILE below if needed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

ROOT = Path(__file__).resolve().parent
DATA_DIR, OUT_DIR = ROOT / "data", ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
DATA_FILE = None  # Example: "data/transactions.csv"

def locate_csv():
    if DATA_FILE:
        p = ROOT / DATA_FILE
        if p.exists(): return p
        raise FileNotFoundError(f"Configured CSV not found: {p}")
    candidates = list(DATA_DIR.glob("*.csv")) + list(ROOT.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError("Place your transaction CSV in this folder or data/; set DATA_FILE.")
    return candidates[0]

path = locate_csv()
raw = pd.read_csv(path, header=None, dtype=str)
print("Input file:", path.name)
print("Raw dimensions:", raw.shape)
print("First five rows:\n", raw.head())
print("Data types:\n", raw.dtypes)

# Each row is a basket; blanks are expected and are not imputed.
baskets = []
for _, row in raw.iterrows():
    items = []
    for value in row.dropna().astype(str):
        value = value.strip().lower()
        if value and value not in {"nan", "none"}:
            items.append(value)
    # Remove repeated products within each transaction, preserving order.
    baskets.append(list(dict.fromkeys(items)))

basket_sizes = pd.Series([len(b) for b in baskets], name="distinct_products")
print("\nBasket-size descriptive statistics:\n", basket_sizes.describe())
print("Empty baskets:", int((basket_sizes == 0).sum()))
nonempty = [(i, b) for i, b in enumerate(baskets) if b]
if len(nonempty) < 3:
    raise ValueError("At least three non-empty transaction rows are required for clustering.")

product_counts = {}
for _, basket in nonempty:
    for product in basket:
        product_counts[product] = product_counts.get(product, 0) + 1
freq = pd.Series(product_counts, name="transaction_frequency").sort_values(ascending=False)
print("\nTop 20 products by transaction frequency:\n", freq.head(20))

plt.figure(figsize=(10, 6))
freq.head(20).sort_values().plot(kind="barh")
plt.title("Top 20 Products by Transaction Frequency")
plt.xlabel("Number of baskets containing product"); plt.ylabel("Product")
plt.tight_layout(); plt.savefig(OUT_DIR / "top_products.png", dpi=160); plt.show()

plt.figure(figsize=(8, 5))
basket_sizes.plot(kind="hist", bins=25)
plt.title("Distribution of Transaction Basket Sizes")
plt.xlabel("Distinct products in basket"); plt.ylabel("Number of transactions")
plt.tight_layout(); plt.savefig(OUT_DIR / "basket_size_distribution.png", dpi=160); plt.show()

# Binary transaction-product matrix
mlb = MultiLabelBinarizer()
X_all = mlb.fit_transform([b for _, b in nonempty])
matrix = pd.DataFrame(X_all, columns=mlb.classes_)
print("\nBinary transaction-product matrix shape:", matrix.shape)
print("Duplicate basket rows (reported, not removed):", int(matrix.duplicated().sum()))

# Sampling makes PCA and spectral clustering more practical for large files.
rng = np.random.RandomState(42)
max_rows = min(2000, len(matrix))
sample_idx = np.sort(rng.choice(len(matrix), size=max_rows, replace=False)) if len(matrix) > max_rows else np.arange(len(matrix))
X = matrix.iloc[sample_idx].to_numpy(dtype=float)
# Remove product columns that have no variance in the sample.
keep = X.std(axis=0) > 0
X = X[:, keep]
if X.shape[1] < 2:
    raise ValueError("Need at least two varying products in the sampled baskets.")
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("PCA explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance in first two components:", pca.explained_variance_ratio_.sum())

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=12, alpha=.55)
plt.title("Transaction Baskets in Two-Dimensional PCA Space")
plt.xlabel("Principal component 1"); plt.ylabel("Principal component 2")
plt.tight_layout(); plt.savefig(OUT_DIR / "pca_transactions.png", dpi=160); plt.show()

def valid_scores(labels, data, ignore_noise=False):
    labels = np.asarray(labels)
    mask = labels != -1 if ignore_noise else np.ones(len(labels), dtype=bool)
    labs = labels[mask]; xx = data[mask]
    n = len(set(labs))
    if n < 2 or n >= len(labs):
        return np.nan, np.nan
    return (silhouette_score(xx, labs, sample_size=min(1000, len(labs)), random_state=42),
            davies_bouldin_score(xx, labs))

results, label_sets = [], {}
# K-means: select k=2..6 with the highest silhouette score.
k_candidates = []
for k in range(2, min(6, len(X)-1) + 1):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
    sil, dbi = valid_scores(labels, X_scaled)
    k_candidates.append((sil, k, dbi, labels))
valid_k = [r for r in k_candidates if np.isfinite(r[0])]
if not valid_k:
    raise ValueError("Could not compute a valid K-means silhouette score.")
best_sil, best_k, best_dbi, km_labels = max(valid_k, key=lambda r: r[0])
label_sets["kmeans"] = km_labels
results.append({"Algorithm": f"K-means (k={best_k})", "Clusters": len(set(km_labels)),
                "Noise points": 0, "Silhouette": best_sil, "Davies-Bouldin": best_dbi})

# Hierarchical and spectral clustering use the same chosen k for comparison.
hier_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X_scaled)
sil, dbi = valid_scores(hier_labels, X_scaled)
label_sets["hierarchical"] = hier_labels
results.append({"Algorithm": f"Hierarchical (k={best_k})", "Clusters": len(set(hier_labels)),
                "Noise points": 0, "Silhouette": sil, "Davies-Bouldin": dbi})

db_labels = DBSCAN(eps=3.0, min_samples=5).fit_predict(X_scaled)
sil, dbi = valid_scores(db_labels, X_scaled, ignore_noise=True)
label_sets["dbscan"] = db_labels
results.append({"Algorithm": "DBSCAN (eps=3, min_samples=5)", "Clusters": len(set(db_labels)-({-1} if -1 in db_labels else set())),
                "Noise points": int((db_labels == -1).sum()), "Silhouette": sil, "Davies-Bouldin": dbi})

spectral_labels = SpectralClustering(n_clusters=best_k, affinity="nearest_neighbors",
                                     n_neighbors=min(10, len(X)-1), random_state=42).fit_predict(X_scaled)
sil, dbi = valid_scores(spectral_labels, X_scaled)
label_sets["spectral"] = spectral_labels
results.append({"Algorithm": f"Spectral (k={best_k})", "Clusters": len(set(spectral_labels)),
                "Noise points": 0, "Silhouette": sil, "Davies-Bouldin": dbi})

scores = pd.DataFrame(results)
print("\nClustering evaluation:\n", scores.to_string(index=False))
scores.to_csv(OUT_DIR / "clustering_metrics.csv", index=False)
print("\nK-means cluster sizes:\n", pd.Series(km_labels).value_counts().sort_index())

for name, labels in label_sets.items():
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=12, alpha=.65, cmap="tab10")
    plt.title(f"PCA Projection with {name.title()} Cluster Labels")
    plt.xlabel("Principal component 1"); plt.ylabel("Principal component 2")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"clusters_{name}.png", dpi=160); plt.show()

# Product profiles for the K-means solution, based on the same sampled rows/columns.
sampled_matrix = matrix.iloc[sample_idx].loc[:, matrix.columns[keep]]
sampled_matrix = sampled_matrix.reset_index(drop=True)
sampled_matrix["cluster"] = km_labels
profile = sampled_matrix.groupby("cluster").mean().T
profile.to_csv(OUT_DIR / "kmeans_cluster_product_profiles.csv")
print("\nSaved charts and tables in:", OUT_DIR)
print("Interpretation: silhouette closer to 1 is generally better; lower Davies-Bouldin is generally better.")
print("DBSCAN noise points are labeled -1 and excluded from its internal cluster metrics.")
print("Basket frequency counts transactions containing a product, not units sold or revenue.")
