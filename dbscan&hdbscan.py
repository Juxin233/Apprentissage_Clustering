import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan

# =========================================================
# 1. Load dataset
# =========================================================
path = "../artificial/"
dataframe, meta = arff.loadarff(path + 'hepta.arff')

x = dataframe['x']
y = dataframe['y']
X = np.array([[x[i], y[i]] for i in range(len(x))])

# =========================================================
# 2. Utility functions
# =========================================================
def plot_clusters(X, labels, title=""):
    plt.figure(figsize=(6,5))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap="viridis", s=30, alpha=0.8)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def evaluate_clustering(X, labels):
    """Return clustering metrics for valid partitions."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters <= 1:
        return {"Silhouette": np.nan, "CH": np.nan, "DB": np.nan}
    return {
        "Silhouette": silhouette_score(X, labels),
        "CH": calinski_harabasz_score(X, labels),
        "DB": davies_bouldin_score(X, labels)
    }

print("Objectif 1 : Identification des hyperparamètres importants ✔")
print("- DBSCAN: eps, min_samples")
print("- HDBSCAN: min_cluster_size, min_samples\n")

# =========================================================
# 3. DBSCAN – Grid search on eps & min_samples
# =========================================================
eps_values = np.linspace(0.1, 1.0, 10)
min_samples_values = [3, 5, 10]

db_results = []

for eps in eps_values:
    for ms in min_samples_values:
        model = DBSCAN(eps=eps, min_samples=ms)
        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels)
        metrics.update({"eps": eps, "min_samples": ms, "n_clusters": len(set(labels)) - (1 if -1 in labels else 0)})
        db_results.append(metrics)

df_db = pd.DataFrame(db_results)
print("Résumé DBSCAN:\n", df_db.head())

# Plot metric evolution for DBSCAN
plt.figure(figsize=(8,5))
for metric in ["Silhouette", "CH", "DB"]:
    plt.plot(df_db["eps"], df_db.groupby("eps")[metric].mean(), label=metric)
plt.title("Impact de eps sur les scores (DBSCAN)")
plt.xlabel("eps")
plt.ylabel("Score")
plt.legend()
plt.show()

# Example of best DBSCAN parameters (highest Silhouette)
best_db = df_db.loc[df_db["Silhouette"].idxmax()]
print("\n✅ Meilleur DBSCAN:", best_db)
best_model = DBSCAN(eps=best_db["eps"], min_samples=int(best_db["min_samples"]))
labels = best_model.fit_predict(X)
plot_clusters(X, labels, f"Meilleur DBSCAN (eps={best_db['eps']:.2f}, min_samples={int(best_db['min_samples'])})")

# =========================================================
# 4. HDBSCAN – Grid search on min_cluster_size & min_samples
# =========================================================
min_cluster_sizes = [3, 5, 10, 20, 30]
min_samples_values = [3, 5, 10]

hdb_results = []

for mcs in min_cluster_sizes:
    for ms in min_samples_values:
        model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels)
        metrics.update({"min_cluster_size": mcs, "min_samples": ms, "n_clusters": len(set(labels)) - (1 if -1 in labels else 0)})
        hdb_results.append(metrics)

df_hdb = pd.DataFrame(hdb_results)
print("\nRésumé HDBSCAN:\n", df_hdb.head())

# Plot metric evolution for HDBSCAN
plt.figure(figsize=(8,5))
for metric in ["Silhouette", "CH", "DB"]:
    plt.plot(df_hdb["min_cluster_size"], df_hdb.groupby("min_cluster_size")[metric].mean(), label=metric)
plt.title("Impact de min_cluster_size sur les scores (HDBSCAN)")
plt.xlabel("min_cluster_size")
plt.ylabel("Score")
plt.legend()
plt.show()

# Example of best HDBSCAN parameters (highest Silhouette)
best_hdb = df_hdb.loc[df_hdb["Silhouette"].idxmax()]
print("\n✅ Meilleur HDBSCAN:", best_hdb)
best_model = hdbscan.HDBSCAN(min_cluster_size=int(best_hdb["min_cluster_size"]), min_samples=int(best_hdb["min_samples"]))
labels = best_model.fit_predict(X)
plot_clusters(X, labels, f"Meilleur HDBSCAN (min_cluster_size={int(best_hdb['min_cluster_size'])}, min_samples={int(best_hdb['min_samples'])})")
