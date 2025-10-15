import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan

# =====================
# 0. Chargement des données
# =====================
path = "../artificial/"
dataframe, meta = arff.loadarff(path + 'hepta.arff')

x = dataframe['x']
y = dataframe['y']
X = np.array([[x[i], y[i]] for i in range(len(x))])

# =====================
# 1. Fonctions utilitaires
# =====================
def plot_clusters(X, labels, title=""):
    plt.figure(figsize=(6,5))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap="viridis", s=30, alpha=0.8)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def evaluate_clustering(X, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters <= 1:
        return {"Silhouette": None, "Calinski-Harabasz": None, "Davies-Bouldin": None}
    return {
        "Silhouette": silhouette_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels)
    }

# ============================================================
#                   DBSCAN EXPERIMENTS
# ============================================================

eps_values = np.linspace(0.1, 1.0, 10)
min_samples_values = [3, 5, 10]

results = []

# Boucle d’expérimentation
for eps in eps_values:
    for ms in min_samples_values:
        model = DBSCAN(eps=eps, min_samples=ms)
        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels)
        metrics.update({
            "eps": eps,
            "min_samples": ms,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0)
        })
        results.append(metrics)

# Tableau récapitulatif
df_dbscan = pd.DataFrame(results)
print("\nRésumé des expériences DBSCAN :")
print(df_dbscan.sort_values(by="Silhouette", ascending=False).head(10))

# Meilleur modèle (selon Silhouette)
best_dbscan = df_dbscan.loc[df_dbscan["Silhouette"].idxmax()]
print("\n✅ Meilleur modèle DBSCAN trouvé :")
print(best_dbscan)

best_model = DBSCAN(eps=best_dbscan["eps"], min_samples=int(best_dbscan["min_samples"]))
best_labels = best_model.fit_predict(X)

# --- Plot du meilleur clustering ---
plot_clusters(
    X,
    best_labels,
    f"Best DBSCAN: eps={best_dbscan['eps']:.2f}, min_samples={int(best_dbscan['min_samples'])}\nSilhouette={best_dbscan['Silhouette']:.3f}"
)

# --- Comparaison des métriques (par valeur de eps) ---
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
metrics_grouped = df_dbscan.groupby("eps")[["Silhouette","Calinski-Harabasz","Davies-Bouldin"]].mean()
best_eps = best_dbscan["eps"]

axes[0].plot(metrics_grouped.index, metrics_grouped["Silhouette"], marker="o", label="Silhouette")
axes[0].axvline(x=best_eps, color="red", linestyle="--", label=f"Best eps={best_eps:.2f}")
axes[0].set_ylabel("Silhouette")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(metrics_grouped.index, metrics_grouped["Calinski-Harabasz"], marker="s", color="green", label="Calinski-Harabasz")
axes[1].axvline(x=best_eps, color="red", linestyle="--")
axes[1].set_ylabel("CH score")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(metrics_grouped.index, metrics_grouped["Davies-Bouldin"], marker="^", color="purple", label="Davies-Bouldin")
axes[2].axvline(x=best_eps, color="red", linestyle="--")
axes[2].set_xlabel("eps")
axes[2].set_ylabel("DB score")
axes[2].legend()
axes[2].grid(True)

plt.suptitle("Comparison of clustering metrics for DBSCAN", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# ============================================================
#                   HDBSCAN EXPERIMENTS
# ============================================================

min_cluster_sizes = [3, 5, 10, 20, 30]
min_samples_values = [3, 5, 10]
results = []

# Boucle d’expérimentation
for mcs in min_cluster_sizes:
    for ms in min_samples_values:
        model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels)
        metrics.update({
            "min_cluster_size": mcs,
            "min_samples": ms,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0)
        })
        results.append(metrics)

# Tableau récapitulatif
df_hdbscan = pd.DataFrame(results)
print("\nRésumé des expériences HDBSCAN :")
print(df_hdbscan.sort_values(by="Silhouette", ascending=False).head(10))

# Meilleur modèle (selon Silhouette)
best_hdbscan = df_hdbscan.loc[df_hdbscan["Silhouette"].idxmax()]
print("\n✅ Meilleur modèle HDBSCAN trouvé :")
print(best_hdbscan)

best_model = hdbscan.HDBSCAN(
    min_cluster_size=int(best_hdbscan["min_cluster_size"]),
    min_samples=int(best_hdbscan["min_samples"])
)
best_labels = best_model.fit_predict(X)

# --- Plot du meilleur clustering ---
plot_clusters(
    X,
    best_labels,
    f"Best HDBSCAN: min_cluster_size={int(best_hdbscan['min_cluster_size'])}, min_samples={int(best_hdbscan['min_samples'])}\nSilhouette={best_hdbscan['Silhouette']:.3f}"
)

# --- Comparaison des métriques (par valeur de min_cluster_size) ---
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
metrics_grouped = df_hdbscan.groupby("min_cluster_size")[["Silhouette","Calinski-Harabasz","Davies-Bouldin"]].mean()
best_mcs = best_hdbscan["min_cluster_size"]

axes[0].plot(metrics_grouped.index, metrics_grouped["Silhouette"], marker="o", label="Silhouette")
axes[0].axvline(x=best_mcs, color="red", linestyle="--", label=f"Best min_cluster_size={best_mcs}")
axes[0].set_ylabel("Silhouette")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(metrics_grouped.index, metrics_grouped["Calinski-Harabasz"], marker="s", color="green", label="Calinski-Harabasz")
axes[1].axvline(x=best_mcs, color="red", linestyle="--")
axes[1].set_ylabel("CH score")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(metrics_grouped.index, metrics_grouped["Davies-Bouldin"], marker="^", color="purple", label="Davies-Bouldin")
axes[2].axvline(x=best_mcs, color="red", linestyle="--")
axes[2].set_xlabel("min_cluster_size")
axes[2].set_ylabel("DB score")
axes[2].legend()
axes[2].grid(True)

plt.suptitle("Comparison of clustering metrics for HDBSCAN", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


