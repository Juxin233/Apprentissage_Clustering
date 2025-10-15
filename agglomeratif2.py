import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from scipy.io import arff

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

# =====================
# 0. Chargement des données
# =====================
path = "../artificial/"
dataframe, meta = arff.loadarff(path + 'aggregation.arff')

x = dataframe['x']
y = dataframe['y']
X = np.array([[x[i], y[i]] for i in range(len(x))])

# =====================
# 1. Fonction d'affichage
# =====================
def plot_clusters(X, labels, title=""):
    plt.figure(figsize=(6,5))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap="viridis", s=30, alpha=0.8)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# =====================
# 2. Évaluation
# =====================
def evaluate_clustering(X, labels):
    n_clusters = len(set(labels))
    if n_clusters <= 1:
        return {"Silhouette": None, "CH": None, "DB": None}
    return {
        "Silhouette": silhouette_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels)
    }

# =====================
# 3. Grille d’hyperparamètres
# =====================
n_clusters_list = [2, 3, 4, 5, 6, 7, 8]
linkage_methods = ['ward', 'average', 'complete']

results = []

# =====================
# 4. Boucle d’expérimentation
# =====================
for n_clusters in n_clusters_list:
    for link in linkage_methods:
        start=time.time()
        model=AgglomerativeClustering(n_clusters=n_clusters, linkage=link)
        
        labels = model.fit_predict(X)
        end = time.time()
        runtime = end - start
        inertia = None
        if link == "ward":
            inertia = sum(np.sum((X[labels == k] - np.mean(X[labels == k], axis=0))**2) for k in range(n_clusters))
        metrics = evaluate_clustering(X, labels)
        metrics.update({
            "n_clusters": n_clusters,
            "linkage": link,
            "runtime": runtime,
            "inertia": inertia
        })
        results.append(metrics)

# =====================
# 5. Tableau récapitulatif
# =====================
df_results = pd.DataFrame(results)
print("\nRésumé des expériences :")
print(df_results.sort_values(by="Silhouette", ascending=False).head(10))

# =====================
# 6. Meilleur modèle selon la Silhouette
# =====================
best = df_results.loc[df_results["Calinski-Harabasz"].idxmax()]
print("\n✅ Meilleur modèle trouvé :")
print(best)

best_model = AgglomerativeClustering(
    n_clusters=int(best["n_clusters"]),
    linkage=best["linkage"]
)
best_labels = best_model.fit_predict(X)
plot_clusters(X, best_labels, f"Agglomératif optimal (k={best['n_clusters']}, linkage={best['linkage']})")

# =====================
# 7. Analyse de l’effet des paramètres
# =====================
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Moyenne des scores par n_clusters (tous linkage confondus)
metrics_grouped = df_results.groupby("n_clusters")[["Silhouette","Calinski-Harabasz","Davies-Bouldin"]].mean()
best_k = int(best["n_clusters"])

# Silhouette
axes[0].plot(metrics_grouped.index, metrics_grouped["Silhouette"], marker="o", label="Silhouette")
axes[0].axvline(x=best_k, color="red", linestyle="--", label=f"Best k={best_k}")
axes[0].set_ylabel("Silhouette")
axes[0].legend()
axes[0].grid(True)

# Calinski-Harabasz
axes[1].plot(metrics_grouped.index, metrics_grouped["Calinski-Harabasz"], marker="s", color="green", label="Calinski-Harabasz")
axes[1].axvline(x=best_k, color="red", linestyle="--")
axes[1].set_ylabel("CH score")
axes[1].legend()
axes[1].grid(True)

# Davies-Bouldin
axes[2].plot(metrics_grouped.index, metrics_grouped["Davies-Bouldin"], marker="^", color="purple", label="Davies-Bouldin")
axes[2].axvline(x=best_k, color="red", linestyle="--")
axes[2].set_xlabel("Number of clusters (k)")
axes[2].set_ylabel("DB score")
axes[2].legend()
axes[2].grid(True)

plt.suptitle("Comparison of clustering metrics (Agglomerative)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

#runtime
runtime_grouped = df_results.groupby("n_clusters")["runtime"].mean()
axes[0].plot(runtime_grouped.index, runtime_grouped.values, marker="o", color="orange")
axes[0].set_ylabel("Runtime (s)")
axes[0].set_title("Average runtime vs. number of clusters")
axes[0].grid(True)

#inertia
inertia_grouped = df_results.groupby("n_clusters")["inertia"].mean()
axes[1].plot(inertia_grouped.index, inertia_grouped.values, marker="s", color="blue")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Inertia (WCSS, ward only)")
axes[1].set_title("Inertia vs. number of clusters (Ward linkage)")
axes[1].grid(True)

plt.tight_layout()
plt.show()

