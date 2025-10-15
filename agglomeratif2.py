import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

# =====================
# 0. Chargement des données
# =====================
path = "../artificial/"
dataframe, meta = arff.loadarff(path + 'hepta.arff')

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
        # ⚠️ 'ward' ne fonctionne que si la distance = euclidienne
        if link == 'ward':
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=link)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=link, affinity='euclidean')
        
        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels)
        metrics.update({
            "n_clusters": n_clusters,
            "linkage": link
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
best = df_results.loc[df_results["Silhouette"].idxmax()]
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
# Effet du nombre de clusters
plt.figure(figsize=(6,4))
for link in linkage_methods:
    subset = df_results[df_results["linkage"] == link]
    plt.plot(subset["n_clusters"], subset["Silhouette"], 'o-', label=f"linkage={link}")
plt.title("Impact de n_clusters et linkage sur le score Silhouette")
plt.xlabel("n_clusters")
plt.ylabel("Silhouette")
plt.legend()
plt.show()

# =====================
# 8. Dendrogramme pour le meilleur modèle
# =====================
Z = linkage(X, method=best["linkage"])
plt.figure(figsize=(10,5))
dendrogram(Z, truncate_mode="level", p=5)
plt.title(f"Dendrogramme (linkage={best['linkage']})")
plt.show()
