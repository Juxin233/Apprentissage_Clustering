import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# =====================
# 0. Chargement des données
# =====================
path = "../artificial/"
dataframe, meta = arff.loadarff(path + 'mopsi-joensuu.arff')

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
n_clusters_list = [2, 3, 4, 5, 6]
init_methods = ['k-means++', 'random']
n_init_list = [5, 10, 20]

results = []

# =====================
# 4. Boucle d’expérimentation
# =====================
for n_clusters in n_clusters_list:
    for init in init_methods:
        for n_init in n_init_list:
            model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=0)
            labels = model.fit_predict(X)
            metrics = evaluate_clustering(X, labels)
            metrics.update({
                "n_clusters": n_clusters,
                "init": init,
                "n_init": n_init,
                "inertia": model.inertia_,
                "iterations": model.n_iter_
            })
            results.append(metrics)

# =====================
# 5. Tableau récapitulatif
# =====================
df_results = pd.DataFrame(results)
print("\nRésumé des expériences :")
print(df_results.sort_values(by="Silhouette", ascending=False).head(10))

# =====================
# 6. Meilleur modèle (selon Silhouette)
# =====================
best = df_results.loc[df_results["Silhouette"].idxmax()]
print("\n✅ Meilleur modèle trouvé :")
print(best)

best_model = KMeans(
    n_clusters=int(best["n_clusters"]),
    init=best["init"],
    n_init=int(best["n_init"]),
    random_state=0
)
best_labels = best_model.fit_predict(X)
plot_clusters(X, best_labels, f"Meilleur K-Means: k={best['n_clusters']}, init={best['init']}, n_init={best['n_init']}")

# =====================
# 7. Visualisation de l’effet des paramètres
# =====================
# Impact du nombre de clusters
plt.figure(figsize=(6,4))
plt.plot(df_results["n_clusters"], df_results["Silhouette"], 'o-')
plt.title("Impact de n_clusters sur le score Silhouette")
plt.xlabel("n_clusters")
plt.ylabel("Silhouette")
plt.show()

# Impact du n_init (stabilité)
plt.figure(figsize=(6,4))
for init in init_methods:
    subset = df_results[df_results["init"] == init]
    plt.plot(subset["n_init"], subset["Silhouette"], 'o-', label=f"init={init}")
plt.title("Impact de n_init et init sur la stabilité du clustering")
plt.xlabel("n_init")
plt.ylabel("Silhouette")
plt.legend()
plt.show()
