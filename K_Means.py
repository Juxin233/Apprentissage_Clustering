import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

path = "../artificial/"
dataframe, meta = arff.loadarff(path + 'mopsi-joensuu.arff')
# print(dataframe.dtype)
# print(dataframe.dtype.names)
x = dataframe['x']#np.array(dataframe['tumor'],dtype=float)
y = dataframe['y']#np.array(dataframe['relapse'],dtype=float)
X=np.array([[x[i],y[i]] for i in range(len(x))])
# print(X[:,0])
# print(X[:,1])

# =====================
# 1. Fonction d'affichage
# =====================
def plot_clusters(X, labels, title=""):
    plt.figure(figsize=(6,5))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap="viridis", s=30, alpha=0.8)
    plt.title(title)
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.show()

# =====================
# 4. Évaluation
# =====================
def evaluate_clustering(X, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters <= 1:
        return {"Silhouette": None, "CH": None, "DB": None}
    return {
        "Silhouette": silhouette_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels)
    }

# =====================
# 5. Modèles
# =====================
model = KMeans(n_clusters=3, random_state=0)

# =====================
# 6. Exécution des méthodes
# =====================
labels = model.fit_predict(X)
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_
print(f"iteration :  {iteration}")
print(f"inertie   :  {inertie}")
# print("centroids : " + centroids)


# =====================
# 7. Résultats comparatifs
# =====================
result = evaluate_clustering(X, labels)
print(result)
plot_clusters(X, labels, "Kmeans résultat")