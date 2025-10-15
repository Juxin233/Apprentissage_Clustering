
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

import hdbscan

#Jeux de données
path = "../artificial/"
dataframe, meta = arff.loadarff(path + 'hepta.arff')
x = dataframe['x']#np.array(dataframe['tumor'],dtype=float)
y = dataframe['y']#np.array(dataframe['relapse'],dtype=float)
X=np.array([[x[i],y[i]] for i in range(len(x))])


#Fonctions utiles
def plot_clusters(X, labels, title=""):
    plt.figure(figsize=(6,5))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap="viridis", s=30, alpha=0.8)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def evaluate_clustering(X, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters <= 1:
        return {"Silhouette": None, "CH": None, "DB": None}
    return {
        "Silhouette": silhouette_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels)
    }

#  Ici on identifie les paramètres principaux pour chaque méthode
# - KMeans: n_clusters, init, n_init
# - Agglomératif: n_clusters, linkage
# - DBSCAN: eps, min_samples
# - HDBSCAN: min_cluster_size, min_samples

print("Objectif 1 : Identification des hyperparamètres importants ✔")



# Pour DBSCAN et HDBSCAN → on teste plusieurs eps / min_cluster_size
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)
plot_clusters(X, labels, "DBSCAN (eps=0.3, min_samples=5)")
print("DBSCAN:", evaluate_clustering(X, labels))

hdb = hdbscan.HDBSCAN(min_cluster_size=5)
labels = hdb.fit_predict(X)
plot_clusters(X, labels, "HDBSCAN (min_cluster_size=5)")
print("HDBSCAN:", evaluate_clustering(X, labels))

dbscan = DBSCAN(eps=0.2, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)
plot_clusters(X, labels_dbscan, "DBSCAN sur Moons (réussite → clusters en croissant)")
