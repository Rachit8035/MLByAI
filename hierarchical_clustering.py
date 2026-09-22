"""Group iris observations with agglomerative (bottom-up) clustering."""
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
labels = AgglomerativeClustering(n_clusters=3).fit_predict(X)
print(f"Adjusted Rand Index: {adjusted_rand_score(y, labels):.3f}")
