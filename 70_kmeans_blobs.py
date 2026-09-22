"""Discover three synthetic groups using K-means clustering."""
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

X, true_labels = make_blobs(n_samples=300, centers=3, cluster_std=0.70, random_state=42)
# K-means assigns each point to its nearest learned centroid.
labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)
print(f"Adjusted Rand Index: {adjusted_rand_score(true_labels, labels):.3f}")
