"""Find dense groups and mark isolated points as noise with DBSCAN."""
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

X, y = make_moons(n_samples=300, noise=0.06, random_state=42)
# Label -1 denotes noise; DBSCAN can discover non-spherical cluster shapes.
labels = DBSCAN(eps=0.20, min_samples=5).fit_predict(X)
print(f"Adjusted Rand Index: {adjusted_rand_score(y, labels):.3f}")
