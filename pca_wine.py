"""Reduce 13 wine measurements to two principal components."""
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X, _ = load_wine(return_X_y=True)
# PCA is sensitive to units, therefore standardise each feature first.
X_scaled = StandardScaler().fit_transform(X)
reduced = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
print("Reduced shape:", reduced.shape)
print("First transformed row:", reduced[0].round(3))
