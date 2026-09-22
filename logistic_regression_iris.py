"""Classify iris flower species with a scaled logistic-regression pipeline."""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
# Scaling is fitted only on training data inside the pipeline, avoiding data leakage.
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
