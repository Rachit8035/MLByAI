"""Classify small handwritten digit images using a feed-forward neural network."""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
# One hidden layer learns nonlinear combinations of the input pixels.
model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=42)).fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
