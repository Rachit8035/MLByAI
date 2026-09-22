"""Train an interpretable decision tree on the iris dataset."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
# Limiting depth makes the learned rules easier to inspect and reduces overfitting.
model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
