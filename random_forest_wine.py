"""Use an ensemble of decision trees to classify wine varieties."""
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
# Each tree sees a random sample/features; averaging makes predictions more robust.
model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
