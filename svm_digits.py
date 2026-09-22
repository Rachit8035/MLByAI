"""Recognise handwritten digits with a support-vector machine."""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, gamma="scale")).fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
