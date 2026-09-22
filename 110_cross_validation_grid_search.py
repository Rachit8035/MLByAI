"""Select a regularisation value with cross-validation and GridSearchCV."""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
# The prefix identifies the final pipeline step; CV chooses C without seeing test data.
search = GridSearchCV(pipeline, {"logisticregression__C": [0.01, 0.1, 1, 10]}, cv=5).fit(X_train, y_train)
print("Best C:", search.best_params_["logisticregression__C"])
print(f"Test accuracy: {search.score(X_test, y_test):.3f}")
