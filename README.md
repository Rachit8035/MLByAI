# MLByAI

A beginner-friendly collection of standalone, well-commented machine-learning programs. Each example uses a small dataset bundled with `scikit-learn` or generated in code, so no separate download is required.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run a program from the repository root, for example:

```powershell
python supervised\logistic_regression_iris.py
```

## Contents

| Area | Programs |
| --- | --- |
| Supervised learning | Linear regression, logistic regression, K-nearest neighbours, decision tree, random forest, support-vector machine |
| Unsupervised learning | K-means clustering, hierarchical clustering, DBSCAN |
| Dimensionality reduction | PCA |
| Neural networks | Multi-layer perceptron classifier |
| Model selection | Cross-validation and grid search |

Every script prints a useful evaluation metric. The examples are deliberately small and educational; for real projects, split data carefully, validate on unseen data, and persist a fitted preprocessing pipeline alongside the model.
