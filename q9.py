import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=1000)

def evaluate_model(X_transformed, y):
    scores = cross_val_score(clf, X_transformed, y, cv=cv)
    return scores.mean()

results = {}

# -------------------------
# CASE 1: 2 FEATURES
# -------------------------

# PCA
pca_2 = PCA(n_components=2)
results["PCA_2"] = evaluate_model(pca_2.fit_transform(X_scaled), y)

# LDA (max 2 components)
lda_2 = LinearDiscriminantAnalysis(n_components=2)
results["LDA_2"] = evaluate_model(lda_2.fit_transform(X_scaled, y), y)

# SVD
svd_2 = TruncatedSVD(n_components=2)
results["SVD_2"] = evaluate_model(svd_2.fit_transform(X_scaled), y)

# t-SNE
tsne_2 = TSNE(n_components=2, random_state=42)
results["TSNE_2"] = evaluate_model(tsne_2.fit_transform(X_scaled), y)

# -------------------------
# CASE 2: 3 FEATURES
# -------------------------

# PCA
pca_3 = PCA(n_components=3)
results["PCA_3"] = evaluate_model(pca_3.fit_transform(X_scaled), y)

# SVD
svd_3 = TruncatedSVD(n_components=3)
results["SVD_3"] = evaluate_model(svd_3.fit_transform(X_scaled), y)

# t-SNE
tsne_3 = TSNE(n_components=3, random_state=42)
results["TSNE_3"] = evaluate_model(tsne_3.fit_transform(X_scaled), y)

# LDA cannot have 3 components (max = 2 for iris)

print("Cross Validation Accuracy Results:\n")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
