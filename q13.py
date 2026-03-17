import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression


# ==============================
# 1. Load Dataset
# ==============================

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ==============================
# 2. Feature Scaling (IMPORTANT)
# ==============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==============================
# 3. Train ALL sklearn Classifiers
# ==============================

results = []

for name, Classifier in all_estimators(type_filter='classifier'):
    try:
        # Special handling for models needing more iterations
        if name == "LogisticRegression":
            model = Classifier(max_iter=1000)
        elif name == "MLPClassifier":
            model = Classifier(max_iter=1000)
        else:
            model = Classifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "F1": f1_score(y_test, y_pred, average='macro', zero_division=0)
        })

    except Exception:
        continue


# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)

print("\n==============================")
print("Top 10 Classifiers")
print("==============================")
print(results_df.head(10))


# ==============================
# 4. Select Top 4 Models
# ==============================

top4_names = results_df.head(4)["Model"].values
print("\nTop 4 Models:", top4_names)

top4_models = []

for name, Classifier in all_estimators(type_filter='classifier'):
    if name in top4_names:
        if name == "LogisticRegression":
            model = Classifier(max_iter=1000)
        elif name == "MLPClassifier":
            model = Classifier(max_iter=1000)
        else:
            model = Classifier()

        model.fit(X_train, y_train)
        top4_models.append((name, model))


# ==============================
# 5. Ensemble Methods
# ==============================

# ---- BAGGING ----
best_model = top4_models[0][1]

bagging = BaggingClassifier(
    estimator=best_model,
    n_estimators=50,
    random_state=42
)

bagging.fit(X_train, y_train)
y_pred_bag = bagging.predict(X_test)

print("\nBagging Accuracy:", accuracy_score(y_test, y_pred_bag))


# ---- BOOSTING (AdaBoost) ----
boosting = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
)

boosting.fit(X_train, y_train)
y_pred_boost = boosting.predict(X_test)

print("Boosting Accuracy:", accuracy_score(y_test, y_pred_boost))


# ---- STACKING ----
stacking = StackingClassifier(
    estimators=top4_models,
    final_estimator=LogisticRegression(max_iter=1000)
)

stacking.fit(X_train, y_train)
y_pred_stack = stacking.predict(X_test)

print("Stacking Accuracy:", accuracy_score(y_test, y_pred_stack))


# ==============================
# 6. Final Comparison
# ==============================

print("\n==============================")
print("Ensemble Comparison")
print("==============================")
print("Bagging  :", accuracy_score(y_test, y_pred_bag))
print("Boosting :", accuracy_score(y_test, y_pred_boost))
print("Stacking :", accuracy_score(y_test, y_pred_stack))
