import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif,RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Load the dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target

# 2. Split into Features (X) and Target (y)
X = df.drop('species', axis=1)
y = df['species']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nFirst 5 rows:\n", df.head())

#Pairplot: Visualize relationships between features
sns.pairplot(df, hue='species', palette='viridis')
plt.title('Pairplot of Iris Dataset')
plt.show()

#Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

#1. Univariate Feature Selection
# Select the top 2 features
selector_univariate = SelectKBest(score_func=f_classif, k=2)
X_new_uni = selector_univariate.fit_transform(X, y)

# Get the selected feature names
selected_indices_uni = selector_univariate.get_support(indices=True)
selected_features_uni = X.columns[selected_indices_uni]
print("Univariate Selected Features:", list(selected_features_uni))

#2. Random Forest Feature Importance
# Train a Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Plot importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 4))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices])
plt.show()

print("Top 2 Features based on RF:", list(X.columns[indices[:2]]))

#3.Recursive Feature Elimination (RFE)
# We use a linear kernel because RFE requires model coefficients (weights)
svm_rfe = SVC(kernel="linear")
rfe = RFE(estimator=svm_rfe, n_features_to_select=2, step=1)
rfe.fit(X, y)

selected_features_rfe = X.columns[rfe.support_]
print("RFE Selected Features:", list(selected_features_rfe))


#4. Logistic Regression with Feature Selection
# 1. Split the original full data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train on ALL features
model_full = LogisticRegression(max_iter=200)
model_full.fit(X_train, y_train)
acc_full = accuracy_score(y_test, model_full.predict(X_test))

# 3. Train on SELECTED features (using the RFE selection as our choice)
# Subset the data to only the columns selected by RFE
X_train_rfe = X_train[selected_features_rfe]
X_test_rfe = X_test[selected_features_rfe]

model_sel = LogisticRegression(max_iter=200)
model_sel.fit(X_train_rfe, y_train)
acc_sel = accuracy_score(y_test, model_sel.predict(X_test_rfe))

# 4. Compare
print(f"Accuracy with ALL features (4): {acc_full:.4f}")
print(f"Accuracy with SELECTED features (2): {acc_sel:.4f}")

if acc_sel >= acc_full - 0.05:
    print("\nResult: Feature selection successfully reduced complexity without significant loss in accuracy.")
else:
    print("\nResult: Significant information was lost during feature selection.")
