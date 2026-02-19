from Q1 import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

x_train,x_val,x_test,y_train,y_val,y_test,y_train_full_encoded,y_test_encoded = preprocessing()

#Flatten the images
X_train_flat = x_train.reshape(x_train.shape[0], -1)
X_test_flat = x_test.reshape(x_test.shape[0], -1)

# Convert the labels to integers
if y_train.ndim > 1:
    y_train_integers = np.argmax(y_train, axis=1)
else:
    y_train_integers = y_train

# Check y_test separately
if y_test.ndim > 1:
    y_test_integers = np.argmax(y_test, axis=1)
else:
    y_test_integers = y_test

#Train the model
print("Training Logistic Regression... (this may take 30-60 seconds)")
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
log_reg.fit(X_train_flat, y_train_integers)
print("Training Complete.")

y_pred_test = log_reg.predict(X_test_flat)
print("--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test_integers, y_pred_test):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test_integers, y_pred_test))

# Define the parameter grid (checking 3 different values for C)
param_grid = {'C': [0.1, 1, 10]}

#new model 
grid_model = LogisticRegression(solver='lbfgs',max_iter=500)\

print(f"Starting Grid Search on samples...")

grid_search = GridSearchCV(grid_model, param_grid, cv=3, verbose=2)
grid_search.fit(X_train_flat, y_train_integers)

print(f"Best C value found: {grid_search.best_params_['C']}")
print(f"Best Validation Score: {grid_search.best_score_:.4f}")


# 1. Reduce data to 2 Dimensions using PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_flat) 

# 2. Train a simplified Logistic Regression on the 2D data
lr_pca = LogisticRegression(solver='lbfgs')
lr_pca.fit(X_train_pca, y_train_integers)


# 3. Create a meshgrid to plot the boundaries
h = .5  # Step size in the mesh
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 4. Predict classifications for every point on the grid
Z = lr_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 5. Plot
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8) # The background regions
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_integers, edgecolors='k', cmap=plt.cm.Paired, s=20)
plt.title('Logistic Regression Decision Boundaries (PCA reduced to 2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
