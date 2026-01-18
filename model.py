# === Import libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# Set up plot style
sns.set(style="whitegrid", palette="pastel", font_scale=1.1)

# === Load dataset ===
data = pd.read_csv("heart_cleveland_upload.csv")
data.head()

# Basic info
print("Dataset shape:", data.shape)
print("\nMissing values per column:\n", data.isnull().sum())

# Quick summary statistics
data.describe()

# Check column names
data.columns
# Rename the target column to 'target' for consistency
data = data.rename(columns={'condition': 'target'})

# Target variable distribution
sns.countplot(data=data, x='target', palette='Set2')
plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Example: Feature importance exploration could be done later using RandomForest
print("Number of features:", X_train.shape[1])

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:,1]

print("🔹 Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

# Plot ROC Curves
plt.figure(figsize=(6,5))
RocCurveDisplay.from_estimator(log_reg, X_test_scaled, y_test, name="Logistic Regression")
plt.title("ROC Curves Comparison")
plt.show()


import joblib
joblib.dump(log_reg, 'model.joblib')
