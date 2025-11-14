# ==========================================================
# EEG Emotion Recognition - Random Forest & XGBoost
# Dataset: emotions.csv
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------
data = pd.read_csv('emotions/emotions.csv')
print("Dataset shape:", data.shape)
print("Columns:", list(data.columns[:10]), "...\n")

# ----------------------------------------------------------
# 2. Label Encoding
# ----------------------------------------------------------
label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
data['label'] = data['label'].replace(label_mapping)
print("Label distribution:\n", data['label'].value_counts(), "\n")

# ----------------------------------------------------------
# 3. Select Features
# ----------------------------------------------------------
feature_columns = [ ... (same list you provided) ... ]

data = data[feature_columns + ['label']]
print(f"Selected {len(feature_columns)} features + label.\n")

# ----------------------------------------------------------
# 4. Train/Test Split
# ----------------------------------------------------------
X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y
)

print(f"Training samples: {X_train.shape[0]} | Testing: {X_test.shape[0]}\n")

# ----------------------------------------------------------
# 5. Random Forest
# ----------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    random_state=123,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred) * 100

print(f"Random Forest Accuracy: {rf_acc:.2f}%\n")
print("RF Classification Report:\n")
print(classification_report(y_test, rf_pred, target_names=label_mapping.keys()))

# ----------------------------------------------------------
# 6. XGBoost
# ----------------------------------------------------------
xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=123,
    eval_metric='mlogloss'
)

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred) * 100

print(f"\nXGBoost Accuracy: {xgb_acc:.2f}%\n")
print("XGBoost Classification Report:\n")
print(classification_report(y_test, xgb_pred, target_names=label_mapping.keys()))

# ----------------------------------------------------------
# 7. Confusion Matrices
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='g', cmap='Blues', ax=axes[0])
axes[0].set_title(f'RF Confusion Matrix ({rf_acc:.2f}%)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(label_mapping.keys())
axes[0].set_yticklabels(label_mapping.keys())

cm_xgb = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm_xgb, annot=True, fmt='g', cmap='Greens', ax=axes[1])
axes[1].set_title(f'XGB Confusion Matrix ({xgb_acc:.2f}%)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_xticklabels(label_mapping.keys())
axes[1].set_yticklabels(label_mapping.keys())

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 8. Feature Importance
# ----------------------------------------------------------
# Random Forest
importances_rf = rf.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1][:15]

plt.figure(figsize=(10, 5))
plt.title("Top 15 Features - Random Forest")
plt.bar(range(len(indices_rf)), importances_rf[indices_rf])
plt.xticks(range(len(indices_rf)), [X.columns[i] for i in indices_rf], rotation=90)
plt.tight_layout()
plt.show()

# XGBoost
importances_xgb = xgb.feature_importances_
indices_xgb = np.argsort(importances_xgb)[::-1][:15]

plt.figure(figsize=(10, 5))
plt.title("Top 15 Features - XGBoost")
plt.bar(range(len(indices_xgb)), importances_xgb[indices_xgb], color='orange')
plt.xticks(range(len(indices_xgb)), [X.columns[i] for i in indices_xgb], rotation=90)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 9. Summary
# ----------------------------------------------------------
print("\n============================")
print("Model Performance Comparison")
print("============================")
print(f"Random Forest Accuracy: {rf_acc:.2f}%")
print(f"XGBoost Accuracy:      {xgb_acc:.2f}%")

best_model = "Random Forest" if rf_acc > xgb_acc else "XGBoost"
print(f"\nBest Model: {best_model}")


