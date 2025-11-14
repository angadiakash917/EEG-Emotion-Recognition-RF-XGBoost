# ==========================================================
# EEG Emotion Recognition using Random Forest and XGBoost
# Dataset: EEG Brainwave Dataset (emotions.csv)
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
# 3. Feature Selection (your selected subset)
# ----------------------------------------------------------
feature_columns = [
    '# mean_0_a', 'mean_2_a', 'mean_3_a', 'mean_d_0_a2', 'mean_d_2_a2',
    'mean_d_3_a2', 'mean_d_5_a', 'mean_d_7_a', 'mean_d_8_a', 'mean_d_10_a',
    'mean_d_12_a', 'mean_d_13_a', 'mean_d_15_a', 'mean_d_17_a',
    'mean_d_18_a', 'stddev_0_a', 'stddev_2_a', 'moments_2_a', 'moments_5_a',
    'moments_7_a', 'moments_12_a', 'moments_15_a', 'moments_17_a',
    'max_3_a', 'max_q_3_a', 'max_q_8_a', 'max_q_13_a', 'max_q_18_a',
    'min_0_a', 'min_2_a', 'min_3_a', 'min_q_0_a', 'min_q_2_a', 'min_q_3_a',
    'min_q_5_a', 'min_q_7_a', 'min_q_8_a', 'min_q_10_a', 'min_q_12_a',
    'min_q_13_a', 'min_q_15_a', 'min_q_17_a', 'min_q_18_a', 'covmat_0_a',
    'covmat_1_a', 'covmat_3_a', 'covmat_4_a', 'covmat_5_a', 'covmat_8_a',
    'covmat_9_a', 'covmat_12_a', 'covmat_13_a', 'covmat_15_a',
    'covmat_17_a', 'covmat_20_a', 'covmat_36_a', 'covmat_37_a',
    'covmat_44_a', 'covmat_48_a', 'covmat_56_a', 'covmat_57_a',
    'covmat_60_a', 'covmat_61_a', 'covmat_68_a', 'covmat_69_a',
    'covmat_96_a', 'covmat_97_a', 'covmat_99_a', 'covmat_100_a',
    'covmat_101_a', 'covmat_104_a', 'covmat_105_a', 'covmat_108_a',
    'covmat_112_a', 'covmat_113_a', 'covmat_116_a', 'covmat_117_a',
    'eigen_0_a', 'eigen_1_a', 'eigen_2_a', 'eigen_3_a', 'logm_0_a',
    'logm_8_a', 'logm_9_a', 'correlate_0_a', 'correlate_1_a',
    'correlate_2_a', 'correlate_3_a', 'correlate_4_a', 'correlate_6_a',
    'correlate_8_a', 'correlate_9_a', 'correlate_12_a', 'correlate_13_a',
    'correlate_14_a', 'correlate_15_a', 'correlate_18_a', 'correlate_21_a',
    'correlate_24_a', 'correlate_27_a','correlate_28_a', 'correlate_33_a', 'correlate_36_a', 'correlate_39_a',
    'correlate_42_a', 'correlate_44_a', 'correlate_45_a', 'correlate_47_a',
    'correlate_48_a', 'correlate_51_a', 'correlate_54_a', 'correlate_56_a',
    'correlate_57_a', 'correlate_58_a', 'correlate_60_a', 'correlate_61_a',
    'correlate_63_a', 'correlate_66_a', 'correlate_68_a', 'correlate_69_a',
    'correlate_72_a', 'correlate_74_a', 'fft_0_a', 'fft_1_a', 'fft_4_a',
    'fft_15_a', 'fft_16_a', 'fft_19_a', 'fft_30_a', 'fft_31_a', 'fft_34_a',
    'fft_45_a', 'fft_46_a', 'fft_60_a', 'fft_61_a', 'fft_64_a', 'fft_75_a',
    'fft_90_a', 'fft_105_a', 'fft_106_a', 'fft_109_a', 'fft_120_a',
    'fft_135_a', 'fft_136_a', 'fft_139_a', 'fft_140_a', 'fft_150_a',
    'fft_151_a', 'fft_154_a', 'fft_165_a', 'fft_175_a', 'fft_180_a',
    'fft_181_a', 'fft_184_a', 'fft_195_a', 'fft_196_a', 'fft_199_a',
    'fft_210_a', 'fft_211_a', 'fft_214_a', 'fft_225_a', 'fft_226_a',
    'fft_229_a', 'fft_240_a', 'fft_241_a', 'fft_244_a', 'fft_255_a',
    'fft_256_a', 'fft_259_a', 'fft_270_a', 'fft_271_a', 'fft_274_a',
    'fft_285_a', 'fft_286_a', 'fft_289_a', 'fft_300_a', 'fft_301_a',
    'fft_304_a', 'fft_315_a', 'fft_316_a', 'fft_319_a', 'fft_325_a',
    'fft_330_a', 'fft_331_a', 'fft_334_a', 'fft_335_a', 'fft_336_a',
    'fft_339_a', 'fft_345_a', 'fft_346_a', 'fft_349_a', 'fft_361_a',
    'fft_364_a', 'fft_370_a', 'fft_375_a', 'fft_390_a', 'fft_391_a',
    'fft_394_a', 'fft_395_a', 'fft_406_a','fft_409_a', 'fft_415_a', 'fft_420_a', 'fft_421_a', 'fft_424_a',
    'fft_435_a', 'fft_436_a', 'fft_439_a', 'fft_450_a', 'fft_451_a',
    'fft_454_a', 'fft_465_a', 'fft_469_a', 'fft_480_a', 'fft_510_a',
    'fft_511_a', 'fft_514_a', 'fft_516_a', 'fft_519_a', 'fft_526_a',
    'fft_529_a', 'fft_540_a', 'fft_541_a', 'fft_544_a', 'fft_555_a',
    'fft_570_a', 'fft_585_a', 'fft_600_a', 'fft_601_a', 'fft_604_a',
    'fft_615_a', 'fft_616_a', 'fft_619_a', 'fft_630_a', 'fft_631_a',
    'fft_634_a', 'fft_640_a', 'fft_645_a', 'fft_660_a', 'fft_675_a',
    'fft_690_a', 'fft_691_a', 'fft_694_a', 'fft_700_a', 'fft_705_a',
    'fft_711_a', 'fft_714_a', 'fft_720_a', 'fft_721_a', 'fft_724_a',
    'fft_735_a', 'fft_736_a', 'fft_739_a', 'mean_0_b', 'mean_2_b',
    'mean_3_b', 'mean_d_0_b2', 'mean_d_2_b2', 'mean_d_3_b2', 'mean_d_5_b',
    'mean_d_7_b', 'mean_d_8_b', 'mean_d_10_b', 'mean_d_12_b', 'mean_d_13_b',
    'mean_d_15_b', 'mean_d_17_b', 'mean_d_18_b', 'stddev_0_b', 'stddev_2_b',
    'moments_2_b', 'moments_5_b', 'moments_7_b', 'moments_12_b',
    'moments_15_b', 'moments_17_b', 'max_2_b', 'max_3_b', 'max_q_2_b',
    'max_q_3_b', 'max_q_8_b', 'max_q_13_b', 'max_q_18_b', 'min_0_b',
    'min_2_b', 'min_3_b', 'min_q_0_b', 'min_q_2_b', 'min_q_3_b',
    'min_q_5_b', 'min_q_7_b', 'min_q_8_b', 'min_q_10_b', 'min_q_12_b',
    'min_q_13_b', 'min_q_15_b', 'min_q_17_b', 'min_q_18_b', 'covmat_0_b',
    'covmat_1_b','covmat_3_b', 'covmat_4_b', 'covmat_5_b', 'covmat_8_b', 'covmat_9_b',
    'covmat_12_b', 'covmat_13_b', 'covmat_15_b', 'covmat_17_b',
    'covmat_20_b', 'covmat_36_b', 'covmat_37_b', 'covmat_44_b',
    'covmat_48_b', 'covmat_56_b', 'covmat_57_b', 'covmat_60_b',
    'covmat_61_b', 'covmat_69_b', 'covmat_96_b', 'covmat_97_b',
    'covmat_99_b', 'covmat_100_b', 'covmat_104_b', 'covmat_105_b',
    'covmat_108_b', 'covmat_112_b', 'covmat_113_b', 'covmat_116_b',
    'covmat_117_b', 'eigen_0_b', 'eigen_1_b', 'eigen_2_b', 'eigen_3_b',
    'logm_0_b', 'logm_8_b', 'logm_9_b', 'correlate_0_b', 'correlate_2_b',
    'correlate_3_b', 'correlate_4_b', 'correlate_6_b', 'correlate_8_b',
    'correlate_9_b', 'correlate_11_b', 'correlate_12_b', 'correlate_14_b',
    'correlate_15_b', 'correlate_18_b', 'correlate_20_b', 'correlate_21_b',
    'correlate_22_b', 'correlate_23_b', 'correlate_24_b', 'correlate_27_b',
    'correlate_28_b', 'correlate_30_b', 'correlate_32_b', 'correlate_33_b',
    'correlate_36_b', 'correlate_37_b', 'correlate_39_b', 'correlate_40_b',
    'correlate_42_b', 'correlate_45_b', 'correlate_47_b', 'correlate_48_b',
    'correlate_51_b', 'correlate_53_b', 'correlate_54_b', 'correlate_55_b',
    'correlate_56_b', 'correlate_57_b', 'correlate_60_b', 'correlate_61_b',
    'correlate_63_b', 'correlate_65_b', 'correlate_66_b', 'correlate_68_b',
    'correlate_69_b', 'fft_0_b', 'fft_1_b', 'fft_4_b', 'fft_10_b',
    'fft_15_b', 'fft_16_b', 'fft_19_b', 'fft_31_b', 'fft_34_b', 'fft_46_b',
    'fft_49_b', 'fft_60_b', 'fft_61_b', 'fft_64_b', 'fft_75_b', 'fft_79_b',
    'fft_90_b', 'fft_106_b', 'fft_120_b', 'fft_121_b','fft_124_b', 'fft_135_b', 'fft_151_b', 'fft_154_b', 'fft_165_b',
    'fft_166_b', 'fft_169_b', 'fft_180_b', 'fft_184_b', 'fft_195_b',
    'fft_196_b', 'fft_199_b', 'fft_210_b', 'fft_211_b', 'fft_214_b',
    'fft_221_b', 'fft_224_b', 'fft_225_b', 'fft_226_b', 'fft_229_b',
    'fft_240_b', 'fft_241_b', 'fft_244_b', 'fft_255_b', 'fft_256_b',
    'fft_259_b', 'fft_265_b', 'fft_270_b', 'fft_280_b', 'fft_285_b',
    'fft_286_b', 'fft_289_b', 'fft_300_b', 'fft_301_b', 'fft_304_b',
    'fft_305_b', 'fft_315_b', 'fft_316_b', 'fft_319_b', 'fft_330_b',
    'fft_331_b', 'fft_334_b', 'fft_335_b', 'fft_345_b', 'fft_346_b',
    'fft_349_b', 'fft_360_b', 'fft_361_b', 'fft_364_b', 'fft_375_b',
    'fft_376_b', 'fft_379_b', 'fft_390_b', 'fft_391_b', 'fft_394_b',
    'fft_395_b', 'fft_405_b', 'fft_415_b', 'fft_420_b', 'fft_424_b',
    'fft_435_b', 'fft_436_b', 'fft_439_b', 'fft_441_b', 'fft_450_b',
    'fft_455_b', 'fft_456_b', 'fft_459_b', 'fft_465_b', 'fft_480_b',
    'fft_481_b', 'fft_484_b', 'fft_486_b', 'fft_489_b', 'fft_495_b',
    'fft_510_b', 'fft_525_b', 'fft_526_b', 'fft_529_b', 'fft_530_b',
    'fft_540_b', 'fft_541_b', 'fft_544_b', 'fft_555_b', 'fft_570_b',
    'fft_571_b', 'fft_574_b', 'fft_585_b', 'fft_590_b', 'fft_600_b',
    'fft_615_b', 'fft_616_b', 'fft_619_b', 'fft_620_b', 'fft_631_b',
    'fft_634_b', 'fft_640_b', 'fft_645_b', 'fft_660_b', 'fft_675_b',
    'fft_680_b', 'fft_690_b', 'fft_691_b', 'fft_694_b', 'fft_705_b',
    'fft_710_b', 'fft_720_b', 'fft_721_b', 'fft_724_b', 'fft_735_b'
]

data = data[feature_columns + ['label']]
print(f"Selected {len(feature_columns)} features + label column.\n")

# ----------------------------------------------------------
# 4. Train/Test Split
# ----------------------------------------------------------
X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y
)

print(f"Training samples: {X_train.shape[0]}  |  Testing samples: {X_test.shape[0]}\n")

# ----------------------------------------------------------
# 5. RANDOM FOREST MODEL
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

print(f"✅ Random Forest Accuracy: {rf_acc:.2f}%")
print("\nRandom Forest Classification Report:\n")
print(classification_report(y_test, rf_pred, target_names=label_mapping.keys()))

# ----------------------------------------------------------
# 6. XGBOOST MODEL
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

print(f"\n✅ XGBoost Accuracy: {xgb_acc:.2f}%")
print("\nXGBoost Classification Report:\n")
print(classification_report(y_test, xgb_pred, target_names=label_mapping.keys()))

# ----------------------------------------------------------
# 7. Confusion Matrices
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='g', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Random Forest Confusion Matrix ({rf_acc:.2f}%)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(label_mapping.keys())
axes[0].set_yticklabels(label_mapping.keys())

cm_xgb = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm_xgb, annot=True, fmt='g', cmap='Greens', ax=axes[1])
axes[1].set_title(f'XGBoost Confusion Matrix ({xgb_acc:.2f}%)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_xticklabels(label_mapping.keys())
axes[1].set_yticklabels(label_mapping.keys())

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 8. Feature Importance Visualization
# ----------------------------------------------------------
# Random Forest
importances_rf = rf.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1][:15]

plt.figure(figsize=(10, 5))
plt.title("Top 15 Important Features - Random Forest")
plt.bar(range(len(indices_rf)), importances_rf[indices_rf], align='center')
plt.xticks(range(len(indices_rf)), [X.columns[i] for i in indices_rf], rotation=90)
plt.tight_layout()
plt.show()

# XGBoost
importances_xgb = xgb.feature_importances_
indices_xgb = np.argsort(importances_xgb)[::-1][:15]

plt.figure(figsize=(10, 5))
plt.title("Top 15 Important Features - XGBoost")
plt.bar(range(len(indices_xgb)), importances_xgb[indices_xgb], align='center', color='orange')
plt.xticks(range(len(indices_xgb)), [X.columns[i] for i in indices_xgb], rotation=90)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 9. Summary Comparison
# ----------------------------------------------------------
print("\n============================")
print("Model Performance Comparison")
print("============================")
print(f"Random Forest Accuracy: {rf_acc:.2f}%")
print(f"XGBoost Accuracy:      {xgb_acc:.2f}%")

best_model = "Random Forest" if rf_acc > xgb_acc else "XGBoost"
print(f"\n🏆 Best Performing Model: {best_model}")

