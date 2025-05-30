# === MODULE IMPORTS === #
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# === DATA LOADING === #
df = pd.read_csv("/Users/ayca/Downloads/darwin/data.csv")

# === FEATURE ENGINEERING === #
feature_bases = [
    'paper_time',
    'mean_speed_on_paper',
    'mean_speed_in_air',
    'pressure_mean',
    'gmrt_on_paper',
    'gmrt_in_air'
]

for base in feature_bases:
    columns = [f"{base}{i}" for i in range(1, 26)]
    df[f"{base}_avg"] = df[columns].mean(axis=1)
    df[f"{base}_std"] = df[columns].std(axis=1)
    df[f"{base}_var"] = df[columns].var(axis=1)

avg_features = [f"{base}_avg" for base in feature_bases]
std_features = [f"{base}_std" for base in feature_bases]
var_features = [f"{base}_var" for base in feature_bases]
all_features = avg_features + std_features + var_features

# === MODELING === #
X = df[all_features]
y = df['class'].map({'H': 0, 'P': 1})

# === CLASS BALANCING (SMOTE) === #
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# === TRAIN-TEST SPLIT === #
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# === SCALING === #
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === RECURSIVE FEATURE ELIMINATION (RFE) === #
model = GradientBoostingClassifier(random_state=42)
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X_train_scaled, y_train)

# === SELECTED FEATURES AFTER RFE === #
selected_features = np.array(all_features)[rfe.support_]
print("\n=== Selected Features After RFE ===")
print(selected_features)

# === DROPPING FEATURES THAT CAUSED INCREASE IN ACCURACY === #
features_to_drop = [
    'mean_speed_in_air_var',
    'gmrt_in_air_var',
    'mean_speed_in_air_std',
    'pressure_mean_avg',
    'gmrt_on_paper_var'
]

# === FINAL FEATURE SET === #
final_features = [feature for feature in selected_features if feature not in features_to_drop]
print("\n=== Features After Drop Column Analysis ===")
print(final_features)

# === UPDATE TRAIN AND TEST SET === #
X_train_final = pd.DataFrame(X_train_scaled, columns=all_features)[final_features]
X_test_final = pd.DataFrame(X_test_scaled, columns=all_features)[final_features]

# === ENSEMBLE VOTING CLASSIFIER === #
voting_model = VotingClassifier(estimators=[
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('lr', LogisticRegression(random_state=42, max_iter=1000))
], voting='soft')

voting_model.fit(X_train_final, y_train)

# === PREDICTION AND EVALUATION === #
y_pred = voting_model.predict(X_test_final)

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# === CROSS-VALIDATION === #
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(voting_model, X_train_final, y_train, cv=cv, scoring='accuracy')

print("\n=== Cross-Validation Scores ===")
for fold_idx, score in enumerate(cross_val_scores):
    print(f"Fold {fold_idx + 1}: {score:.4f}")

print(f"\nAverage Cross-Validation Accuracy: {cross_val_scores.mean():.4f}")
print(f"Standard Deviation: {cross_val_scores.std():.4f}")
