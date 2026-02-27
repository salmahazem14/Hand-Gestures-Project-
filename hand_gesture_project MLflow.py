# ============================================================
#  Machine Learning - Supervised Learning
#  Hand Gesture Classification
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import plotly.graph_objects as go

mlflow.set_tracking_uri("http://127.0.0.1:5002")
mlflow.set_experiment("Hand_Gesture_Classification")

gestures_data = pd.read_csv("hand_landmarks_data.csv")

print(gestures_data.head())
print(gestures_data.columns)
gestures_data.info()
print(gestures_data["label"].unique())
print(gestures_data.value_counts("label"))
print("Duplicates:", gestures_data.duplicated().sum())
print("Has nulls:", gestures_data.isnull().values.any())


numeric_values = gestures_data.select_dtypes(include=['number'])

for i in range(0, len(numeric_values.columns), 8):
    numeric_values.iloc[:, i:i+8].boxplot(figsize=(10, 5))
    plt.xticks(rotation=45)
    plt.show()

for i in range(0, len(numeric_values.columns), 8):
    numeric_values.iloc[:, i:i+8].hist(figsize=(10, 5))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

corr = numeric_values.corr()
plt.figure(figsize=(30, 20))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Correlation Between Features")
plt.tight_layout()
plt.show()

le = LabelEncoder()
y_encoded = le.fit_transform(gestures_data['label'])

for col in numeric_values:
    print(f'{col}: {numeric_values[col].min()} - {numeric_values[col].max()}')

X_train, X_test_validate, y_train, y_test_validate = train_test_split(
    numeric_values, y_encoded, test_size=0.4, stratify=y_encoded, random_state=42
)
X_validate, X_test, y_validate, y_test = train_test_split(
    X_test_validate, y_test_validate, test_size=0.5,
    stratify=y_test_validate, random_state=42
)

minmax_scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = minmax_scaler.fit_transform(X_train)
val_minmax   = minmax_scaler.transform(X_validate)
test_minmax  = minmax_scaler.transform(X_test)

robust_scaler = RobustScaler()
robust_scaled_data = robust_scaler.fit_transform(X_train)
val_robust   = robust_scaler.transform(X_validate)
test_robust  = robust_scaler.transform(X_test)

def normalize_row(row):
    landmarks = row.values.reshape(21, 3)
    wrist = landmarks[0]
    landmarks[:, 0] -= wrist[0]
    landmarks[:, 1] -= wrist[1]
    mid_tip = landmarks[12]
    scale = np.sqrt(mid_tip[0]**2 + mid_tip[1]**2)
    if scale == 0:
        scale = 1e-6
    landmarks[:, 0] /= scale
    landmarks[:, 1] /= scale
    return landmarks.flatten()

X_normalized      = X_train.apply(normalize_row, axis=1, result_type='expand')
X_normalized.columns = X_train.columns
X_val_normalized  = X_validate.apply(normalize_row, axis=1, result_type='expand')
X_val_normalized.columns = X_train.columns
X_test_normalized = X_test.apply(normalize_row, axis=1, result_type='expand')
X_test_normalized.columns = X_train.columns

with mlflow.start_run(run_name="LR_Raw"):
    mlflow.set_tag("cell", "46")
    mlflow.set_tag("model", "LogisticRegression")
    mlflow.set_tag("scaling", "Raw")
    lr_model = LogisticRegression(max_iter=10000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_predict = lr_model.predict(X_validate)
    score = lr_model.score(X_validate, y_validate)
    mlflow.log_params({"max_iter": 10000, "random_state": 42, "scaling": "Raw"})
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(lr_model, "model")
    print(f"[Cell 46 · LR · Raw] val_accuracy={score:.4f}")

with mlflow.start_run(run_name="LR_MinMax"):
    mlflow.set_tag("cell", "48")
    mlflow.set_tag("model", "LogisticRegression")
    mlflow.set_tag("scaling", "MinMax")
    lr_model = LogisticRegression(max_iter=10000, random_state=42)
    lr_model.fit(data_normalized, y_train)
    lr_predict = lr_model.predict(val_minmax)
    score = lr_model.score(val_minmax, y_validate)
    mlflow.log_params({"max_iter": 10000, "random_state": 42, "scaling": "MinMax"})
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(lr_model, "model")
    print(f"[Cell 48 · LR · MinMax] val_accuracy={score:.4f}")

with mlflow.start_run(run_name="LR_Robust"):
    mlflow.set_tag("cell", "50")
    mlflow.set_tag("model", "LogisticRegression")
    mlflow.set_tag("scaling", "Robust")
    lr_model = LogisticRegression(max_iter=10000, random_state=42)
    lr_model.fit(robust_scaled_data, y_train)
    lr_predict = lr_model.predict(val_robust)
    score = lr_model.score(val_robust, y_validate)
    mlflow.log_params({"max_iter": 10000, "random_state": 42, "scaling": "Robust"})
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(lr_model, "model")
    print(f"[Cell 50 · LR · Robust] val_accuracy={score:.4f}")

with mlflow.start_run(run_name="LR_TranslateScale"):
    mlflow.set_tag("cell", "53")
    mlflow.set_tag("model", "LogisticRegression")
    mlflow.set_tag("scaling", "TranslateScale")
    lr_model = LogisticRegression(max_iter=10000, random_state=42)
    lr_model.fit(X_normalized, y_train)
    lr_predict = lr_model.predict(X_val_normalized)
    score = lr_model.score(X_val_normalized, y_validate)
    mlflow.log_params({"max_iter": 10000, "random_state": 42, "scaling": "TranslateScale"})
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(lr_model, "model")
    print(f"[Cell 53 · LR · T+S Norm] val_accuracy={score:.4f}")

with mlflow.start_run(run_name="LR_GridSearch_Raw"):
    mlflow.set_tag("cell", "55")
    mlflow.set_tag("model", "LogisticRegression_GridSearch")
    mlflow.set_tag("scaling", "Raw")
    mlflow.set_tag("class_weight", "None")
    param_grid = [
        {'penalty': ['l2'], 'C': [0.1, 1, 10]},
        {'penalty': ['l1'], 'C': [0.1, 1]},
        {'penalty': ['elasticnet'], 'C': [1], 'l1_ratio': [0.5]}
    ]
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_lr = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000, solver='saga'),
        param_grid=param_grid, scoring='accuracy',
        cv=stratified_cv, n_jobs=-1, verbose=1
    )
    grid_lr.fit(X_train, y_train)
    y_pred = grid_lr.predict(X_validate)
    score = accuracy_score(y_validate, y_pred)
    mlflow.log_params({**grid_lr.best_params_, "n_splits": 5, "scaling": "Raw", "class_weight": "None"})
    mlflow.log_metric("best_cv_accuracy", grid_lr.best_score_)
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(grid_lr.best_estimator_, "model")
    print(f"[Cell 55] Best: {grid_lr.best_params_} | CV: {grid_lr.best_score_:.4f} | Val: {score:.4f}")
    print(classification_report(y_validate, y_pred))

with mlflow.start_run(run_name="LR_GridSearch_Raw_Balanced"):
    mlflow.set_tag("cell", "56")
    mlflow.set_tag("model", "LogisticRegression_GridSearch")
    mlflow.set_tag("scaling", "Raw")
    mlflow.set_tag("class_weight", "balanced")
    param_grid = [
        {'penalty': ['l2'], 'C': [0.1, 1, 10]},
        {'penalty': ['l1'], 'C': [0.1, 1]},
        {'penalty': ['elasticnet'], 'C': [1], 'l1_ratio': [0.5]}
    ]
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_lr = GridSearchCV(
        estimator=LogisticRegression(class_weight='balanced', max_iter=1000, solver='saga'),
        param_grid=param_grid, scoring='accuracy',
        cv=stratified_cv, n_jobs=-1, verbose=1
    )
    grid_lr.fit(X_train, y_train)
    y_pred = grid_lr.predict(X_validate)
    score = accuracy_score(y_validate, y_pred)
    mlflow.log_params({**grid_lr.best_params_, "n_splits": 5, "scaling": "Raw", "class_weight": "balanced"})
    mlflow.log_metric("best_cv_accuracy", grid_lr.best_score_)
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(grid_lr.best_estimator_, "model")
    print(f"[Cell 56] Best: {grid_lr.best_params_} | CV: {grid_lr.best_score_:.4f} | Val: {score:.4f}")
    print(classification_report(y_validate, y_pred))

with mlflow.start_run(run_name="LR_GridSearch_MinMax_Balanced"):
    mlflow.set_tag("cell", "57")
    mlflow.set_tag("model", "LogisticRegression_GridSearch")
    mlflow.set_tag("scaling", "MinMax")
    mlflow.set_tag("class_weight", "balanced")
    param_grid = [
        {'penalty': ['l2'], 'C': [0.1, 1, 10]},
        {'penalty': ['l1'], 'C': [0.1, 1]},
        {'penalty': ['elasticnet'], 'C': [1], 'l1_ratio': [0.5]}
    ]
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_lr = GridSearchCV(
        estimator=LogisticRegression(class_weight='balanced', max_iter=1000, solver='saga'),
        param_grid=param_grid, scoring='accuracy',
        cv=stratified_cv, n_jobs=-1, verbose=1
    )
    grid_lr.fit(data_normalized, y_train)
    validation_normalized = minmax_scaler.transform(X_validate)
    y_pred = grid_lr.predict(validation_normalized)
    score = accuracy_score(y_validate, y_pred)
    mlflow.log_params({**grid_lr.best_params_, "n_splits": 5, "scaling": "MinMax", "class_weight": "balanced"})
    mlflow.log_metric("best_cv_accuracy", grid_lr.best_score_)
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(grid_lr.best_estimator_, "model")
    print(f"[Cell 57] Best: {grid_lr.best_params_} | CV: {grid_lr.best_score_:.4f} | Val: {score:.4f}")
    print(classification_report(y_validate, y_pred))

with mlflow.start_run(run_name="LR_GridSearch_TranslateScale"):
    mlflow.set_tag("cell", "59")
    mlflow.set_tag("model", "LogisticRegression_GridSearch")
    mlflow.set_tag("scaling", "TranslateScale")
    mlflow.set_tag("class_weight", "balanced")
    param_grid = [
        {'penalty': ['l2'], 'C': [0.1, 1, 10]},
        {'penalty': ['l1'], 'C': [0.1, 1]},
        {'penalty': ['elasticnet'], 'C': [1], 'l1_ratio': [0.5]}
    ]
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_lr = GridSearchCV(
        estimator=LogisticRegression(class_weight='balanced', max_iter=1000, solver='saga'),
        param_grid=param_grid, scoring='accuracy',
        cv=stratified_cv, n_jobs=-1, verbose=1
    )
    grid_lr.fit(X_normalized, y_train)
    y_pred = grid_lr.predict(X_val_normalized)
    score = accuracy_score(y_validate, y_pred)
    mlflow.log_params({**grid_lr.best_params_, "n_splits": 5, "scaling": "TranslateScale", "class_weight": "balanced"})
    mlflow.log_metric("best_cv_accuracy", grid_lr.best_score_)
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(grid_lr.best_estimator_, "model")
    print(f"[Cell 59] Best: {grid_lr.best_params_} | CV: {grid_lr.best_score_:.4f} | Val: {score:.4f}")
    print(classification_report(y_validate, y_pred))

with mlflow.start_run(run_name="LR_GridSearch_Raw_3fold"):
    mlflow.set_tag("cell", "60")
    mlflow.set_tag("model", "LogisticRegression_GridSearch")
    mlflow.set_tag("scaling", "Raw")
    mlflow.set_tag("class_weight", "balanced")
    param_grid = [
        {'penalty': ['l2'], 'C': [0.1, 1, 5]},
        {'penalty': ['l1'], 'C': [0.1, 1]},
        {'penalty': ['elasticnet'], 'C': [1], 'l1_ratio': [0.5]}
    ]
    stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_lr = GridSearchCV(
        estimator=LogisticRegression(class_weight='balanced', max_iter=1000, solver='saga'),
        param_grid=param_grid, scoring='accuracy',
        cv=stratified_cv, n_jobs=-1, verbose=1
    )
    grid_lr.fit(X_train, y_train)
    y_pred = grid_lr.predict(X_validate)
    score = accuracy_score(y_validate, y_pred)
    mlflow.log_params({**grid_lr.best_params_, "n_splits": 3, "scaling": "Raw", "class_weight": "balanced"})
    mlflow.log_metric("best_cv_accuracy", grid_lr.best_score_)
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(grid_lr.best_estimator_, "model")
    print(f"[Cell 60] Best: {grid_lr.best_params_} | CV: {grid_lr.best_score_:.4f} | Val: {score:.4f}")
    print(classification_report(y_validate, y_pred))

with mlflow.start_run(run_name="SVM_Raw"):
    mlflow.set_tag("cell", "62")
    mlflow.set_tag("model", "SVC")
    mlflow.set_tag("scaling", "Raw")
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train, y_train)
    svm_prediction = svm_model.predict(X_validate)
    score = svm_model.score(X_validate, y_validate)
    mlflow.log_params({"C": 1.0, "kernel": "rbf", "gamma": "scale", "random_state": 42, "scaling": "Raw"})
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(svm_model, "model")
    print(f"[Cell 62 · SVM · Raw] val_accuracy={score:.4f}")

with mlflow.start_run(run_name="SVM_MinMax"):
    mlflow.set_tag("cell", "64")
    mlflow.set_tag("model", "SVC")
    mlflow.set_tag("scaling", "MinMax")
    svm_model = SVC(random_state=42)
    svm_model.fit(data_normalized, y_train)
    svm_predict = svm_model.predict(val_minmax)
    score = svm_model.score(val_minmax, y_validate)
    mlflow.log_params({"C": 1.0, "kernel": "rbf", "gamma": "scale", "random_state": 42, "scaling": "MinMax"})
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(svm_model, "model")
    print(f"[Cell 64 · SVM · MinMax] val_accuracy={score:.4f}")

with mlflow.start_run(run_name="SVM_Robust"):
    mlflow.set_tag("cell", "66")
    mlflow.set_tag("model", "SVC")
    mlflow.set_tag("scaling", "Robust")
    svm_model = SVC(random_state=42)
    svm_model.fit(robust_scaled_data, y_train)
    svm_predict = svm_model.predict(val_robust)
    score = svm_model.score(val_robust, y_validate)
    mlflow.log_params({"C": 1.0, "kernel": "rbf", "gamma": "scale", "random_state": 42, "scaling": "Robust"})
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(svm_model, "model")
    print(f"[Cell 66 · SVM · Robust] val_accuracy={score:.4f}")

with mlflow.start_run(run_name="SVM_TranslateScale"):
    mlflow.set_tag("cell", "68")
    mlflow.set_tag("model", "SVC")
    mlflow.set_tag("scaling", "TranslateScale")
    svm_model = SVC(random_state=42)
    svm_model.fit(X_normalized, y_train)
    svm_predict = svm_model.predict(X_val_normalized)
    score = svm_model.score(X_val_normalized, y_validate)
    mlflow.log_params({"C": 1.0, "kernel": "rbf", "gamma": "scale", "random_state": 42, "scaling": "TranslateScale"})
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(svm_model, "model")
    print(f"[Cell 68 · SVM · T+S Norm] val_accuracy={score:.4f}")

with mlflow.start_run(run_name="SVM_GridSearch_Robust_5fold_LowC"):
    mlflow.set_tag("cell", "70")
    mlflow.set_tag("model", "SVC_GridSearch")
    mlflow.set_tag("scaling", "Robust")
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_svm = GridSearchCV(
        estimator=SVC(), param_grid=param_grid_svm,
        scoring='accuracy', cv=stratified_cv, n_jobs=-1, verbose=1
    )
    grid_svm.fit(robust_scaled_data, y_train)
    robust_validate = robust_scaler.transform(X_validate)
    y_pred_svm = grid_svm.predict(robust_validate)
    score = accuracy_score(y_validate, y_pred_svm)
    mlflow.log_params({**grid_svm.best_params_, "n_splits": 5, "scaling": "Robust"})
    mlflow.log_metric("best_cv_accuracy", grid_svm.best_score_)
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(grid_svm.best_estimator_, "model")
    print(f"[Cell 70] Best: {grid_svm.best_params_} | CV: {grid_svm.best_score_:.4f} | Val: {score:.4f}")
    print(classification_report(y_validate, y_pred_svm, target_names=le.classes_))
    # Save best estimator for use in ensemble cells
    grid_svm_c70 = grid_svm
    robust_validate_c70 = robust_validate

with mlflow.start_run(run_name="SVM_GridSearch_Raw_3fold_HighC"):
    mlflow.set_tag("cell", "72")
    mlflow.set_tag("model", "SVC_GridSearch")
    mlflow.set_tag("scaling", "Raw")
    param_grid_svm = {
        'C': [10, 50, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_svm = GridSearchCV(
        estimator=SVC(), param_grid=param_grid_svm,
        scoring='accuracy', cv=stratified_cv, n_jobs=-1, verbose=1
    )
    grid_svm.fit(X_train, y_train)
    y_pred_svm = grid_svm.predict(X_validate)
    score = accuracy_score(y_validate, y_pred_svm)
    mlflow.log_params({**grid_svm.best_params_, "n_splits": 3, "scaling": "Raw"})
    mlflow.log_metric("best_cv_accuracy", grid_svm.best_score_)
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(grid_svm.best_estimator_, "model")
    print(f"[Cell 72] Best: {grid_svm.best_params_} | CV: {grid_svm.best_score_:.4f} | Val: {score:.4f}")
    print(classification_report(y_validate, y_pred_svm, target_names=le.classes_))
    # Save best estimator for use in ensemble cells
    grid_svm_c72 = grid_svm

with mlflow.start_run(run_name="SVM_C100_RBF_TranslateScale"):
    mlflow.set_tag("cell", "95")
    mlflow.set_tag("model", "SVC")
    mlflow.set_tag("scaling", "TranslateScale")
    mlflow.set_tag("stage", "final_test")
    svm_model = SVC(C=100, kernel='rbf', random_state=42)
    svm_model.fit(X_normalized, y_train)
    svm_predict = svm_model.predict(X_test_normalized)
    svm_score = svm_model.score(X_test_normalized, y_test)
    mlflow.log_params({"C": 100, "kernel": "rbf", "random_state": 42, "scaling": "TranslateScale"})
    mlflow.log_metric("test_accuracy", svm_score)
    mlflow.sklearn.log_model(svm_model, "model")
    print(f"[Cell 95 · SVM · Final] test_accuracy={svm_score:.2%}")


with mlflow.start_run(run_name="Ensemble_LR_SVM_Robust"):
    mlflow.set_tag("cell", "74")
    mlflow.set_tag("model", "VotingClassifier")
    mlflow.set_tag("scaling", "Robust")
    mlflow.set_tag("voting", "hard")
    ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=10000, random_state=42)),
            ('svm', grid_svm_c70.best_estimator_)
        ],
        voting='hard'
    )
    ensemble.fit(robust_scaled_data, y_train)
    y_pred_ensemble = ensemble.predict(robust_validate_c70)
    score = accuracy_score(y_validate, y_pred_ensemble)
    mlflow.log_params({
        "lr_max_iter": 10000, "voting": "hard", "scaling": "Robust",
        "svm_best_params": str(grid_svm_c70.best_params_)
    })
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(ensemble, "model")
    print(f"[Cell 74 · Ensemble · Robust] val_accuracy={score:.4f}")
    print(classification_report(y_validate, y_pred_ensemble, target_names=le.classes_))

with mlflow.start_run(run_name="Ensemble_LR_SVM_TranslateScale"):
    mlflow.set_tag("cell", "75")
    mlflow.set_tag("model", "VotingClassifier")
    mlflow.set_tag("scaling", "TranslateScale")
    mlflow.set_tag("voting", "hard")
    ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=10000, random_state=42)),
            ('svm', grid_svm_c72.best_estimator_)
        ],
        voting='hard'
    )
    ensemble.fit(X_normalized, y_train)
    y_pred_ensemble = ensemble.predict(X_val_normalized)
    score = accuracy_score(y_validate, y_pred_ensemble)
    mlflow.log_params({
        "lr_max_iter": 10000, "voting": "hard", "scaling": "TranslateScale",
        "svm_best_params": str(grid_svm_c72.best_params_)
    })
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(ensemble, "model")
    print(f"[Cell 75 · Ensemble · T+S Norm] val_accuracy={score:.4f}")
    print(classification_report(y_validate, y_pred_ensemble, target_names=le.classes_))

with mlflow.start_run(run_name="RF_Raw"):
    mlflow.set_tag("cell", "77")
    mlflow.set_tag("model", "RandomForestClassifier")
    mlflow.set_tag("scaling", "Raw")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    score = rf_model.score(X_validate, y_validate)
    mlflow.log_params({"n_estimators": 100, "max_depth": "None", "random_state": 42, "scaling": "Raw"})
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(rf_model, "model")
    print(f"[Cell 77 · RF · Raw] val_accuracy={score:.4f}")

with mlflow.start_run(run_name="RF_TranslateScale"):
    mlflow.set_tag("cell", "78")
    mlflow.set_tag("model", "RandomForestClassifier")
    mlflow.set_tag("scaling", "TranslateScale")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_normalized, y_train)
    score = rf_model.score(X_val_normalized, y_validate)
    mlflow.log_params({"n_estimators": 100, "max_depth": "None", "random_state": 42, "scaling": "TranslateScale"})
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(rf_model, "model")
    print(f"[Cell 78 · RF · T+S Norm] val_accuracy={score:.4f}")

with mlflow.start_run(run_name="RF_GridSearch_Raw"):
    mlflow.set_tag("cell", "80")
    mlflow.set_tag("model", "RandomForestClassifier_GridSearch")
    mlflow.set_tag("scaling", "Raw")
    param_grid_rf = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_rf = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid_rf, scoring='accuracy',
        cv=5, n_jobs=-1, verbose=1
    )
    grid_rf.fit(X_train, y_train)
    y_pred_rf = grid_rf.predict(X_validate)
    score = accuracy_score(y_validate, y_pred_rf)
    mlflow.log_params({**grid_rf.best_params_, "n_splits": 5, "scaling": "Raw"})
    mlflow.log_metric("best_cv_accuracy", grid_rf.best_score_)
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(grid_rf.best_estimator_, "model")
    print(f"[Cell 80] Best: {grid_rf.best_params_} | CV: {grid_rf.best_score_:.4f} | Val: {score:.4f}")
    print(classification_report(y_validate, y_pred_rf, target_names=le.classes_))

with mlflow.start_run(run_name="RF_GridSearch_Raw_TestSet"):
    mlflow.set_tag("cell", "81")
    mlflow.set_tag("model", "RandomForestClassifier_GridSearch")
    mlflow.set_tag("scaling", "Raw")
    mlflow.set_tag("eval_on", "test_set")
    param_grid_rf = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20],
        'min_samples_split': [4, 5],
        'min_samples_leaf': [1, 2]
    }
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_rf = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid_rf, scoring='accuracy',
        cv=stratified_cv, n_jobs=-1, verbose=1
    )
    grid_rf.fit(X_train, y_train)
    y_pred_rf = grid_rf.predict(X_test)
    score = accuracy_score(y_test, y_pred_rf)
    mlflow.log_params({**grid_rf.best_params_, "n_splits": 5, "scaling": "Raw"})
    mlflow.log_metric("best_cv_accuracy", grid_rf.best_score_)
    mlflow.log_metric("test_accuracy", score)
    mlflow.sklearn.log_model(grid_rf.best_estimator_, "model")
    print(f"[Cell 81] Best: {grid_rf.best_params_} | CV: {grid_rf.best_score_:.4f} | Test: {score:.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

with mlflow.start_run(run_name="RF_TranslateScale"):
    mlflow.set_tag("cell", "97")
    mlflow.set_tag("model", "RandomForestClassifier")
    mlflow.set_tag("scaling", "TranslateScale")
    mlflow.set_tag("stage", "final_test")
    rf_model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
    rf_model.fit(X_normalized, y_train)
    rf_predict = rf_model.predict(X_test_normalized)
    rf_score = rf_model.score(X_test_normalized, y_test)
    mlflow.log_params({"n_estimators": 500, "max_depth": 20, "random_state": 42, "scaling": "TranslateScale"})
    mlflow.log_metric("test_accuracy", rf_score)
    mlflow.sklearn.log_model(rf_model, "model")
    print(f"[Cell 97 · RF · Final] test_accuracy={rf_score:.2%}")

with mlflow.start_run(run_name="XGB_Raw"):
    mlflow.set_tag("cell", "83")
    mlflow.set_tag("model", "XGBClassifier")
    mlflow.set_tag("scaling", "Raw")
    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    score = xgb_model.score(X_validate, y_validate)
    mlflow.log_params({**xgb_model.get_params(), "scaling": "Raw"})
    mlflow.log_metric("val_accuracy", score)
    mlflow.xgboost.log_model(xgb_model, "model")
    print(f"[Cell 83 · XGB · Raw] val_accuracy={score:.4f}")

with mlflow.start_run(run_name="XGB_TranslateScale"):
    mlflow.set_tag("cell", "85")
    mlflow.set_tag("model", "XGBClassifier")
    mlflow.set_tag("scaling", "TranslateScale")
    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_normalized, y_train)
    score = xgb_model.score(X_val_normalized, y_validate)
    mlflow.log_params({**xgb_model.get_params(), "scaling": "TranslateScale"})
    mlflow.log_metric("val_accuracy", score)
    mlflow.xgboost.log_model(xgb_model, "model")
    print(f"[Cell 85 · XGB · T+S Norm] val_accuracy={score:.4f}")

with mlflow.start_run(run_name="XGB_GridSearch_Raw"):
    mlflow.set_tag("cell", "87")
    mlflow.set_tag("model", "XGBClassifier_GridSearch")
    mlflow.set_tag("scaling", "Raw")
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1]
    }
    grid_xgb = GridSearchCV(
        estimator=xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
        param_grid=param_grid_xgb, scoring='accuracy',
        cv=5, n_jobs=-1, verbose=1
    )
    grid_xgb.fit(X_train, y_train)
    y_pred_xgb = grid_xgb.predict(X_validate)
    score = accuracy_score(y_validate, y_pred_xgb)
    mlflow.log_params({**grid_xgb.best_params_, "n_splits": 5, "scaling": "Raw"})
    mlflow.log_metric("best_cv_accuracy", grid_xgb.best_score_)
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(grid_xgb.best_estimator_, "model")
    print(f"[Cell 87] Best: {grid_xgb.best_params_} | CV: {grid_xgb.best_score_:.4f} | Val: {score:.4f}")
    print(classification_report(y_validate, y_pred_xgb, target_names=le.classes_))

with mlflow.start_run(run_name="XGB_GridSearch_TranslateScale"):
    mlflow.set_tag("cell", "89")
    mlflow.set_tag("model", "XGBClassifier_GridSearch")
    mlflow.set_tag("scaling", "TranslateScale")
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1]
    }
    grid_xgb = GridSearchCV(
        estimator=xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
        param_grid=param_grid_xgb, scoring='accuracy',
        cv=5, n_jobs=-1, verbose=1
    )
    grid_xgb.fit(X_normalized, y_train)
    y_pred_xgb = grid_xgb.predict(X_val_normalized)
    score = accuracy_score(y_validate, y_pred_xgb)
    mlflow.log_params({**grid_xgb.best_params_, "n_splits": 5, "scaling": "TranslateScale"})
    mlflow.log_metric("best_cv_accuracy", grid_xgb.best_score_)
    mlflow.log_metric("val_accuracy", score)
    mlflow.sklearn.log_model(grid_xgb.best_estimator_, "model")
    print(f"[Cell 89] Best: {grid_xgb.best_params_} | CV: {grid_xgb.best_score_:.4f} | Val: {score:.4f}")
    print(classification_report(y_validate, y_pred_xgb, target_names=le.classes_))

with mlflow.start_run(run_name="XGB_TranslateScale"):
    mlflow.set_tag("cell", "99")
    mlflow.set_tag("model", "XGBClassifier")
    mlflow.set_tag("scaling", "TranslateScale")
    mlflow.set_tag("stage", "final_test")
    xgb_model = xgb.XGBClassifier(learning_rate=0.1, max_depth=6, random_state=42)
    xgb_model.fit(X_normalized, y_train)
    xgb_predict = xgb_model.predict(X_test_normalized)
    xgb_score = xgb_model.score(X_test_normalized, y_test)
    mlflow.log_params({"learning_rate": 0.1, "max_depth": 6, "random_state": 42, "scaling": "TranslateScale"})
    mlflow.log_metric("test_accuracy", xgb_score)
    mlflow.xgboost.log_model(xgb_model, "model")
    print(f"[Cell 99 · XGB · Final] test_accuracy={xgb_score:.2%}")

MODELS = {
    "Logistic Regression": {
        "color": "#38bdf8",
        "trials": [
            {"label": "Raw Features",                     "accuracy": 0.9067},
            {"label": "MinMax Scaled",                    "accuracy": 0.2995},
            {"label": "Robust Scaled",                    "accuracy": 0.9077},
            {"label": "Translate+Scale Norm",             "accuracy": 0.8479},
            {"label": "GridSearch · Robust (C=10, L2)",   "accuracy": 0.8544},
            {"label": "GridSearch · T+S Norm (C=10, L2)", "accuracy": 0.8528},
        ]
    },
    "SVM": {
        "color": "#34d399",
        "trials": [
            {"label": "Raw Features",                        "accuracy": 0.7063},
            {"label": "MinMax Scaled",                       "accuracy": 0.8064},
            {"label": "Robust Scaled",                       "accuracy": 0.8432},
            {"label": "T+S Norm ",                           "accuracy": 0.9215},
            {"label": "GridSearch · Linear (C=10)",          "accuracy": 0.9303},
            {"label": "GridSearch · RBF (C=100, T+S Norm)",  "accuracy": 0.9844},
        ]
    },
    "Ensemble (LR + SVM)": {
        "color": "#f472b6",
        "trials": [
            {"label": "Hard Voting · Robust Scaled", "accuracy": 0.92},
            {"label": "Hard Voting · T+S Norm",      "accuracy": 0.93},
        ]
    },
    "Random Forest": {
        "color": "#fb923c",
        "trials": [
            {"label": "Raw Features",                           "accuracy": 0.8129},
            {"label": "Translate+Scale Norm",                   "accuracy": 0.9745},
            {"label": "GridSearch v1 (no max_depth, 500 est.)", "accuracy": 0.82},
            {"label": "GridSearch v2 (depth=20, 500 est.)",     "accuracy": 0.82},
        ]
    },
    "XGBoost": {
        "color": "#a78bfa",
        "trials": [
            {"label": "Raw Features (Default)",                  "accuracy": 0.9089},
            {"label": "Translate+Scale Norm (Default)",          "accuracy": 0.9815},
            {"label": "GridSearch · Raw (lr=0.1, depth=6)",      "accuracy": 0.92},
            {"label": "GridSearch · T+S Norm (lr=0.1, depth=6)", "accuracy": 0.98},
        ]
    },
}


def plot_best_models():
    names, accs, colors, hover_texts = [], [], [], []
    for model, data in MODELS.items():
        valid = [(t["label"], t["accuracy"]) for t in data["trials"] if t["accuracy"] is not None]
        best_label, best_acc = max(valid, key=lambda x: x[1])
        names.append(model)
        accs.append(best_acc)
        colors.append(data["color"])
        hover_texts.append(f"<b>{model}</b><br>Best Trial: {best_label}<br>Accuracy: {best_acc:.2%}")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=accs,
        marker=dict(color=colors, opacity=0.85, line=dict(color=colors, width=1.5)),
        text=[f"{a:.1%}" for a in accs], textposition="outside",
        hovertext=hover_texts, hoverinfo="text",
    ))
    fig.update_layout(
        template="plotly_dark",
        title=dict(text="Best Accuracy — All Models", font=dict(size=18)),
        yaxis=dict(tickformat=".0%", range=[0, 1.15], title="Accuracy", showgrid=False, zeroline=False),
        xaxis=dict(title="Model", showgrid=False),
        plot_bgcolor="#0f172a", paper_bgcolor="#0b0f1a",
        font=dict(color="#e2e8f0"), height=450, showlegend=False,
    )
    fig.show()


def plot_model_trials():
    fig = go.Figure()
    first_model = list(MODELS.keys())[0]
    for model_name, data in MODELS.items():
        color  = data["color"]
        trials = data["trials"]
        labels  = [t["label"] for t in trials]
        accs    = [t["accuracy"] if t["accuracy"] is not None else 0 for t in trials]
        texts   = [f"{a:.1%}" if a > 0 else "N/A" for a in accs]
        max_acc = max(accs)
        bar_opacity = [1.0 if a == max_acc and a > 0 else 0.65 for a in accs]
        hover = [
            f"<b>{l}</b><br>Accuracy: {a:.2%}" if a > 0 else f"<b>{l}</b><br>Did not converge"
            for l, a in zip(labels, accs)
        ]
        fig.add_trace(go.Bar(
            x=labels, y=accs, name=model_name,
            visible=(model_name == first_model),
            marker=dict(color=[color]*len(accs), opacity=bar_opacity, line=dict(color=color, width=1.5)),
            text=texts, textposition="outside",
            hovertext=hover, hoverinfo="text",
        ))
    buttons = []
    for model_name in MODELS.keys():
        visibility = [model_name == m for m in MODELS.keys()]
        buttons.append(dict(
            label=model_name, method="update",
            args=[{"visible": visibility}, {"title": f"{model_name} — All Trials"}]
        ))
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"{first_model} — All Trials", font=dict(size=18)),
        yaxis=dict(tickformat=".0%", range=[0, 1.15], title="Accuracy", showgrid=False, zeroline=False),
        xaxis=dict(title="Trial", tickangle=-30, showgrid=False),
        plot_bgcolor="#0f172a", paper_bgcolor="#0b0f1a",
        font=dict(color="#e2e8f0"), height=500, showlegend=False,
        margin=dict(t=100, b=120),
        updatemenus=[dict(
            type="dropdown", direction="down", x=1, y=1.25, showactive=True,
            bgcolor="#1e293b", bordercolor="#334155", font=dict(color="#e2e8f0"),
            buttons=buttons,
        )]
    )
    fig.update_traces(hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font=dict(color="#e2e8f0")))
    fig.show()


print("=" * 55)
print("  Hand Gesture Classification — Model Comparison")
print("=" * 55)
print()
plot_best_models()
print()
plot_model_trials()

import cv2
import mediapipe as mp
from scipy import stats
from collections import deque

LABELS = [
    "call", "dislike", "fist", "four", "like",
    "mute", "ok", "one", "palm", "peace",
    "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted"
]


def preprocess_landmarks(hand_landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = landmarks[0].copy()
    landmarks[:, 0] -= wrist[0]
    landmarks[:, 1] -= wrist[1]
    mid_tip = landmarks[12].copy()
    scale = np.sqrt(mid_tip[0]**2 + mid_tip[1]**2)
    if scale == 0:
        scale = 1e-6
    landmarks[:, 0] /= scale
    landmarks[:, 1] /= scale
    return landmarks.flatten()


WINDOW_SIZE = 10
prediction_window = deque(maxlen=WINDOW_SIZE)


def stabilize(prediction):
    prediction_window.append(prediction)
    mode_result = stats.mode(prediction_window, keepdims=True)
    return mode_result.mode[0]


mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
print("Starting webcam... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(rgb_frame)

    label_text      = "No Hand Detected"
    confidence_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
            features          = preprocess_landmarks(hand_landmarks)
            prediction        = svm_model.predict([features])[0]
            stable_prediction = stabilize(prediction)
            label_text        = LABELS[stable_prediction]
            decision          = svm_model.decision_function([features])
            confidence        = np.max(decision)
            confidence_text   = f"Conf: {confidence:.2f}"

    cv2.rectangle(frame, (0, 0), (350, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesture: {label_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 150), 2)
    cv2.putText(frame, confidence_text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
