# 🖐️ Hand Gesture Classification

A supervised machine learning project that classifies **18 hand gestures** from landmark coordinate data collected via MediaPipe. The project covers the full ML lifecycle — EDA, preprocessing, training multiple models with various scaling strategies, hyperparameter tuning, and real-time inference via webcam.

All experiments are tracked and registered using **MLflow** on port `5002`.

---

## 🎯 Gestures / Labels

```
call, dislike, fist, four, like, mute, ok, one, palm, peace,
peace_inverted, rock, stop, stop_inverted, three, three2,
two_up, two_up_inverted
```

---

## 📊 Dataset

- **Features:** 63 columns — `x`, `y`, `z` coordinates for each of 21 hand landmarks
- **Label:** gesture class (18 unique classes)
- **Splits:** 60% Train · 20% Validation · 20% Test

---

## ⚙️ Preprocessing & Scaling

Four scaling strategies were tried across all models:

| Strategy | Description |
|---|---|
| **Raw** | No scaling — original landmark pixel coordinates |
| **MinMax** | Scales features to range [0, 1] |
| **Robust** | Scales using median and IQR — resistant to outliers |
| **Translate+Scale Norm** | Custom normalisation — subtracts wrist position, scales by middle finger distance. Best performer. |

---

## 🤖 Models Trained

### Logistic Regression (9 runs)
| Scaling | Notes |
|---------|-------|
| Raw | Default params |
| MinMax | Default params |
| Robust | Default params |
| Translate+Scale | Default params |
| Raw | GridSearch · no class_weight · 5-fold · C=[0.1,1,10] |
| Raw | GridSearch · balanced · 5-fold · C=[0.1,1,10] |
| MinMax | GridSearch · balanced · 5-fold · C=[0.1,1,10] |
| Translate+Scale | GridSearch · balanced · 5-fold · C=[0.1,1,10] |
| Raw | GridSearch · balanced · 3-fold · C=[0.1,1,5] |

### SVM (7 runs)
| Scaling | Notes |
|---------|-------|
| Raw | Default params |
| MinMax | Default params |
| Robust | Default params |
| Translate+Scale | Default params |
| Robust | GridSearch · 5-fold · C=[0.1,1,10] · kernels=[linear,rbf,poly] |
| Raw | GridSearch · 3-fold · C=[10,50,100] · kernels=[linear,rbf,poly] |
| Translate+Scale | **Final model** · C=100 · kernel=rbf · evaluated on Test set |

### Ensemble — Voting Classifier (LR + SVM) (2 runs)
| Scaling | Notes |
|---------|-------|
| Robust | Hard voting |
| Translate+Scale | Hard voting |

### Random Forest (5 runs)
| Scaling | Notes |
|---------|-------|
| Raw | Default params |
| Translate+Scale | Default params |
| Raw | GridSearch · cv=5 · depth=[None,10,20] · estimators=[100,200,500] |
| Raw | GridSearch · StratifiedKFold(5) · depth=[10,20] · evaluated on Test set |
| Translate+Scale | **Final model** · 500 estimators · depth=20 · evaluated on Test set |

### XGBoost (5 runs)
| Scaling | Notes |
|---------|-------|
| Raw | Default params |
| Translate+Scale | Default params |
| Raw | GridSearch · cv=5 · lr=[0.01,0.1] · depth=[3,6] |
| Translate+Scale | GridSearch · cv=5 · lr=[0.01,0.1] · depth=[3,6] |
| Translate+Scale | **Final model** · lr=0.1 · depth=6 · evaluated on Test set |

---

## 🏆 Best Results

| Model | Best Config | Accuracy |
|---|---|---|
| **SVM** | C=100 · rbf · Translate+Scale Norm | **98.44%** |
| XGBoost | GridSearch · T+S Norm · lr=0.1 · depth=6 | 98.15% |
| Random Forest | T+S Norm · 500 estimators · depth=20 | 97.45% |
| Ensemble (LR+SVM) | Hard Voting · T+S Norm | 93.00% |
| Logistic Regression | Robust Scaled | 90.77% |

> After evaluating all models, the top 3 performers — **SVM**, **XGBoost**, and **Random Forest** — were taken forward and tested on the held-out test set. After comparing their test results, **SVM (C=100, rbf, Translate+Scale Norm)** was selected as the final model and is used for real-time webcam inference.

---

## 📦 Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost mlflow plotly opencv-python mediapipe scipy
```

---

## 🚀 How to Run

### 1. Start MLflow UI
```bash
mlflow ui --port 5002
```
Then open **http://127.0.0.1:5002** in your browser.

### 2. Run the training script
```bash
python hand_gesture_project.py
```

This will train all **28 runs** across 5 model families and log everything to MLflow automatically.

---

## 📈 MLflow Tracking

Every run logs:
- **Parameters** — model hyperparameters, scaling strategy, CV folds
- **Metrics** — `val_accuracy`, `test_accuracy`, `best_cv_accuracy`
- **Model artifact** — serialised model saved per run
- **Tags** — `model_type`, `scaling`, `stage`

### Registered Models (visible under "Models" tab in UI)

| Registered Name | Versions |
|---|---|
| `LogisticRegression_HandGesture` | 9 |
| `SVM_HandGesture` | 7 |
| `Ensemble_LR_SVM_HandGesture` | 2 |
| `RandomForest_HandGesture` | 5 |
| `XGBoost_HandGesture` | 5 |

---

## 🎥 Real-Time Inference

After training, the script launches a **live webcam demo** using MediaPipe for hand landmark detection and the final SVM model for gesture classification.

- Press **`q`** to quit the webcam window
- Predictions are stabilised over a rolling window of 10 frames using mode voting
