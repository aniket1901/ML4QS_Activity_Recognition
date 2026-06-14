# ML4QS Group 40 — Results Summary (person-independent)

Generated from the actual re-run of `modelling.ipynb`. Train = subject_1 + subject_3; Test = held-out subject_2. 5 balanced classes.

## Dataset / windowing

- Sampling rate (after resampling): **50 Hz**
- Window: **2 s, 50% overlap** (1 s step)
- Windows total: **3360** | train (subj 1+3): **2240** | test (subj 2): **1120**
- Per activity: 672 total (448 train / 224 test), 5 activities ['smoking', 'typing', 'idle', 'cooking', 'exercising']
- Feature counts: time **156** + frequency **84** + correlation **9** = **249** total

## Results table (sorted by test macro-F1)

| Model | Best hyperparameters | Grouped-CV acc | Test acc | Test macro-F1 |
|---|---|---|---|---|
| Stacking (SVM+RF+kNN+DT → LR) | bases: tuned SVM+RF+kNN+DT; final: LogReg | 0.9920 | 0.9295 | 0.9287 |
| Voting (hard) | bases: tuned SVM+RF+kNN+DT | 0.9924 | 0.9277 | 0.9271 |
| Voting (soft) | bases: tuned SVM+RF+kNN+DT | 0.9929 | 0.9223 | 0.9213 |
| Gradient Boosting | {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 200} | 0.9893 | 0.9036 | 0.9016 |
| SVM (RBF) | {'C': 100, 'gamma': 0.001} | 0.9915 | 0.8893 | 0.8892 |
| kNN (k=3) | {'n_neighbors': 3} | 0.9920 | 0.8848 | 0.8840 |
| Naive Bayes | defaults (no tuning) | 0.9353 | 0.8812 | 0.8792 |
| Random Forest | {'max_depth': None, 'max_features': 0.3, 'n_estimators': 300} | 0.9879 | 0.8804 | 0.8762 |
| Decision Tree | {'max_depth': 7} | 0.9362 | 0.8250 | 0.8197 |

## Hyperparameter grids searched (GridSearchCV under GroupKFold)

- **kNN** — grid: `{'n_neighbors': [1, 3, 5, 7, 9, 11, 15]}` → best: `{'n_neighbors': 3}`
- **Decision Tree** — grid: `{'max_depth': [3, 5, 7, 10, 15, None]}` → best: `{'max_depth': 7}`
- **Random Forest** — grid: `{'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'max_features': ['sqrt', 0.3]}` → best: `{'max_depth': None, 'max_features': 0.3, 'n_estimators': 300}`
- **SVM (RBF)** — grid: `{'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.001, 0.01, 0.1]}` → best: `{'C': 100, 'gamma': 0.001}`
- **Naive Bayes** — grid: `no tuning (GaussianNB defaults)` → best: `defaults (no tuning)`
- **Gradient Boosting** — grid: `{'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [2, 3]}` → best: `{'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 200}`

## Grouped-CV vs test gap (top models) + LOSO

| Model | Grouped-CV acc | Test acc | CV→test gap |
|---|---|---|---|
| Stacking (SVM+RF+kNN+DT → LR) | 0.9920 | 0.9295 | +0.0625 |
| Voting (hard) | 0.9924 | 0.9277 | +0.0647 |
| Voting (soft) | 0.9929 | 0.9223 | +0.0705 |
| Gradient Boosting | 0.9893 | 0.9036 | +0.0857 |

Leave-one-subject-out (train on one training subject, validate on the other):

- **SVM (RBF)**: LOSO acc = **0.5286**
- **Random Forest**: LOSO acc = **0.4879**
- **Stacking**: LOSO acc = **0.5455**

**Interpretation:** grouped-CV (0.992) and test (0.929) are close (gap +0.062) for the best model, so grouped-CV is an honest predictor of new-recording performance; the lower LOSO number shows the harder jump to a genuinely new *person*.

## Best model: Stacking (SVM+RF+kNN+DT → LR)

- Test accuracy: **0.9295**, test macro-F1: **0.9287**

Per-class precision / recall / F1:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| cooking | 0.935 | 0.830 | 0.879 |
| exercising | 0.886 | 0.969 | 0.925 |
| idle | 0.991 | 0.991 | 0.991 |
| smoking | 0.886 | 0.871 | 0.878 |
| typing | 0.953 | 0.987 | 0.969 |

Main confusions (true → predicted):

- cooking → smoking: 19
- cooking → exercising: 18
- smoking → cooking: 10
- smoking → exercising: 10
- smoking → typing: 8
- exercising → smoking: 5

## Top 10 features by Random Forest importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | acc_y_rms | 0.0343 |
| 2 | acc_y_sma | 0.0340 |
| 3 | gyr_mag_p2p | 0.0172 |
| 4 | gyr_x_rms | 0.0171 |
| 5 | gyr_mag_range | 0.0165 |
| 6 | gyr_x_mad | 0.0159 |
| 7 | gyr_y_p2p | 0.0147 |
| 8 | gyr_z_iqr | 0.0146 |
| 9 | gyr_x_spec_energy | 0.0143 |
| 10 | lin_z_rms | 0.0138 |

## Feature-group ablation (RandomForest, person-independent)

| Feature set | n_features | Test accuracy | Test macro-F1 |
|---|---|---|---|
| Time-domain only | 156 | 0.8955 | 0.8926 |
| Frequency-domain only | 84 | 0.8268 | 0.8236 |
| Correlation only | 9 | 0.4732 | 0.4601 |
| All features | 249 | 0.8884 | 0.8858 |

## Figures (paths)

- `notebooks/outputs/eda/model_comparison.png` — all models, 4 metrics, random baseline 0.20
- `notebooks/outputs/eda/cm_best.png` — normalised confusion matrix of the best model
- `notebooks/outputs/eda/rf_feature_importance.png` — top-20 RF feature importances
