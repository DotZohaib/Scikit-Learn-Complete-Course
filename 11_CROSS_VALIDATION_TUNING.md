# Part 13: Cross-Validation and Hyperparameter Tuning (`sklearn.model_selection`)

Ek single `train_test_split` par bharosa karna kyun khatarnaak ho sakta hai? Aur model ke behtareen hyperparameters (`C`, `max_depth`, `learning_rate`) bina **Test-Set Leakage** ke kaise dhoonde jayein? Is chapter mein hum [Scikit-learn Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html) aur [Grid/Randomized Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html) ke tamam 24 topics seekhenge.

---

## 1. Why One Train/Test Split May Be Unreliable & K-Fold Cross-Validation

- **Problem with a Single Split:** Agar aap ka dataset chota ya darmiyana hai (maslan 200 rows), to `random_state=1` rakhne par ho sakta hai saare aasan samples Test Set mein aa jayein (Accuracy = `98%`) aur `random_state=2` rakhne par mushkil samples Test Set mein aa jayein (Accuracy = `79%`)!
- **K-Fold Cross-Validation Solution (`cv=5`):**
  Training data ko **5 barabar hisson (Folds)** mein taqseem kiya jata hai. Model 5 dafa train hota hai:
  - Run 1: Fold 2,3,4,5 par Train $\rightarrow$ Fold 1 par Validate
  - Run 2: Fold 1,3,4,5 par Train $\rightarrow$ Fold 2 par Validate ... aur aakhir mein **5 scores ka Mean aur Standard Deviation** nikala jata hai!

---

## 2. All 9 Scikit-learn Cross-Validation Splitters

| Splitter Class | How It Splits Data | When You MUST Use It |
| :--- | :--- | :--- |
| **1. `KFold(n_splits=5, shuffle=True, random_state=42)`** | Standard $K$ folds | **Regression** problems ke liye default choice. |
| **2. `StratifiedKFold(n_splits=5)`** | Har fold mein classes ka tanasub (ratio) barabar rakhta hai | **Classification** (khaas kar imbalanced data) ke liye default choice! |
| **3. `GroupKFold(n_splits=5)`** | Ek hi group (maslan ek hi **Patient** ke 5 medical scans, ya ek hi **City** ke records) ko kabhi Train aur Validation dono mein ek sath nahi aane deta! | Jab ek hi person/hospital/device ke multiple rows hon (prevents Group Leakage!). |
| **4. `StratifiedGroupKFold`** | Group integrity BHI rakhta hai aur class balance BHI! | Medical classification jahan har patient ke multiple samples hon. |
| **5. `TimeSeriesSplit(n_splits=5)`** | **Past par Train, Future par Validate** (Rolling/Expanding window; kabhi shuffle nahi karta!) | Stock prices, sales forecasting, weather time-series ke liye! |
| **6. `ShuffleSplit` & 7. `StratifiedShuffleSplit`** | Random Monte-Carlo train/val splits (`n_splits=10, test_size=0.2`) | Jab folds ki tadad aur test_size ko azaadana control karna ho. |
| **8. `LeaveOneOut` (LOO)** | $N$ samples mein se $N-1$ par train aur sirf $1$ sample par test ($N$ models bante hain!) | Sirf bohat chote datasets ($N < 100$) ke liye; bare data par bohot slow aur high variance! |
| **9. `LeavePOut(p=2)`** | Har dafa $p$ samples bahar rakh kar $\binom{N}{p}$ models train karta hai | Combinatorial evaluation on tiny datasets. |

---

## 3. `cross_val_score`, `cross_validate`, and `cross_val_predict`

1. **`cross_val_score(estimator, X, y, cv=5, scoring="accuracy")`**:
   - Sirf ek metric ke $K$ validation scores ka 1D array return karta hai.
2. **`cross_validate(estimator, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"], return_train_score=True)`**:
   - **Professional Choice!** Ek hi waqt mein multiple metrics, fit/score timings, aur training scores (overfitting check karne ke liye) dictionary mein return karta hai.
3. **`cross_val_predict(estimator, X, y, cv=5, method="predict_proba")`**:
   - Har sample ke liye woh prediction deta hai jab woh sample validation fold mein tha (out-of-fold predictions).
   - **Limitation Warning:** Official docs ke mutabiq `cross_val_predict` ke output par global metric calculate karna fold-averaged `cross_val_score` se mukhtalif ho sakta hai; isay sirf visualization (jaise residual plots ya ROC curves) aur stacking/blending ke liye use karein.

---

## 4. Hyperparameter Tuning: `GridSearchCV`, `RandomizedSearchCV`, and Successive Halving

| Tuner Class | Search Strategy | When to Choose & Computational Cost |
| :--- | :--- | :--- |
| **1. `GridSearchCV`** | `param_grid` ke **har mumkina combination** ko azmata hai (maslan $4 \times 3 \times 2 = 24$ combinations $\times 5$ folds $= 120$ fits). | Jab hyperparameters kam hon ($2$–$3$ parameters) aur chota search space ho. |
| **2. `RandomizedSearchCV`** | Continuous distributions (`scipy.stats.loguniform`, `randint`) ya lists mein se **sirf `n_iter` (maslan 20) random combinations** azmata hai! | Jab search space bara ho—90% waqt mein `GridSearchCV` jitna acha model 5x kam waqt mein dhoond leta hai! |
| **3. `HalvingGridSearchCV` & 4. `HalvingRandomSearchCV`** | **Successive Halving Tournament**: Pehle thore se samples (resources) par saare candidates chalata hai, kamzor aadhe candidates bahar nikal deta hai, aur jeetne walon ko zyada data de kar agle round mein le jata hai! | **Experimental Header Required:** `from sklearn.experimental import enable_halving_search_cv` pehle import karna zaroori hai! |

### Important Tuning Parameters (`n_jobs`, `scoring`, `cv`, `refit`) & Nested CV
- `n_jobs=-1`: Computer ke tamam CPU cores par parallel training chalata hai.
- `refit=True` (default): Best hyperparameter milne ke baad poore `X_train` par best model ko dobara automatically `.fit()` kar deta hai (`search.best_estimator_`).
- **Nested Cross-Validation:** Jab aap model tune BHI karna chahte hain aur us tuning process ki unbiased generalization performance BHI report karna chahte hain, to `cross_val_score(GridSearchCV(...), X, y, cv=5)` chalate hain (Inner loop hyperparameters chunta hai, Outer loop unbiased score deta hai!).

---

## 5. Complete Working Code: Splitters, `cross_validate`, `GridSearchCV`, `RandomizedSearchCV`, `HalvingRandomSearchCV` & Nested CV

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GroupKFold, TimeSeriesSplit,
    cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
)
# Required explicit import to enable Successive Halving
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data & Hold Out Final Test Set
bc = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    bc.data, bc.target, test_size=0.2, random_state=42, stratify=bc.target
)

pipe_svc = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, random_state=42))
])

# 2. Multi-Metric Cross-Validation with cross_validate
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_res = cross_validate(
    pipe_svc, X_train, y_train, cv=skf,
    scoring=["accuracy", "f1", "roc_auc"], return_train_score=True
)
print("=== 1. 5-Fold Stratified Cross-Validation Results ===")
print(f"Train Accuracy Mean: {cv_res['train_accuracy'].mean():.4f}")
print(f"Val Accuracy Mean  : {cv_res['test_accuracy'].mean():.4f} (+/- {cv_res['test_accuracy'].std():.4f})")
print(f"Val ROC-AUC Mean   : {cv_res['test_roc_auc'].mean():.4f}")

# 3. GridSearchCV over Pipeline (Notice 'svc__C' syntax!)
param_grid = {
    "svc__C": [0.1, 1.0, 10.0],
    "svc__gamma": ["scale", 0.01, 0.1]
}
grid = GridSearchCV(pipe_svc, param_grid=param_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)

print("\n=== 2. GridSearchCV Results ===")
print("Best Params :", grid.best_params_)
print(f"Best CV AUC : {grid.best_score_:.4f}")
print(f"Test Set AUC: {grid.score(X_test, y_test):.4f}")

# 4. RandomizedSearchCV & HalvingRandomSearchCV on RandomForest
rf = RandomForestClassifier(random_state=42)
param_dist = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 5, 8, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"]
}

rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=6, cv=3, random_state=42)
rand_search.fit(X_train, y_train)

halving_search = HalvingRandomSearchCV(rf, param_distributions=param_dist, factor=2, cv=3, random_state=42)
halving_search.fit(X_train, y_train)

print("\n=== 3. RandomizedSearchCV vs HalvingRandomSearchCV ===")
print("Randomized Best Score:", round(rand_search.best_score_, 4), "| Params:", rand_search.best_params_)
print("Halving Best Score   :", round(halving_search.best_score_, 4), "| Params:", halving_search.best_params_)

# 5. Nested Cross-Validation Demo (Unbiased evaluation of tuning procedure)
nested_scores = cross_val_score(grid, X_train, y_train, cv=3, scoring="roc_auc")
print(f"\n=== 4. Nested CV Mean ROC-AUC: {nested_scores.mean():.4f} ===")

# 6. Quick Demo of GroupKFold and TimeSeriesSplit indices
print("\n=== 5. TimeSeriesSplit Indices (Notice train always precedes test!) ===")
ts_cv = TimeSeriesSplit(n_splits=3)
for fold, (tr_idx, val_idx) in enumerate(ts_cv.split(np.arange(12)), 1):
    print(f"  Fold {fold} -> Train indices: {tr_idx} | Val indices: {val_idx}")
```

### Expected Output
```text
=== 1. 5-Fold Stratified Cross-Validation Results ===
Train Accuracy Mean: 0.9857
Val Accuracy Mean  : 0.9736 (+/- 0.0110)
Val ROC-AUC Mean   : 0.9949

=== 2. GridSearchCV Results ===
Best Params : {'svc__C': 10.0, 'svc__gamma': 0.01}
Best CV AUC : 0.9968
Test Set AUC: 0.9977

=== 3. RandomizedSearchCV vs HalvingRandomSearchCV ===
Randomized Best Score: 0.9626 | Params: {'n_estimators': 150, 'min_samples_split': 2, 'max_features': 'log2', 'max_depth': 8}
Halving Best Score   : 0.9604 | Params: {'n_estimators': 100, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': None}

=== 4. Nested CV Mean ROC-AUC: 0.9958 ===

=== 5. TimeSeriesSplit Indices (Notice train always precedes test!) ===
  Fold 1 -> Train indices: [0 1 2] | Val indices: [3 4 5]
  Fold 2 -> Train indices: [0 1 2 3 4 5] | Val indices: [6 7 8]
  Fold 3 -> Train indices: [0 1 2 3 4 5 6 7 8] | Val indices: [ 9 10 11]
```

---

## 6. Practice Exercises & Solutions

### Exercises
1. **Beginner:** `load_iris()` par `LogisticRegression(max_iter=500)` ka `StratifiedKFold(n_splits=5)` ke sath `cross_val_score` nikalen aur 5 folds ka mean print karein.
2. **Intermediate:** Farz karein 8 samples 4 patients (`groups = [1, 1, 2, 2, 3, 3, 4, 4]`) se aaye hain. `GroupKFold(n_splits=2)` chala kar verify karein ke kisi bhi patient ke do samples alag folds mein nahi gaye.
3. **Challenge:** `GridSearchCV` mein multiple metrics (`scoring=['accuracy', 'f1_macro']`) pass karein aur `refit='f1_macro'` set kar ke `load_wine()` par run karein.

### Solutions
```python
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import StratifiedKFold, GroupKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# 1. Beginner
iris = load_iris()
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
scores = cross_val_score(pipe, iris.data, iris.target, cv=StratifiedKFold(5))
print("5-Fold Mean Accuracy:", round(scores.mean(), 4))

# 2. Intermediate
X_g = np.zeros((8, 2))
groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
for tr, val in GroupKFold(n_splits=2).split(X_g, groups=groups):
    print("Train Patient IDs:", set(groups[tr]), "| Val Patient IDs:", set(groups[val]))

# 3. Challenge
wine = load_wine()
gs = GridSearchCV(
    pipe, {"logisticregression__C": [0.1, 1.0, 10.0]},
    scoring=["accuracy", "f1_macro"], refit="f1_macro", cv=3
).fit(wine.data, wine.target)
print("Best F1-Macro:", round(gs.best_score_, 4))
```

---
**Next Chapter:** [12_PIPELINES.md](12_PIPELINES.md)
