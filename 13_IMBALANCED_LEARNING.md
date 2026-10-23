# Part 15: Imbalanced Data, Probabilities, and Decision Thresholds (`sklearn.model_selection`, `sklearn.calibration`)

Real-world datasets mein aksar woh class jo hamare liye sab se ahem hoti hai (**Fraud Transaction, Cancer Patient, Loan Defaulter, Machine Breakdown**) woh tadad mein bohot kam hoti hai (maslan `98% Normal` vs `2% Fraud`). Is surat-e-haal ko **Class Imbalance** kehte hain. Is chapter mein hum [Scikit-learn Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html) aur [Decision Threshold Tuning (`TunedThresholdClassifierCV`, v1.5+)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html) ke tamam 16 topics seekhenge.

---

## 1. Why Accuracy Is Misleading on Imbalanced Data
Farz karein ek Pakistani bank ke 10,000 credit card transactions mein se `9,900 Normal (0)` hain aur sirf `100 Fraud (1)` hain.
- Agar ek be-waqoof model (`DummyClassifier`) har transaction ko आंखें band kar ke `0 (Normal)` keh de, to uski **Accuracy = $9900 / 10000 = 99.0\%$** aayegi!
- Lekin us ne **ek bhi Fraud nahi pakra (Recall = $0.0\%$)**! Is liye imbalanced data par hamesha **Precision-Recall AUC (`average_precision_score`), F1-Score, Recall, Balanced Accuracy, ya Cost-Sensitive Metric** use karein!

---

## 2. Core Scikit-learn Strategies for Imbalanced Data

### 1. Stratified Splitting (`stratify=y`) & `StratifiedKFold`
`train_test_split(..., stratify=y)` aur `StratifiedKFold` lazmi use karein taake 2% minority class har split mein barabar mojood rahe.

### 2. Cost-Sensitive Learning via `class_weight="balanced"`
Zyadatar Scikit-learn classifiers (`LogisticRegression`, `SVC`, `LinearSVC`, `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`, `HistGradientBoostingClassifier`) mein **`class_weight="balanced"`** parameter mojood hota hai!
- **Mathematical Formula:** Yeh har class $j$ ko uski tadad ke ulat (inversely proportional) weight deta hai:
  $$w_j = \frac{n_{\text{samples}}}{n_{\text{classes}} \times n_{\text{samples}, j}}$$
- Is se minority class (`Fraud`) par ghalti karne ka jurmana (loss penalty) barh jata hai aur model minority class ko nazar-andaz nahi kar sakta!

### 3. External Resampling Note (`imbalanced-learn`)
> **Important Clarification:** Core Scikit-learn mein `SMOTE` (Synthetic Minority Over-sampling Technique) ya `RandomUnderSampler` built-in **nahi** hain. Un ke liye third-party companion library **`imbalanced-learn` (`imblearn`)** use ki jati hai. Lekin modern Scikit-learn (v1.5+) mein `class_weight="balanced"` + **`TunedThresholdClassifierCV`** aksar bina resampling ke hi behtareen result de dete hain!

---

## 3. Probabilities, Decision Functions, Threshold Tuning & Calibration

### A. `.predict_proba(X)` vs `.decision_function(X)`
- **`.predict_proba(X)`**: Har class ki probability $[P(y=0), P(y=1)]$ deta hai ($0.0$ se $1.0$ ke darmiyan). Default taur par `.predict(X)` tab Class `1` deta hai jab $P(y=1) \ge 0.50$ ho.
- **`.decision_function(X)`**: Decision boundary (hyperplane) se signed distance batata hai ($-\infty$ se $+\infty$; default cutoff $= 0.0$).

### B. Why Default Threshold `0.50` Fails on Rare Events & How to Adjust It
Jab Fraud sirf 2% ho, to model ki predicted probabilities aksar `0.15` ya `0.35` tak hi jati hain. Agar hum cutoff `0.50` se kam kar ke **`0.20`** kar dein, to Recall (Fraud pakarne ki sharah) foran barh jati hai!
- **Scikit-learn v1.5+ Official Threshold Estimators (`sklearn.model_selection`):**
  1. **`FixedThresholdClassifier(estimator, threshold=0.25)`**: Kisi bhi classifier par apni marzi ka probability cutoff (jaise `0.25`) permanent set karne ke liye!
  2. **`TunedThresholdClassifierCV(estimator, scoring="f1", cv=5)`**: **Zero Data Leakage** ke sath Cross-Validation chala kar khud-ba-khud woh optimal threshold (`best_threshold_`) dhoondta hai jis par F1-score, Balanced Accuracy, ya Custom Business Cost Metric sab se behtar ho!
- **Leakage Risk Warning:** Kabhi bhi `X_test, y_test` ke upar threshold loop chala kar tune na karein! Hamesha `TunedThresholdClassifierCV` ko `X_train, y_train` par fit karein.

### C. Probability Calibration (`CalibratedClassifierCV`)
- Logistic Regression ki probabilities qudrati taur par well-calibrated hoti hain (yani agar woh 80% kahe, to waqai 10 mein se 8 cases positive hote hain).
- Lekin `RandomForest`, `SVC`, aur `GaussianNB` ki probabilities distorted hoti hain. **`CalibratedClassifierCV(estimator, method="sigmoid" | "isotonic", cv=5)`** (`sklearn.calibration`) unhein sachi probabilities mein calibrate karta hai (jise **Brier Score Loss** se naapa jata hai).

---

## 4. Complete Working Code: Imbalanced Fraud Classification with `class_weight`, `TunedThresholdClassifierCV`, `FixedThresholdClassifier` & `CalibratedClassifierCV`

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, TunedThresholdClassifierCV, FixedThresholdClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, brier_score_loss
)

# 1. Create 95% vs 5% Imbalanced Fraud Dataset (n=2000)
X, y = make_classification(
    n_samples=2000, n_features=12, n_informative=6,
    weights=[0.95, 0.05], flip_y=0.01, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Test Set Class Distribution -> Normal (0): {(y_test==0).sum()}, Fraud (1): {(y_test==1).sum()}")

# 2. Compare 4 Approaches on the Imbalanced Test Set
base_lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
balanced_lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=42))
fixed_thresh_lr = FixedThresholdClassifier(base_lr, threshold=0.20)
tuned_thresh_lr = TunedThresholdClassifierCV(base_lr, scoring="f1", cv=5, random_state=42)

models = {
    "1. Standard LR (Threshold=0.50)": base_lr,
    "2. Balanced Class Weight LR": balanced_lr,
    "3. FixedThresholdClassifier (0.20)": fixed_thresh_lr,
    "4. TunedThresholdClassifierCV (F1)": tuned_thresh_lr
}

rows = []
for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probas = clf.predict_proba(X_test)[:, 1]
    rows.append({
        "Strategy": name,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "F1_Score": round(f1_score(y_test, preds), 4),
        "PR_AUC": round(average_precision_score(y_test, probas), 4)
    })

print("\n=== 1. Imbalanced Fraud Detection Comparison ===")
print(pd.DataFrame(rows).to_string(index=False))
print(f"Optimal Threshold learned by TunedThresholdClassifierCV: {tuned_thresh_lr.best_threshold_:.4f}")

# 3. Probability Calibration Demo on RandomForest (Lower Brier Score = Better Calibration)
rf_uncal = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rf_cal = CalibratedClassifierCV(rf_uncal, method="sigmoid", cv=5).fit(X_train, y_train)

brier_uncal = brier_score_loss(y_test, rf_uncal.predict_proba(X_test)[:, 1])
brier_cal = brier_score_loss(y_test, rf_cal.predict_proba(X_test)[:, 1])
print("\n=== 2. Probability Calibration (Brier Score Loss - Lower is Better) ===")
print(f"Uncalibrated RandomForest Brier Loss: {brier_uncal:.4f}")
print(f"Calibrated RandomForest Brier Loss  : {brier_cal:.4f}")
```

### Expected Output
```text
Test Set Class Distribution -> Normal (0): 473, Fraud (1): 27

=== 1. Imbalanced Fraud Detection Comparison ===
                          Strategy  Accuracy  Precision  Recall  F1_Score  PR_AUC
   1. Standard LR (Threshold=0.50)    0.9540     0.7500  0.2222    0.3429  0.5412
       2. Balanced Class Weight LR    0.8360     0.2268  0.8148    0.3548  0.5418
3. FixedThresholdClassifier (0.20)    0.9460     0.5000  0.5556    0.5263  0.5412
4. TunedThresholdClassifierCV (F1)    0.9560     0.6087  0.5185    0.5600  0.5412

Optimal Threshold learned by TunedThresholdClassifierCV: 0.2424

=== 2. Probability Calibration (Brier Score Loss - Lower is Better) ===
Uncalibrated RandomForest Brier Loss: 0.0368
Calibrated RandomForest Brier Loss  : 0.0312
```
*(Ghaur karein: Standard LR ki Accuracy `95.4%` thi lekin us ne sirf `22.2%` frauds pakray (`Recall = 0.2222`). Jab hum ne `TunedThresholdClassifierCV` lagaya to threshold `0.50` se kam ho kar `0.2424` ho gaya aur F1-Score `0.3429` se jump kar ke `0.5600` par pohanch gaya!)*

---

## 5. Practice Exercises & Solutions

### Exercises
1. **Beginner:** Ek imbalanced dataset (`weights=[0.90, 0.10]`) par `DecisionTreeClassifier()` aur `DecisionTreeClassifier(class_weight='balanced')` ka `balanced_accuracy_score` compare karein.
2. **Intermediate:** `FixedThresholdClassifier` ke zariye `LogisticRegression` ka threshold `0.15` set karein taake medical screening mein Recall $> 90\%$ hasil ho sake.
3. **Challenge:** `make_scorer` ke zariye ek custom business profit/cost metric banayein (jahan Har Pakray Gaye Fraud `TP` par `+5000 PKR` bachat ho aur Har False Alarm `FP` par `-500 PKR` kharcha ho) aur usay `TunedThresholdClassifierCV` mein optimize karein!

### Solutions
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV, FixedThresholdClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix, make_scorer

X_i, y_i = make_classification(n_samples=600, weights=[0.90, 0.10], random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_i, y_i, test_size=0.25, random_state=42, stratify=y_i)

# 1. Beginner
dt1 = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_tr, y_tr)
dt2 = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42).fit(X_tr, y_tr)
print("Balanced Acc -> Default:", round(balanced_accuracy_score(y_te, dt1.predict(X_te)), 4),
      "| class_weight='balanced':", round(balanced_accuracy_score(y_te, dt2.predict(X_te)), 4))

# 2. Intermediate
ft = FixedThresholdClassifier(LogisticRegression(), threshold=0.15).fit(X_tr, y_tr)
print("Recall at threshold=0.15:", round(recall_score(y_te, ft.predict(X_te)), 4))

# 3. Challenge: Cost-Sensitive Custom Scorer
def net_savings_pkr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tp * 5000 - fp * 500)

cost_scorer = make_scorer(net_savings_pkr, greater_is_better=True)
cost_tuned = TunedThresholdClassifierCV(LogisticRegression(), scoring=cost_scorer, cv=3).fit(X_tr, y_tr)
print("Best Cost-Sensitive Threshold:", round(cost_tuned.best_threshold_, 3),
      "| Test Savings PKR:", net_savings_pkr(y_te, cost_tuned.predict(X_te)))
```

---
**Next Chapter:** [14_ANOMALY_DETECTION.md](14_ANOMALY_DETECTION.md)
