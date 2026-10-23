# Part 16: Anomaly and Novelty Detection (`sklearn.ensemble`, `sklearn.neighbors`, `sklearn.svm`, `sklearn.covariance`, `sklearn.linear_model`)

Bank fraud, server hacking (intrusion detection), ya factory ki machine mein achanak kharabi pakarne ke liye aksar hamare paas hazaron normal records to hote hain, lekin kharabi (anomaly) ke labels nahi hote. Is chapter mein hum [Scikit-learn Outlier and Novelty Detection](https://scikit-learn.org/stable/modules/outlier_detection.html) ke tamam 12 topics aur 5 core algorithms seekhenge.

---

## 1. Anomaly (Outlier) Detection vs Novelty Detection

Scikit-learn mein in dono terms mein ek bohot ahem farq hai:

| Concept | Training Data ($X_{\text{train}}$) Kaisa Hota Hai? | Maqsad Kya Hai? |
| :--- | :--- | :--- |
| **1. Outlier / Anomaly Detection** (Unsupervised) | Training data ke andar hi chand gande/unusual points (**Outliers**) pehle se mile hue (contaminated) hote hain. | Training data ke dense hisse ko samajhna aur door-daraaz points ko Outlier qarar dena. |
| **2. Novelty Detection** (Semi-Supervised) | Training data **100% Saaf (Pure Normal / Inliers)** hota hai—us mein koi outlier nahi hota! | Jab mustaqbil mein कोई naya test point (`X_new`) aaye, to check karna ke kya woh normal pattern se milta hai ya bilkul naya/ajeeb (**Novelty**) hai! |

### Scikit-learn Output Convention for All Outlier Detectors (`+1` vs `-1`)
Scikit-learn ke **tamam 5 outlier detectors** ka `.predict()` output bilkul ek jaisa hota hai:
- **`+1` (Inlier / Normal Sample)**
- **`-1` (Outlier / Anomaly / Novelty)**
Aur `.decision_function(X)` ya `.score_samples(X)` mein **positive/high score = Normal (`+1`)** aur **negative/low score = Anomaly (`-1`)** hota hai!

---

## 2. All 5 Scikit-learn Anomaly & Novelty Detection Algorithms

| Algorithm & Module | How It Works Intuitively | Best Use Case & Limitations |
| :--- | :--- | :--- |
| **1. `IsolationForest`** (`sklearn.ensemble`) | Random features aur random splits se trees banata hai. **Anomalies bohot kam splits (chote raste / short path length) mein akele (isolate) ho jati hain**, jabke normal bheer wale points ko alag karne mein bohot saare splits lagte hain! | **Best all-around choice for high-dimensional tabular data!** Fast, scalable, handles irrelevant features well. (`contamination='auto'` or float `0.05`). |
| **2. `LocalOutlierFactor` (LOF)** (`sklearn.neighbors`) | Har point ki local density ka uske `n_neighbors` ki density se mawazna karta hai. Agar kisi point ki density apne parosiyon se bohot kam ho to woh Outlier hai! | Variable-density clusters mein outliers pakarne ke liye best! **Important:** Default taur par sirf `.fit_predict(X)` chalta hai; naye test data par `.predict(X_test)` chalane ke liye **`novelty=True`** set karna LAZMI hai! |
| **3. `OneClassSVM`** (`sklearn.svm`) | Normal data ke gird Kernel (`'rbf'`) ki madad se ek tang ghera (boundary) khinchta hai. Parameter `nu` (default `0.5`) outliers ki upper bound fraction control karta hai. | **Novelty Detection** (jab training data saaf ho) aur non-linear boundaries ke liye acha hai; outliers in training data se sensitive hai. |
| **4. `EllipticEnvelope`** (`sklearn.covariance`) | Farz karta hai ke tamam normal data ek **Single Gaussian (Bell-curve Ellipse)** mein hai aur Minimum Covariance Determinant (MCD) fit karta hai. | Sirf tab use karein jab data Gaussian ho; multi-cluster ya non-Gaussian data par fail ho jata hai. |
| **5. `SGDOneClassSVM`** (`sklearn.linear_model`) | Stochastic Gradient Descent ke zariye Linear One-Class SVM ko bohot bare datasets (یا `RBFSampler` kernel approximation ke sath) tezi se fit karta hai (`partial_fit` supported). | Huge streaming datasets ke liye fast online anomaly detection. |

---

## 3. Complete Working Code: Comparing All 5 Detectors on Bank Transaction Fraud / Unusual Activity

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import f1_score, precision_score, recall_score

# 1. Create Synthetic Bank Transactions (285 Normal + 15 Unusual/Fraud = 300 total, 5% contamination)
rng = np.random.RandomState(42)
# Normal transactions: [Amount_Thousands_PKR, Hour_Of_Day, Tx_Count_Last_Hour]
X_normal = rng.normal(loc=[15.0, 14.0, 2.0], scale=[5.0, 3.0, 0.8], size=(285, 3))
# Anomalous/Fraud transactions: Midnight hours, huge amounts, rapid bursts
X_anom = rng.uniform(low=[80.0, 1.0, 10.0], high=[250.0, 4.0, 25.0], size=(15, 3))

X_all = np.vstack([X_normal, X_anom])
# True labels following sklearn convention: +1 for Normal, -1 for Anomaly
y_true = np.array([1] * 285 + [-1] * 15)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# 2. Initialize All 5 Anomaly Detectors (setting expected contamination = 0.05)
detectors = {
    "1. IsolationForest": IsolationForest(contamination=0.05, random_state=42),
    "2. LocalOutlierFactor (Outlier Mode)": LocalOutlierFactor(n_neighbors=20, contamination=0.05),
    "3. EllipticEnvelope (Gaussian)": EllipticEnvelope(contamination=0.05, random_state=42),
    "4. OneClassSVM (nu=0.05)": OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"),
    "5. SGDOneClassSVM": SGDOneClassSVM(nu=0.05, random_state=42)
}

rows = []
for name, det in detectors.items():
    # LOF in default outlier mode only has fit_predict()
    if isinstance(det, LocalOutlierFactor):
        y_pred = det.fit_predict(X_scaled)
    else:
        det.fit(X_scaled)
        y_pred = det.predict(X_scaled)

    # Evaluate Anomaly (-1) detection using pos_label=-1
    rows.append({
        "Detector": name,
        "Predicted_Anomalies": int((y_pred == -1).sum()),
        "Precision (-1)": round(precision_score(y_true, y_pred, pos_label=-1, zero_division=0), 4),
        "Recall (-1)": round(recall_score(y_true, y_pred, pos_label=-1), 4),
        "F1_Score (-1)": round(f1_score(y_true, y_pred, pos_label=-1), 4)
    })

print("=== 1. Unsupervised Anomaly Detection Comparison (pos_label = -1) ===")
print(pd.DataFrame(rows).to_string(index=False))

# 3. Novelty Detection Mode Demo: Fit LOF(novelty=True) ONLY on Clean Normal Data!
X_clean_train = scaler.fit_transform(X_normal[:200])
X_new_test = scaler.transform(np.vstack([X_normal[200:205], X_anom[:3]]))

lof_novelty = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05)
lof_novelty.fit(X_clean_train)
novelty_preds = lof_novelty.predict(X_new_test)
novelty_scores = lof_novelty.decision_function(X_new_test)

print("\n=== 2. Novelty Detection on 5 Normal + 3 Fraud New Transactions ===")
print("True Status     : [+1, +1, +1, +1, +1, -1, -1, -1]")
print("LOF Predictions :", list(novelty_preds))
print("Decision Scores :", np.round(novelty_scores, 2))
```

### Expected Output
```text
=== 1. Unsupervised Anomaly Detection Comparison (pos_label = -1) ===
                            Detector  Predicted_Anomalies  Precision (-1)  Recall (-1)  F1_Score (-1)
                  1. IsolationForest                   15          1.0000       1.0000         1.0000
2. LocalOutlierFactor (Outlier Mode)                   15          0.9333       0.9333         0.9333
      3. EllipticEnvelope (Gaussian)                   15          1.0000       1.0000         1.0000
            4. OneClassSVM (nu=0.05)                   18          0.7778       0.9333         0.8485
                   5. SGDOneClassSVM                   15          1.0000       1.0000         1.0000

=== 2. Novelty Detection on 5 Normal + 3 Fraud New Transactions ===
True Status     : [+1, +1, +1, +1, +1, -1, -1, -1]
LOF Predictions : [1, 1, 1, 1, 1, -1, -1, -1]
Decision Scores : [  0.41   0.38   0.45   0.32   0.4  -11.24 -14.85  -9.62]
```

---

## 4. Limitations of Unsupervised Anomaly Detection
1. **High False Positive Rate:** Har ghair-mamooli (unusual) transaction chori ya fraud nahi hoti! Maslan Eid ya shadi ke mahine mein ek customer agar achanak 5 Lakh ki shopping kare, to unsupervised anomaly detector usay `-1` (Anomaly) bolega.
2. **Sensitivity to `contamination`:** Agar aap ko andaza na ho ke data mein `0.1%` outliers hain ya `10%`, to threshold ghalat set ho sakta hai. Jab thore se bhi labeled anomalies mojood hon, to `decision_function` scores par Precision-Recall curve dekh kar threshold chunein.

---

## 5. Practice Exercises & Solutions

### Exercises
1. **Beginner:** `IsolationForest(contamination=0.1, random_state=42)` ko 100 samples par fit karein aur sab se kam `decision_function` score wale top 3 most extreme anomalies ke indices nikalen.
2. **Intermediate:** Kya hoga agar aap `LocalOutlierFactor()` (bina `novelty=True` ke) par `.predict(X_new)` call karein? Try karein aur uska hal likhein.
3. **Challenge:** `OneClassSVM(kernel='rbf', nu=0.05)` ko clean training data par fit kar ke test data par evaluate karein.

### Solutions
```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

rng = np.random.RandomState(42)
X_s = np.vstack([rng.normal(0, 1, (95, 2)), rng.uniform(6, 10, (5, 2))])

# 1. Beginner
iso = IsolationForest(contamination=0.05, random_state=42).fit(X_s)
scores = iso.decision_function(X_s)
worst_3_idx = np.argsort(scores)[:3]
print("Top 3 Most Extreme Anomaly Indices:", worst_3_idx, "| Scores:", scores[worst_3_idx].round(3))

# 2. Intermediate
lof_nov = LocalOutlierFactor(novelty=True).fit(X_s[:80])
print("LOF(novelty=True) prediction on new point [8, 8]:", lof_nov.predict([[8.0, 8.0]]))

# 3. Challenge
ocsvm = OneClassSVM(kernel="rbf", nu=0.05).fit(X_s[:90])
print("OneClassSVM on [0, 0] (Normal) vs [8, 8] (Anomaly):", ocsvm.predict([[0.0, 0.0], [8.0, 8.0]]))
```

---
**Next Chapter:** [15_SEMI_SUPERVISED_LEARNING.md](15_SEMI_SUPERVISED_LEARNING.md)
