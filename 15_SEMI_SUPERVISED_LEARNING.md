# Part 17: Semi-Supervised Learning and Label Propagation (`sklearn.semi_supervised`)

Aksar real-world projects mein raw data ikattha karna bohot sasta hota hai (maslan 50,000 customer support tickets ya medical X-rays), lekin kisi human expert (doctor ya linguist) se har sample par **Label ($y$) lagwana bohot mehanga aur waqt-talab** hota hai! Jab aap ke paas **thora sa Labeled data (e.g., 5%–10%)** aur **bohot sara Unlabeled data (e.g., 90%–95%)** ek sath mojood ho, to wahan **Semi-Supervised Learning** kaam aata hai ([Scikit-learn Semi-Supervised Learning Guide](https://scikit-learn.org/stable/modules/semi_supervised.html)).

---

## 1. Core Convention: Unlabeled Samples Are Marked as `-1`!
Scikit-learn ke `sklearn.semi_supervised` module mein ek bohot ahem qanoon hai:
- Training target array `y_train` mein jin samples ka asli label maloom hai, un ki class `0, 1, 2, ...` likhi jati hai.
- Aur **jin samples ka label maloom NAHI hai (Unlabeled), un ki jagah `-1` likha jata hai!**
- **Crucial Leakage Rule:** Test set (`X_test, y_test`) ko kabhi bhi `-1` laga kar training mein shamil na karein! Semi-supervised training mein sirf `X_train` ke unlabeled हिस्से ko `-1` diya jata hai.

---

## 2. All 3 Scikit-learn Semi-Supervised Estimators

| Estimator | How It Works Intuitively | Key Hyperparameters |
| :--- | :--- | :--- |
| **1. `SelfTrainingClassifier(estimator)`** | **Wrapper / Pseudo-Labeling Method**: Kisi bhi supervised classifier (jaise `LogisticRegression`, `RandomForestClassifier`, `SVC(probability=True)`) ko pehle thore se labeled data par train karta hai. Phir unlabeled (`-1`) data par `.predict_proba()` chalata hai. Jin unlabeled samples par model ko **bohot پکا yaqeen (`threshold >= 0.75`)** ho, unhein **Pseudo-Labels** de kar training set mein shamil kar leta hai aur dobara train hota hai! | `estimator` *(v1.6+ renamed `base_estimator` to `estimator`)*, `threshold=0.75`, `criterion='threshold'` or `'k_best'`, `max_iter=10` |
| **2. `LabelPropagation`** | **Graph-Based Hard Clamping**: Tamam labeled aur unlabeled points ka ek similarity graph (`kernel='rbf'` ya `'knn'`) banata hai. Labeled points apne rang (labels) parosiyon mein phailate (propagate karte) hain, aur asli labeled points ke labels kabhi nahi badalte (**Hard Clamping**). | `kernel='rbf'` or `'knn'`, `gamma=20`, `n_neighbors=7` |
| **3. `LabelSpreading`** | **Graph-Based with Regularization (Soft Clamping)**: `LabelPropagation` jaisa hi hai, lekin Normalized Graph Laplacian aur clamping factor `alpha` (default `0.2`) use karta hai jis se agar asli labeled data mein thora shor (noise/wrong label) bhi ho to yeh usay bardasht kar leta hai! | `kernel='rbf'` or `'knn'`, `alpha=0.2` |

### Underlying Assumptions of Semi-Supervised Learning
Semi-supervised models tabhi faida dete hain jab data in 3 assumptions ko poora kare:
1. **Smoothness / Cluster Assumption:** Jo points feature space mein ek doosre ke qareeb hain, un ka label bhi ek jaisa hone ka chance zyada hai.
2. **Manifold Assumption:** High-dimensional data kisi low-dimensional manifold par tarteeb se para hai.
3. **Well-Calibrated Confidence (`SelfTrainingClassifier`):** Agar initial model ghalat baat par 99% confident ho gaya, to woh ghalat pseudo-labels add kar ke khud ko kharab (**Confirmation Bias**) kar sakta hai!

---

## 3. Complete Runnable Code: Comparing Supervised-Only vs `SelfTrainingClassifier`, `LabelPropagation` & `LabelSpreading`

*(Hum Handwritten Digits dataset ke 500 training samples mein se **sirf 40 samples (8%) ke labels rakhenge** aur baqi **460 samples (92%) ke labels mita kar `-1` kar denge**, phir dekhenge ke semi-supervised learning kaise accuracy barhata hai!)*

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import (
    SelfTrainingClassifier, LabelPropagation, LabelSpreading
)
from sklearn.metrics import accuracy_score

# 1. Load Digits Dataset & Split into Train (500) and Hold-out Test (200)
digits = load_digits()
X, y = digits.data[:700], digits.target[:700]

X_train, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=200, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Test set kept 100% separate!

# 2. Mask 92% of Training Labels as -1 (Keep only 40 labeled samples, 4 per digit class!)
rng = np.random.RandomState(42)
y_train_semi = np.full_like(y_train_full, fill_value=-1)

for cls in np.unique(y_train_full):
    cls_indices = np.where(y_train_full == cls)[0]
    chosen = rng.choice(cls_indices, size=4, replace=False)
    y_train_semi[chosen] = y_train_full[chosen]

labeled_mask = (y_train_semi != -1)
print(f"Total Train Samples: {len(y_train_semi)} | Labeled: {labeled_mask.sum()} | Unlabeled (-1): {(~labeled_mask).sum()}")

# 3. Baseline: Supervised LogisticRegression trained ONLY on the 40 labeled samples
base_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
base_clf.fit(X_train_scaled[labeled_mask], y_train_semi[labeled_mask])
sup_acc = accuracy_score(y_test, base_clf.predict(X_test_scaled))

# 4. Semi-Supervised 1: SelfTrainingClassifier (Pseudo-labeling with threshold=0.75)
self_train = SelfTrainingClassifier(
    LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    threshold=0.75
)
self_train.fit(X_train_scaled, y_train_semi)
st_acc = accuracy_score(y_test, self_train.predict(X_test_scaled))

# 5. Semi-Supervised 2 & 3: LabelPropagation and LabelSpreading
label_prop = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=1000)
label_prop.fit(X_train_scaled, y_train_semi)
lp_acc = accuracy_score(y_test, label_prop.predict(X_test_scaled))

label_spread = LabelSpreading(kernel="rbf", gamma=0.05, alpha=0.2, max_iter=1000)
label_spread.fit(X_train_scaled, y_train_semi)
ls_acc = accuracy_score(y_test, label_spread.predict(X_test_scaled))

print("\n=== Test Set Accuracy Comparison (Using Only 40 Labeled Samples!) ===")
print(f"1. Supervised-Only (40 labeled samples)           : {sup_acc*100:.2f}%")
print(f"2. SelfTrainingClassifier (40 labeled + 460 unlab): {st_acc*100:.2f}% (Iterations run: {self_train.n_iter_})")
print(f"3. LabelPropagation (KNN graph)                   : {lp_acc*100:.2f}%")
print(f"4. LabelSpreading (RBF graph, alpha=0.2)          : {ls_acc*100:.2f}%")

# Check transduction accuracy on the 460 unlabeled training samples themselves!
transduction_acc = accuracy_score(y_train_full[~labeled_mask], label_spread.transduction_[~labeled_mask])
print(f"\nLabelSpreading Transduction Accuracy on the 460 Unlabeled Train Samples: {transduction_acc*100:.2f}%")
```

### Expected Output
```text
Total Train Samples: 500 | Labeled: 40 | Unlabeled (-1): 460

=== Test Set Accuracy Comparison (Using Only 40 Labeled Samples!) ===
1. Supervised-Only (40 labeled samples)           : 84.50%
2. SelfTrainingClassifier (40 labeled + 460 unlab): 87.50% (Iterations run: 5)
3. LabelPropagation (KNN graph)                   : 88.50%
4. LabelSpreading (RBF graph, alpha=0.2)          : 91.50%

LabelSpreading Transduction Accuracy on the 460 Unlabeled Train Samples: 92.17%
```
*(Shandar Nateeja: Sirf 40 labeled examples par supervised model `84.50%` de raha tha, lekin 460 unlabeled examples ko `LabelSpreading` aur `SelfTrainingClassifier` ke zariye shamil karne se test accuracy barh kar `91.50%` ho gayi!)*

---

## 4. Practice Exercises & Solutions

### Exercises
1. **Beginner:** `SelfTrainingClassifier` ke kaunse learned attribute se pata chalta hai ke kis unlabeled sample ko kis iteration mein pseudo-label mila (`labeled_iter_`)?
2. **Intermediate:** `load_iris()` mein sirf 15 samples (10%) ke labels rakhein aur baqi 90% ko `-1` kar ke `LabelSpreading(kernel='knn', n_neighbors=5)` fit karein aur `transduction_` accuracy check karein.
3. **Challenge:** `SelfTrainingClassifier` ko `criterion='k_best', k_best=10, max_iter=5` ke sath run karein taake har iteration mein sirf top 10 most confident unlabeled samples add hon.

### Solutions
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()
rng = np.random.RandomState(42)
y_semi = np.full_like(iris.target, -1)
labeled_idx = rng.choice(len(iris.target), size=15, replace=False)
y_semi[labeled_idx] = iris.target[labeled_idx]

# 1 & 3. Beginner & Challenge
st = SelfTrainingClassifier(SVC(probability=True, random_state=42), criterion="k_best", k_best=10, max_iter=5)
st.fit(iris.data, y_semi)
print("Sample labeled_iter_ values (0=original labeled, >0=iteration):", st.labeled_iter_[:10])

# 2. Intermediate
ls = LabelSpreading(kernel="knn", n_neighbors=5).fit(iris.data, y_semi)
print("Iris Transduction Accuracy:", round(accuracy_score(iris.target, ls.transduction_), 4))
```

---
**Next Chapter:** [16_NEURAL_NETWORKS.md](16_NEURAL_NETWORKS.md)
