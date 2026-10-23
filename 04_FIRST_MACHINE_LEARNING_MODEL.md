# Part 6: Your First Complete Machine Learning Model (End-to-End Iris Project)

Ab waqt aa gaya hai ke hum ab tak seekhe gaye tamam concepts ko mila kar apna **pehla mukammal, leakage-free Machine Learning project** step-by-step banayein! Hum classic **Iris Flower Classification** dataset ([`sklearn.datasets.load_iris`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)) istemal karenge aur tamam 15 zaroori marahil (steps) ko cover karenge.

---

## 1. Understanding the Problem & Dataset
- **Goal:** Phool (flower) ki pattiyon (petals aur sepals) ki lambai aur chorai (length & width in cm) dekh kar pehchanna ke yeh Iris ki kaunsi qism (species) hai:
  - Class `0`: **Setosa**
  - Class `1`: **Versicolor**
  - Class `2`: **Virginica**
- **Features ($X$, 4 Numerical Columns):**
  1. `sepal length (cm)`
  2. `sepal width (cm)`
  3. `petal length (cm)`
  4. `petal width (cm)`
- **Problem Type:** **3-Class (Multiclass) Supervised Classification** (`n_samples = 150`, har class ke 50 samples).

---

## 2. Complete End-to-End Working Code (All 15 Steps)

```python
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# STEP 1 & 2: Load Built-in Dataset & Understand Features/Labels
# ==========================================
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
class_names = iris.target_names

print("=== 1. Dataset Overview ===")
print(f"Features Shape (X): {X.shape}")
print(f"Target Shape (y)  : {y.shape}")
print(f"Feature Names     : {list(X.columns)}")
print(f"Target Classes    : {dict(enumerate(class_names))}")

# ==========================================
# STEP 3: Inspect Samples & Class Balance
# ==========================================
print("\n=== 2. First 3 Samples ===")
print(X.head(3))
print("\nClass Counts:\n", y.value_counts())

# ==========================================
# STEP 4: Split Training and Test Data (Stratified)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nSplit Sizes -> Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ==========================================
# STEP 5, 6 & 7: Choose Algorithm, Create Pipeline & Train with fit()
# ==========================================
# KNN distance-based hai, is liye Pipeline mein StandardScaler lagana best practice hai
baseline_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

baseline_pipe.fit(X_train, y_train)

# ==========================================
# STEP 8 & 9: Make Predictions & Compare with Actual Labels
# ==========================================
y_pred = baseline_pipe.predict(X_test)

comparison_df = pd.DataFrame({
    "Actual_Code": y_test.values[:6],
    "Predicted_Code": y_pred[:6],
    "Actual_Name": [class_names[i] for i in y_test.values[:6]],
    "Predicted_Name": [class_names[i] for i in y_pred[:6]]
})
print("\n=== 3. First 6 Predictions vs Actual ===")
print(comparison_df)

# ==========================================
# STEP 10 & 11: Evaluate Accuracy & Beyond Single Accuracy Score
# ==========================================
acc = accuracy_score(y_test, y_pred)
print(f"\nBaseline Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nDetailed Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

# ==========================================
# STEP 12 & 13: Test an Individual New Example & Map Class Label
# ==========================================
# Ek naya phool: sepal_len=5.1, sepal_wid=3.5, petal_len=1.4, petal_wid=0.2
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
pred_code = baseline_pipe.predict(new_flower)[0]
pred_proba = baseline_pipe.predict_proba(new_flower)[0]
print(f"Single Flower Prediction -> Code: {pred_code} ({class_names[pred_code]}) | Probabilities: {pred_proba}")

# ==========================================
# STEP 14: Improve the Model WITHOUT Test-Set Leakage (GridSearchCV on Train)
# ==========================================
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"]
}
grid_search = GridSearchCV(baseline_pipe, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)  # Notice: ONLY X_train and y_train used!

best_model = grid_search.best_estimator_
print("\n=== 4. Model Improvement via 5-Fold Cross-Validation (On Train Set Only) ===")
print("Best Hyperparameters:", grid_search.best_params_)
print(f"Best CV Accuracy (Train): {grid_search.best_score_:.4f}")
print(f"Final Test Set Accuracy : {best_model.score(X_test, y_test):.4f}")

# ==========================================
# STEP 15: Save and Load the Trained Pipeline with joblib
# ==========================================
model_file = Path("iris_knn_pipeline.joblib")
joblib.dump(best_model, model_file)

loaded_model = joblib.load(model_file)
verify_pred = loaded_model.predict(new_flower)[0]
print(f"\nSaved & Reloaded Model Verification -> Predicted: {class_names[verify_pred]}")

# Cleanup temporary model file
if model_file.exists():
    model_file.unlink()
```

---

## 3. Detailed Step-by-Step Explanation in Roman Urdu

1. **Why Accuracy Alone Can Be Misleading (Step 11):**
   - Sirf ek `accuracy_score` (maslan `96.67%`) yeh nahi batata ke model ne kis class mein ghalti ki. Is liye hum ne **`confusion_matrix`** aur **`classification_report`** (Precision, Recall, F1-score) print kiya taake har phool ki qism (`setosa`, `versicolor`, `virginica`) ka alag hisab nazar aaye.
2. **Testing a Single Sample (Step 12):**
   - Jab hum ek single flower test karte hain, tab bhi usay **2D shape `(1, 4)`** mein dena zaroori hai (`[[5.1, 3.5, 1.4, 0.2]]`). Aur kyunki hum ne training mein Pandas DataFrame diya tha, is liye single sample ko bhi `pd.DataFrame(..., columns=X.columns)` bana kar dena best practice hai taake `UserWarning: X does not have valid feature names` na aaye!
3. **Improving the Model Without Test-Set Leakage (Step 14):**
   - Aksar beginners `X_test` par bar-bar `n_neighbors` badal kar dekhte hain ke kis value par test accuracy sab se zyada aati hai. **Yeh Test-Set Leakage hai!**
   - Sahi tareeqa yeh hai ke hum `GridSearchCV(cv=5)` ko **sirf `X_train, y_train`** par chalayein. Jab best hyperparameter चुन liya jaye, tab aakhir mein sirf ek baar `best_model.score(X_test, y_test)` check karein.
4. **Saving the Entire Pipeline (Step 15):**
   - Hum ne sirf `knn` model save nahi kiya, balke poora `Pipeline` (`scaler` + `knn`) `joblib.dump()` se save kiya. Is ka faida yeh hai ke jab naya flower aayega, to `loaded_model.predict(new_flower)` khud hi pehle usay scale karega aur phir predict karega!

---

## 4. Expected Output
```text
=== 1. Dataset Overview ===
Features Shape (X): (150, 4)
Target Shape (y)  : (150,)
Feature Names     : ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Target Classes    : {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

=== 2. First 3 Samples ===
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2

Split Sizes -> Train: 120 samples | Test: 30 samples

=== 3. First 6 Predictions vs Actual ===
   Actual_Code  Predicted_Code Actual_Name Predicted_Name
0            0               0      setosa         setosa
1            2               2   virginica      virginica
2            1               1  versicolor     versicolor
3            1               1  versicolor     versicolor
4            0               0      setosa         setosa
5            1               1  versicolor     versicolor

Baseline Test Accuracy: 0.9333 (93.33%)

Confusion Matrix:
 [[10  0  0]
 [ 0 10  0]
 [ 0  2  8]]

Single Flower Prediction -> Code: 0 (setosa) | Probabilities: [1. 0. 0.]

=== 4. Model Improvement via 5-Fold Cross-Validation (On Train Set Only) ===
Best Hyperparameters: {'knn__n_neighbors': 5, 'knn__weights': 'distance'}
Best CV Accuracy (Train): 0.9667
Final Test Set Accuracy : 0.9667

Saved & Reloaded Model Verification -> Predicted: setosa
```

---

## 5. Practice Exercises & Solutions

### Exercises
1. **Beginner:** Upar wale code mein `KNeighborsClassifier` ki jagah `from sklearn.tree import DecisionTreeClassifier(max_depth=3, random_state=42)` lagayein aur test accuracy print karein.
2. **Intermediate:** Ek naye phool `[6.7, 3.0, 5.2, 2.3]` ke liye predicted class ka naam (`setosa`, `versicolor`, ya `virginica`) aur uski probability print karein.
3. **Challenge:** `load_wine(as_frame=True)` dataset par bilkul isi tarah `StandardScaler` + `LogisticRegression` ka pipeline banayein, `test_size=0.2` rakhein, aur test accuracy nikalen.

### Solutions
```python
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# 1 & 2. Beginner & Intermediate Solution
iris = load_iris(as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
print("DecisionTree Test Accuracy:", round(dt.score(X_te, y_te), 4))

sample = pd.DataFrame([[6.7, 3.0, 5.2, 2.3]], columns=iris.data.columns)
pred_idx = dt.predict(sample)[0]
print("Predicted Species:", iris.target_names[pred_idx], "| Proba:", dt.predict_proba(sample)[0])

# 3. Challenge Solution
wine = load_wine(as_frame=True)
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target)
wine_pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=42)).fit(Xw_tr, yw_tr)
print("Wine Pipeline Test Accuracy:", round(wine_pipe.score(Xw_te, yw_te), 4))
```

---
**Next Chapter:** [05_CLASSIFICATION.md](05_CLASSIFICATION.md)
