# Part 21: Model Inspection and Interpretability (`sklearn.inspection`, `sklearn.metrics`, `sklearn.utils`)

Jab ek bank ka model kisi customer ki Loan Application reject karta hai, ya hospital ka model kisi marij ko High-Risk batata hai, to sirf `"92% Accuracy"` kehna kafi nahi hota—humein yeh samjhana parta hai ke **model ne yeh faisla KYUN liya!** Is chapter mein hum [Scikit-learn Inspection & Interpretability](https://scikit-learn.org/stable/inspection.html) aur Visual Displays ke tamam 15 topics seekhenge.

---

## 1. Why Interpretability Matters & Correlation vs Predictive Importance vs Causation
1. **Trust & Regulation:** Doctors, bankers, aur regulators "Black Box" faislon par andha aitemaad nahi karte.
2. **Debugging Data Leakage & Bias:** Agar aap ke model mein sab se ahem feature `Patient_Hospital_ID` ya `Gender` aa raha hai, to interpretability tools foran parda faash kar dete hain ke model ghalat cheez seekh raha hai!
3. **Correlation vs Predictive Importance vs Causation:**
   - **Correlation:** Do variables ek sath hilte hain (`Barish` aur `Chhatri/Umbrella sales`).
   - **Predictive Importance:** Model ne prediction ke liye `Umbrella sales` ka kitna istemal kiya.
   - **Causation:** Kya chhatri bechne se barish hoti hai? **Hargiz nahi!** Feature importance sirf predictive association batata hai, **Causation (wajah)** sabit karne ke liye Randomized Controlled Trials (A/B tests) ya Causal Inference ki zaroorat hoti hai!

---

## 2. Core Scikit-learn Inspection & Display Tools

| Tool / Class (`sklearn.inspection` & `sklearn.metrics`) | How It Works & When to Use It |
| :--- | :--- |
| **1. Linear Coefficients (`model.coef_`)** | **Warning:** `coef_` ko tabhi compare karein jab tamam features **`StandardScaler`** se barabar scale kiye gaye hon! Warna `Area_Marla` ka weight bada aur `Income_Rupees` ka weight chota nazar aayega. |
| **2. Tree Impurity Importance (`model.feature_importances_`)** | Training data par Gini/MSE kitna kam hua. Fast hai lekin high-cardinality features ki taraf biased hota hai. |
| **3. `permutation_importance(model, X_test, y_test, n_repeats=10)`** | **Gold Standard!** Test set par ek feature ko random shuffle kar ke dekhta hai ke model ka metric kitna girta hai (`importances_mean`). Model-agnostic hai (har model par chalta hai)! |
| **4. `PartialDependenceDisplay.from_estimator(..., kind='average')` (PDP)** | Dikhata hai ke jab ek feature (jaise `Experience`) badalta hai to model ki **average prediction** kaise upar ya neeche jati hai (marginal effect). |
| **5. Individual Conditional Expectation (ICE, `kind='individual'` or `'both'`)** | PDP sirf average line dikhata hai (jo heterogeneous interactions ko chhupa sakti hai). **ICE** har single sample ki alag line dikhata hai! |
| **6. `ConfusionMatrixDisplay.from_estimator` / `.from_predictions`** | Confusion Matrix ka khoobsurat color-coded heatmap banata hai. |
| **7. `RocCurveDisplay` & 8. `PrecisionRecallDisplay`** | ROC Curve aur Precision-Recall Curve ko ek line mein plot karne ke liye. |
| **9. `PredictionErrorDisplay.from_estimator`** | **Regression Residual Plots!** `kind='residual_vs_predicted'` ya `'actual_vs_predicted'` bana kar heteroscedasticity aur outliers dikhata hai. |
| **10. `estimator_html_repr(pipeline)`** (`sklearn.utils.discovery` / `sklearn.utils`) | Jupyter Notebook mein complex `Pipeline` aur `ColumnTransformer` ka interactive diagram (HTML block) banata hai! |

---

## 3. Complete Runnable Code: Coefficients, Permutation Importance, PDP/ICE, Displays & Error Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay, PredictionErrorDisplay
)
from sklearn.utils import estimator_html_repr

# =====================================================================
# 1. Classification Interpretability & Permutation Importance
# =====================================================================
bc = load_breast_cancer(as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    bc.data.iloc[:, :8], bc.target, test_size=0.25, random_state=42, stratify=bc.target
)

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)

# Compare Built-in MDI vs Test-Set Permutation Importance
perm = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=42, scoring="roc_auc")
imp_df = pd.DataFrame({
    "Feature": X_tr.columns,
    "Tree_MDI_Train": rf.feature_importances_.round(4),
    "Permutation_AUC_Drop_Test": perm.importances_mean.round(4)
}).sort_values(by="Permutation_AUC_Drop_Test", ascending=False)

print("=== 1. Feature Importance Comparison (Train MDI vs Test Permutation) ===")
print(imp_df.to_string(index=False))

# =====================================================================
# 2. Error Analysis: Inspecting Worst False Positives / False Negatives
# =====================================================================
y_pred = rf.predict(X_te)
y_proba = rf.predict_proba(X_te)[:, 1]
errors_mask = (y_pred != y_te)
error_df = X_te[errors_mask].copy()
error_df["True_Label"] = y_te[errors_mask]
error_df["Pred_Label"] = y_pred[errors_mask]
error_df["Pred_Proba_1"] = y_proba[errors_mask].round(3)

print(f"\n=== 2. Error Analysis: Total Misclassified Test Samples = {len(error_df)} ===")
print(error_df[["mean radius", "mean concave points", "True_Label", "Pred_Label", "Pred_Proba_1"]].head(4))

# =====================================================================
# 3. Generate 4-Panel Diagnostic Plot (PDP+ICE, ROC, PR, Regression Residuals)
# =====================================================================
diab = load_diabetes(as_frame=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(diab.data, diab.target, test_size=0.2, random_state=42)
hgb_reg = HistGradientBoostingRegressor(random_state=42).fit(Xd_tr, yd_tr)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Panel A: Partial Dependence + ICE (kind="both") for 'bmi' feature
PartialDependenceDisplay.from_estimator(
    hgb_reg, Xd_te, features=["bmi"], kind="both", ax=axes[0, 0], random_state=42
)
axes[0, 0].set_title("1. PDP & ICE Curves for 'bmi'")

# Panel B: ROC Curve Display
RocCurveDisplay.from_estimator(rf, X_te, y_te, ax=axes[0, 1])
axes[0, 1].set_title("2. RocCurveDisplay")

# Panel C: Precision-Recall Curve Display
PrecisionRecallDisplay.from_estimator(rf, X_te, y_te, ax=axes[1, 0])
axes[1, 0].set_title("3. PrecisionRecallDisplay")

# Panel D: Regression Residuals Display
PredictionErrorDisplay.from_estimator(
    hgb_reg, Xd_te, yd_te, kind="residual_vs_predicted", ax=axes[1, 1]
)
axes[1, 1].set_title("4. PredictionErrorDisplay (Residuals)")

plt.tight_layout()
plt.savefig("model_inspection_displays.png", dpi=120)
plt.close()

# Verify estimator_html_repr length
html_str = estimator_html_repr(make_pipeline(StandardScaler(), Ridge()))
print(f"\nSaved 4-panel inspection chart & generated estimator_html_repr ({len(html_str)} chars).")
```

### Expected Output
```text
=== 1. Feature Importance Comparison (Train MDI vs Test Permutation) ===
                Feature  Tree_MDI_Train  Permutation_AUC_Drop_Test
    mean concave points          0.3541                     0.0612
         mean perimeter          0.1812                     0.0145
              mean area          0.1620                     0.0112
           mean texture          0.0895                     0.0098
         mean concavity          0.1045                     0.0041
            mean radius          0.0654                     0.0025
       mean compactness          0.0251                     0.0005
        mean smoothness          0.0182                     0.0001

=== 2. Error Analysis: Total Misclassified Test Samples = 10 ===
     mean radius  mean concave points  True_Label  Pred_Label  Pred_Proba_1
73         13.80              0.05069           0           1         0.680
297        11.76              0.04306           0           1         0.810
...
Saved 4-panel inspection chart & generated estimator_html_repr (11420 chars).
```

---

## 4. Practice Exercises & Solutions

### Exercises
1. **Beginner:** Kyun `LinearRegression` ke `coef_` ko bina `StandardScaler` lagaye feature importance ke taur par parhna ghalat hai?
2. **Intermediate:** `ConfusionMatrixDisplay.from_predictions(y_te, y_pred)` ke اندر mojood `.confusion_matrix` attribute ko print karein.
3. **Challenge:** Agar do features (`Area_SqFt` aur `Area_Marla`) aapas mein 99% correlated hon, to `permutation_importance` par kya asar parta hai aur uska ilaj kya hai?

### Solutions
```python
# 1. Beginner Answer:
# Kyunki agar Feature A Rupay mein (100,000) hai aur Feature B Rooms (1..5) mein, to Rupay ka coefficient chota (0.001) aur Rooms ka bara (25.0) aayega bhale hi Rupay zyada ahem ho!

# 2. Intermediate Solution:
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:\n", confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1]))

# 3. Challenge Answer:
# Jab 2 features 99% correlated hon, to ek feature ko shuffle karne par model doosre duplicate feature se kaam chala leta hai, jis se dono ka permutation importance chota aata hai! Ilaj: Pehle hierarchical clustering (FeatureAgglomeration ya Spearman correlation filter) se duplicate columns hatayein.
```

---
**Next Chapter:** [20_MODEL_PERSISTENCE_DEPLOYMENT.md](20_MODEL_PERSISTENCE_DEPLOYMENT.md)
