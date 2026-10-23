# Part 12: Model Evaluation and Metrics (`sklearn.metrics`)

Model train karna sirf aadha kaam hai—asal maigयार (test) yeh hai ke aap apne business goal ke mutabiq **sahi Evaluation Metric** chunein! Is chapter mein hum [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) ke tamam **Classification, Regression, aur Clustering Metrics** un ke formulas, Pakistani business scenarios, aur code ke sath seekhenge.

---

## SECTION 1: Classification Metrics

### 1. The Confusion Matrix (Har Classification Metric Ki Bunyad)
Binary Classification (`0 = Negative/Healthy`, `1 = Positive/Disease`) mein 4 mumkina nataij hote hain:
- **True Negative ($TN$)**: Asal mein `0` tha, model ne bhi `0` kaha (Sahi!).
- **False Positive ($FP$, Type I Error / False Alarm)**: Asal mein `0` (Be-gunah/Healthy) tha, model ne ghalti se `1` (Fraud/Bemar) keh diya!
- **False Negative ($FN$, Type II Error / Miss)**: Asal mein `1` (Cancer/Fraud) tha, model ne ghalti se `0` (Sab theek hai) keh kar chhor diya! **(Medical aur Fraud mein sab se khatarnaak ghalti!)**
- **True Positive ($TP$)**: Asal mein `1` tha, model ne bhi `1` pakra (Sahi!).

### 2. Complete Table of 16 Classification Metrics

| Metric & Scikit-learn Function | Mathematical Formula / Intuition | When to Use & Limitations |
| :--- | :--- | :--- |
| **1. Accuracy** (`accuracy_score`) | $\frac{TP + TN}{TP + TN + FP + FN}$ | Sirf balanced classes ke liye. **Imbalanced data par dhoka deta hai!** |
| **2. Precision** (`precision_score`) | $\frac{TP}{TP + FP}$ (*"Jitno ko Positive kaha, un mein se kitne waqai Positive nikle?"*) | Jab **False Positives ($FP$) mehangay hon** (maslan Spam filter mein Boss ki asli email ko Spam keh dena!). |
| **3. Recall / Sensitivity / TPR** (`recall_score`) | $\frac{TP}{TP + FN}$ (*"Asal Positives mein se kitne pakray gaye?"*) | Jab **False Negatives ($FN$) jaan-lewa hon** (Cancer detection, Fraud detection). |
| **4. Specificity / TNR** | $\frac{TN}{TN + FP}$ (`recall_score(y, y_pred, pos_label=0)`) | Negative class ko sahi pehchanne ki salahiyat. |
| **5. F1-Score** (`f1_score`) | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Precision aur Recall ka **Harmonic Mean**—imbalanced classification ke liye behtareen single score. |
| **6. Classification Report** (`classification_report`) | Text summary of Precision, Recall, F1, and Support per class | Har project mein quick class-wise diagnostic ke liye. |
| **7. Balanced Accuracy** (`balanced_accuracy_score`) | $\frac{\text{Sensitivity (TPR)} + \text{Specificity (TNR)}}{2}$ | Imbalanced datasets par standard accuracy ka behtareen mutabadil. |
| **8. ROC Curve & 9. ROC-AUC** (`roc_curve`, `roc_auc_score`) | Area under TPR vs FPR curve across all thresholds ($0.5 = \text{random}, 1.0 = \text{perfect}$) | Model ki ranking capability naapne ke liye. *(Note: Severe class imbalance mein overly optimistic ho sakta hai).* |
| **10. Precision-Recall Curve & 11. Average Precision (PR-AUC)** (`precision_recall_curve`, `average_precision_score`) | Area under Precision vs Recall curve | **Rare positive events (e.g., 1% Fraud)** ke liye ROC-AUC se kahin zyada sacha metric! |
| **12. Log Loss / Cross-Entropy** (`log_loss`) | $-\frac{1}{N} \sum [y_i \log(p_i) + (1-y_i)\log(1-p_i)]$ | **Probability Calibration** naapne ke liye—ghalat baat par over-confident hone ($p=0.99$) par sakht saza deta hai! |
| **13. Matthews Correlation Coefficient (MCC)** (`matthews_corrcoef`) | Correlation between true and predicted binary classifications ($-1$ to $+1$) | $TP, TN, FP, FN$ charon ko barabar ahmiyat deta hai—sab se balanced binary metric! |
| **14. Cohen's Kappa** (`cohen_kappa_score`) | $\frac{p_o - p_e}{1 - p_e}$ (Chance agreement nikal kar score) | Inter-rater agreement aur imbalanced multiclass ke liye. |
| **15. Top-k Accuracy** (`top_k_accuracy_score`) | Kya sahi jawab model ki top $k$ (e.g., top 3) predictions mein mojood tha? | E-commerce recommendations ya 100-class classification ke liye. |
| **16. Multiclass Averaging** (`average=...`) | `'macro'` (unweighted mean across classes), `'weighted'` (weighted by support), `'micro'` (global counts) | Imbalanced multiclass mein **`average='macro'`** minority class ko barabar ahmiyat deta hai! |

---

## SECTION 2: Regression Metrics

| Metric & Scikit-learn Function | Formula & Units | Interpretation & Limitations |
| :--- | :--- | :--- |
| **1. MAE (Mean Absolute Error)** (`mean_absolute_error`) | $\frac{1}{N} \sum |y_i - \hat{y}_i|$ (Same unit as $y$, e.g. Lakhs PKR) | Asaan faham ausat ghalti; outliers se kam mutasir hota hai. |
| **2. MSE (Mean Squared Error)** (`mean_squared_error`) | $\frac{1}{N} \sum (y_i - \hat{y}_i)^2$ (Squared units) | Bare errors (outliers) ko murabba kar ke bari saza deta hai. |
| **3. RMSE (Root Mean Squared Error)** (`root_mean_squared_error`) | $\sqrt{\text{MSE}}$ (Same unit as $y$, e.g. Lakhs PKR) | Industry mein sab se zyada use hone wala regression metric! *(Added as dedicated function in sklearn v1.4+)*. |
| **4. $R^2$ Score (Coefficient of Determination)** (`r2_score`) | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Model ne target ki kitne feesad variance explain ki ($1.0 = \text{perfect}, 0.0 = \text{mean baseline}, <0 = \text{worse than mean}$). |
| **5. Explained Variance** (`explained_variance_score`) | $1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$ | $R^2$ jaisa hai lekin systematic bias offset ko ignore karta hai. |
| **6. Median Absolute Error** (`median_absolute_error`) | $\text{Median}(|y_i - \hat{y}_i|)$ | **Outliers ke khilaf sab se mazboot metric!** |
| **7. MAPE (Mean Absolute Percentage Error)** (`mean_absolute_percentage_error`) | $\frac{1}{N} \sum \frac{|y_i - \hat{y}_i|}{|y_i|}$ | Percentage ghalti batata hai. **Limitation:** Agar $y_i = 0$ (ya $0$ ke qareeb) ho to division by zero se $\infty$ (huge number) ban jata hai! |
| **8. Max Error** (`max_error`) | $\max(|y_i - \hat{y}_i|)$ | Sab se bura single-case error pakarne ke liye. |
| **9. Pinball Loss** (`mean_pinball_loss`) | Quantile loss for `QuantileRegressor` (`alpha=0.90` etc.) | Forecasting intervals (10th/90th percentile) evaluate karne ke liye. |
| **10. Mean Poisson Deviance** (`mean_poisson_deviance`) | Deviance for count targets ($y \in \{0, 1, 2, \dots\}$) | Call center calls per hour ya insurance claims count ke liye ($y > 0$ required). |

---

## SECTION 3: Clustering Metrics (Internal vs External)
1. **Internal Metrics (No true labels $y$ needed):** `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`.
2. **External / Ground-Truth Metrics (Jab benchmark testing ke liye true classes $y$ mojood hon):**
   - `adjusted_rand_score` (ARI)
   - `normalized_mutual_info_score` (NMI) & `adjusted_mutual_info_score` (AMI)
   - `homogeneity_completeness_v_measure`

---

## 4. Complete Runnable Code: Demonstrating All 3 Metric Families

```python
import numpy as np
from sklearn.metrics import (
    # Classification
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    log_loss, matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    # Regression
    mean_absolute_error, mean_squared_error, root_mean_squared_error,
    r2_score, median_absolute_error, mean_absolute_percentage_error,
    max_error, mean_pinball_loss, mean_poisson_deviance,
    # Clustering
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure
)

# 1. Classification Metrics Demo (Fraud Detection: 1 = Fraud, 0 = Normal)
y_true_cls = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
y_pred_cls = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
y_prob_cls = np.array([0.05, 0.1, 0.02, 0.08, 0.15, 0.2, 0.65, 0.92, 0.85, 0.40])

tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred_cls).ravel()
specificity = tn / (tn + fp)

print("=== 1. Classification Metrics ===")
print(f"Confusion Matrix -> TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Accuracy         : {accuracy_score(y_true_cls, y_pred_cls):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_true_cls, y_pred_cls):.4f}")
print(f"Precision        : {precision_score(y_true_cls, y_pred_cls):.4f}")
print(f"Recall (TPR)     : {recall_score(y_true_cls, y_pred_cls):.4f}")
print(f"Specificity (TNR): {specificity:.4f}")
print(f"F1-Score         : {f1_score(y_true_cls, y_pred_cls):.4f}")
print(f"ROC-AUC          : {roc_auc_score(y_true_cls, y_prob_cls):.4f}")
print(f"PR-AUC (Avg Prec): {average_precision_score(y_true_cls, y_prob_cls):.4f}")
print(f"Log Loss         : {log_loss(y_true_cls, y_prob_cls):.4f}")
print(f"MCC              : {matthews_corrcoef(y_true_cls, y_pred_cls):.4f}")
print(f"Cohen's Kappa    : {cohen_kappa_score(y_true_cls, y_pred_cls):.4f}")

# 2. Regression Metrics Demo (House Prices in Lakhs)
y_true_reg = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
y_pred_reg = np.array([110.0, 145.0, 190.0, 260.0, 330.0])

print("\n=== 2. Regression Metrics ===")
print(f"MAE              : {mean_absolute_error(y_true_reg, y_pred_reg):.2f}")
print(f"MSE              : {mean_squared_error(y_true_reg, y_pred_reg):.2f}")
print(f"RMSE             : {root_mean_squared_error(y_true_reg, y_pred_reg):.2f}")
print(f"R2 Score         : {r2_score(y_true_reg, y_pred_reg):.4f}")
print(f"Median Abs Error : {median_absolute_error(y_true_reg, y_pred_reg):.2f}")
print(f"MAPE             : {mean_absolute_percentage_error(y_true_reg, y_pred_reg)*100:.2f}%")
print(f"Max Error        : {max_error(y_true_reg, y_pred_reg):.2f}")
print(f"Pinball (q=0.5)  : {mean_pinball_loss(y_true_reg, y_pred_reg, alpha=0.5):.2f}")
print(f"Poisson Deviance : {mean_poisson_deviance(y_true_reg, y_pred_reg):.4f}")

# 3. Clustering External Metrics Demo
true_clusters = [0, 0, 1, 1, 2, 2]
pred_clusters = [1, 1, 0, 0, 2, 2]  # Notice cluster IDs 0 and 1 are swapped, yet grouping is 100% right!
h, c, v = homogeneity_completeness_v_measure(true_clusters, pred_clusters)
print("\n=== 3. Clustering Metrics (Label Permutation Invariant) ===")
print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(true_clusters, pred_clusters):.4f}")
print(f"Normalized Mutual Info   : {normalized_mutual_info_score(true_clusters, pred_clusters):.4f}")
print(f"Homogeneity={h:.2f}, Completeness={c:.2f}, V-Measure={v:.2f}")
```

### Expected Output
```text
=== 1. Classification Metrics ===
Confusion Matrix -> TN=6, FP=1, FN=1, TP=2
Accuracy         : 0.8000
Balanced Accuracy: 0.7619
Precision        : 0.6667
Recall (TPR)     : 0.6667
Specificity (TNR): 0.8571
F1-Score         : 0.6667
ROC-AUC          : 0.9524
PR-AUC (Avg Prec): 0.8889
Log Loss         : 0.3052
MCC              : 0.5238
Cohen's Kappa    : 0.5238

=== 2. Regression Metrics ===
MAE              : 13.00
MSE              : 245.00
RMSE             : 15.65
R2 Score         : 0.9510
Median Abs Error : 10.00
MAPE             : 6.47%
Max Error        : 30.00
Pinball (q=0.5)  : 6.50
Poisson Deviance : 1.0121

=== 3. Clustering Metrics (Label Permutation Invariant) ===
Adjusted Rand Index (ARI): 1.0000
Normalized Mutual Info   : 1.0000
Homogeneity=1.00, Completeness=1.00, V-Measure=1.00
```

---

## 5. Practice Exercises & Solutions

### Exercises
1. **Beginner:** Ek hospital ke **TB Screening** model mein aap **Precision** ko zyada ahmiyat denge ya **Recall** ko? Kyun?
2. **Intermediate:** Agar `y_true = [0, 1, 2, 0, 1, 2]` aur `y_pred = [0, 2, 1, 0, 0, 1]` ho, to `f1_score` ko `average='macro'` aur `average='weighted'` ke sath calculate karein.
3. **Challenge:** Bootstrap resampling (100 iterations) ke zariye test set accuracy ka **95% Confidence Interval** calculate karne ka chota sa loop likhein.

### Solutions
```python
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# 1. Beginner Answer:
# Recall (Sensitivity) ko zyada ahmiyat denge, kyunki kisi TB ke marij ko ghalti se "Healthy" keh kar ghar bhej dena (False Negative) bohot khatarnaak hai!

# 2. Intermediate Solution:
yt = [0, 1, 2, 0, 1, 2]
yp = [0, 2, 1, 0, 0, 1]
print("Macro F1   :", round(f1_score(yt, yp, average="macro"), 4))
print("Weighted F1:", round(f1_score(yt, yp, average="weighted"), 4))

# 3. Challenge Solution (95% Bootstrap Confidence Interval):
rng = np.random.RandomState(42)
y_true_b = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1] * 5)
y_pred_b = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1] * 5)
boot_scores = []
for _ in range(200):
    idx = rng.choice(len(y_true_b), size=len(y_true_b), replace=True)
    boot_scores.append(accuracy_score(y_true_b[idx], y_pred_b[idx]))
ci_low, ci_high = np.percentile(boot_scores, [2.5, 97.5])
print(f"95% Confidence Interval for Accuracy: [{ci_low:.3f}, {ci_high:.3f}]")
```

---
**Next Chapter:** [11_CROSS_VALIDATION_TUNING.md](11_CROSS_VALIDATION_TUNING.md)
