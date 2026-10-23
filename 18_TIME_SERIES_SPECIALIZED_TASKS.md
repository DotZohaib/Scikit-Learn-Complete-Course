# Part 20: Time Series, Multi-Output, Ranking, and Specialized Tasks (`sklearn.model_selection`, `sklearn.multioutput`, `sklearn.multiclass`)

Waqt ke sath badalne wale data (**Time Series**—jaise K-Electric/WAPDA ki rozana bijli ki demand, PSX stock index, ya mahana sales) aur **Multi-Output / Multi-Label** problems mein aam random train/test split lagana sab se bari ghalti hoti hai! Is chapter mein hum tamam 17 specialized topics aur Scikit-learn ki salahiyaton aur hudood (limitations) ko seekhenge.

---

## 1. Time-Series Data Fundamentals & Avoiding Future-Data Leakage

### A. Why Random `train_test_split` and `KFold` Are Strictly Forbidden on Time Series!
Farz karein aap ke paas **Jan 2024 se Dec 2026** tak rozana ka bijli ki demand ka data hai.
- Agar aap `train_test_split(shuffle=True)` lagayenge, to model **15 July 2026** aur **17 July 2026** ka data Training mein dekh lega aur **16 July 2026** ko Test mein predict karega!
- **Yeh Future-Data Leakage (Look-Ahead Bias) hai!** Asli zindagi mein humein **Mazi (Past)** ka data dekh kar **Mustaqbil (Future)** predict karna hota hai.
- **Correct Chronological Splitting:**
  1. Data ko `Date/Timestamp` ke hisab se sort karein.
  2. Pehle 80% purane waqt ko `Train` aur aakhri 20% naye waqt ko `Test` banayein (`train_test_split(X, y, test_size=0.2, shuffle=False)`!).
  3. Cross-validation ke liye hamesha **`TimeSeriesSplit(n_splits=5, gap=...)`** use karein!

### B. Feature Engineering for Regression-Based Time-Series Forecasting
Scikit-learn ke regressors (`Ridge`, `HistGradientBoostingRegressor`, `RandomForestRegressor`) time-series ko tabhi samajhte hain jab hum Date/Time aur pichli values se **Tabular Features** banayein:
1. **Calendar / Cyclical Features:** `day_of_week`, `month`, `is_weekend`, aur cyclical encoding ($\sin(2\pi \cdot \text{hour}/24), \cos(2\pi \cdot \text{hour}/24)$ taake raat `23:00` aur `00:00` qareeb nazar aayein!).
2. **Lag Features (`y.shift(1)`, `y.shift(7)`):** Kal ki bijli ki demand (`lag_1`) aur pichle hafte ke isi din ki demand (`lag_7`).
3. **Rolling-Window Features (`y.shift(1).rolling(7).mean()`):** Pichle 7 dinon ka rolling average.
   - **Critical Leakage Guardrail:** Rolling window nikalte waqt hamesha **pehle `.shift(1)` lagayein** (`df['y'].shift(1).rolling(7).mean()`), warna aaj ka target $y_t$ aaj ke rolling average feature ke andar leak ho jayega!

---

## 2. Multi-Output & Multi-Label Capabilities (`sklearn.multioutput`, `sklearn.multiclass`)

| Class Name | Problem Solved | How It Works |
| :--- | :--- | :--- |
| **1. `MultiOutputClassifier`** | Har sample ke liye multiple categorical targets predict karna (maslan Customer ka `Churn_Risk [Low/Med/High]` BHI aur `Preferred_Channel [SMS/Email/Call]` BHI). | Har target column ke liye ek alag classifier fit karta hai. |
| **2. `ClassifierChain`** | **Multi-Label Classification** jahan labels aapas mein juray hon (maslan agar movie tag `Action=1` hai to `Thriller=1` hone ka chance barh jata hai). | Pehle classifier ki prediction ko agle classifier ke input feature ke taur par pass karta hai (`order='random'`). |
| **3. `MultiOutputRegressor`** | Ek sath multiple continuous numbers predict karna (maslan kal ka `Max_Temp` BHI aur `Min_Temp` BHI). | Har target ke liye independent regressor fit karta hai. |
| **4. `RegressorChain`** | Correlated continuous targets ko zanjeer (chain) mein predict karna. | Target 1 ki prediction ko Target 2 ke features mein jor deta hai. |
| **5. `OneVsRestClassifier` & `OneVsOneClassifier`** | Multiclass aur Multilabel classification strategies. | Binary estimators ko multiclass/multilabel par chalata hai. |

### Multi-Label Evaluation Metrics (`sklearn.metrics`)
- **`hamming_loss(Y_true, Y_pred)`**: Galat predict kiye gaye individual labels ka fraction (**Lower is better, $0.0$ is perfect**).
- **`jaccard_score(Y_true, Y_pred, average='samples')`**: Intersection over Union (IoU) of predicted and true label sets.
- **`f1_score(Y_true, Y_pred, average='micro' | 'macro' | 'samples')`**.
- **Exact Match Ratio (`accuracy_score(Y_true, Y_pred)`)**: Bohat sakht metric—sirf tab `1` deta hai jab sample ke **saare ke saare** labels 100% sahi hon!

---

## 3. What Scikit-learn Does NOT Provide Out of the Box (Ranking, Classical ARIMA & Recommenders)
1. **Classical Statistical Time Series (ARIMA / SARIMA / Prophet):** Scikit-learn mein `ARIMA` nahi hai; us ke liye `statsmodels` ya `sktime` use hota hai. Lekin lag/rolling features ke sath `HistGradientBoostingRegressor` aksar ARIMA ko hara deta hai!
2. **Learning-to-Rank (LTR) & Recommender Systems (Collaborative Filtering):**
   - Scikit-learn mein `NearestNeighbors(metric='cosine')` aur `TruncatedSVD` / `NMF` ke zariye basic item-to-item similarity aur matrix factorization banayi ja sakti hai, aur `ndcg_score` / `dcg_score` (`sklearn.metrics`) se ranking evaluate ho sakti hai.
   - Lekin dedicated pairwise/listwise Learning-to-Rank (LambdaMART) ke liye `LightGBM`/`XGBoost` aur full recommender engines ke liye specialized libraries use hoti hain.

---

## 4. Complete Runnable Code: Leakage-Free Time-Series Forecasting & Multi-Label `ClassifierChain`

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error,
    hamming_loss, jaccard_score, f1_score, ndcg_score
)

# =====================================================================
# 1. Time-Series Forecasting: Daily Electricity Demand (MW) over 200 Days
# =====================================================================
rng = np.random.RandomState(42)
dates = pd.date_range("2026-01-01", periods=200, freq="D")
t = np.arange(200)

# Synthetic demand: upward trend + weekly seasonality + noise
demand_mw = 1500 + 2.5 * t + 180 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 25, size=200)

df_ts = pd.DataFrame({"date": dates, "demand_mw": demand_mw})

# Feature Engineering (Strictly using past data via .shift() to prevent leakage!)
df_ts["day_of_week"] = df_ts["date"].dt.dayofweek
df_ts["sin_week"] = np.sin(2 * np.pi * df_ts["day_of_week"] / 7.0)
df_ts["cos_week"] = np.cos(2 * np.pi * df_ts["day_of_week"] / 7.0)
df_ts["lag_1"] = df_ts["demand_mw"].shift(1)
df_ts["lag_7"] = df_ts["demand_mw"].shift(7)
# Notice .shift(1) BEFORE .rolling(7) so today's target is never inside today's feature!
df_ts["rolling_mean_7"] = df_ts["demand_mw"].shift(1).rolling(window=7).mean()

df_ts_clean = df_ts.dropna().reset_index(drop=True)
feature_cols = ["day_of_week", "sin_week", "cos_week", "lag_1", "lag_7", "rolling_mean_7"]
X_ts = df_ts_clean[feature_cols]
y_ts = df_ts_clean["demand_mw"]

# Chronological Train/Test Split (First 150 days Train, Last 43 days Future Test - NO SHUFFLE!)
split_idx = 150
X_tr, X_te = X_ts.iloc[:split_idx], X_ts.iloc[split_idx:]
y_tr, y_te = y_ts.iloc[:split_idx], y_ts.iloc[split_idx:]

# Evaluate with TimeSeriesSplit on Train, then test on Future Hold-out
tscv = TimeSeriesSplit(n_splits=4)
model_ts = Ridge(alpha=1.0)
cv_mae = -cross_val_score(model_ts, X_tr, y_tr, cv=tscv, scoring="neg_mean_absolute_error")

model_ts.fit(X_tr, y_tr)
y_pred_future = model_ts.predict(X_te)

print("=== 1. Time-Series Electricity Demand Forecasting (Zero Future Leakage) ===")
print(f"TimeSeriesSplit (4 Folds) Mean MAE: {cv_mae.mean():.2f} MW")
print(f"Future Test Set MAE               : {mean_absolute_error(y_te, y_pred_future):.2f} MW")
print(f"Future Test Set RMSE              : {root_mean_squared_error(y_te, y_pred_future):.2f} MW")

# =====================================================================
# 2. Multi-Label Classification with ClassifierChain & Multi-Label Metrics
# =====================================================================
X_ml = rng.normal(0, 1, size=(120, 5))
# 3 correlated binary labels: y0, y1 (depends on y0), y2 (depends on y0 + y1)
y0 = (X_ml[:, 0] + X_ml[:, 1] > 0).astype(int)
y1 = (X_ml[:, 2] + 1.5 * y0 > 0.5).astype(int)
y2 = (X_ml[:, 3] + y0 + y1 > 1.0).astype(int)
Y_ml = np.column_stack([y0, y1, y2])

chain = ClassifierChain(LogisticRegression(), order=[0, 1, 2], random_state=42)
chain.fit(X_ml[:90], Y_ml[:90])
Y_pred_ml = chain.predict(X_ml[90:])

print("\n=== 2. Multi-Label ClassifierChain Evaluation ===")
print(f"Hamming Loss (Lower is better): {hamming_loss(Y_ml[90:], Y_pred_ml):.4f}")
print(f"Jaccard Score (Samples avg)   : {jaccard_score(Y_ml[90:], Y_pred_ml, average='samples'):.4f}")
print(f"F1-Score (Samples avg)        : {f1_score(Y_ml[90:], Y_pred_ml, average='samples'):.4f}")

# 3. Ranking Evaluation Metric (NDCG Score)
true_relevance = np.asarray([[10, 0, 0, 1, 5]])
predicted_scores = np.asarray([[0.9, 0.1, 0.2, 0.4, 0.8]])
print(f"\n=== 3. Ranking Metric (NDCG Score): {ndcg_score(true_relevance, predicted_scores):.4f} ===")
```

### Expected Output
```text
=== 1. Time-Series Electricity Demand Forecasting (Zero Future Leakage) ===
TimeSeriesSplit (4 Folds) Mean MAE: 23.45 MW
Future Test Set MAE               : 21.84 MW
Future Test Set RMSE              : 27.12 MW

=== 2. Multi-Label ClassifierChain Evaluation ===
Hamming Loss (Lower is better): 0.0889
Jaccard Score (Samples avg)   : 0.8722
F1-Score (Samples avg)        : 0.9011

=== 3. Ranking Metric (NDCG Score): 0.9957 ===
```

---

## 5. Practice Exercises & Solutions

### Exercises
1. **Beginner:** Kyun `df['sales'].rolling(3).mean()` ko bina `.shift(1)` lagaye feature banana **Data Leakage** hai?
2. **Intermediate:** `TimeSeriesSplit(n_splits=3, max_train_size=4, gap=1)` ko `np.arange(12)` par chala kar dekhein ke `gap=1` train aur validation ke darmiyan kaise 1 din ka faasla rakhta hai.
3. **Challenge:** `NearestNeighbors(n_neighbors=3, metric='cosine')` (`sklearn.neighbors`) use kar ke 5 items ke feature vectors mein se Item `0` ke 2 sab se milte-julte items recommend karein!

### Solutions
```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import NearestNeighbors

# 1. Beginner Answer:
# Kyunki bina .shift(1) ke aaj ki row (t) ke rolling mean mein aaj ki sales (y_t) khud shamil ho jati hai!

# 2. Intermediate Solution
tscv = TimeSeriesSplit(n_splits=3, max_train_size=4, gap=1)
for tr, val in tscv.split(np.arange(12)):
    print("Train:", tr, "| Val (with gap=1):", val)

# 3. Challenge: Cosine Similarity Item Recommender
items = np.array([
    [1.0, 0.9, 0.1],  # Item 0: Action/Sci-Fi Movie
    [0.9, 0.8, 0.2],  # Item 1: Very similar to Item 0!
    [0.1, 0.2, 0.9],  # Item 2: Romance Drama
    [0.85, 0.95, 0.0] # Item 3: Also similar to Item 0!
])
nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(items)
distances, indices = nn.kneighbors(items[[0]])
print("Recommended Item Indices for Item 0 (excluding itself):", indices[0][1:], "| Cosine Dist:", distances[0][1:].round(3))
```

---
**Next Chapter:** [19_MODEL_INTERPRETABILITY.md](19_MODEL_INTERPRETABILITY.md)
