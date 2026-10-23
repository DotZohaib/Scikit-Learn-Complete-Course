# Part 27A: Professional Revision, Quizzes, Debugging Challenges, Assignments & Capstone Assessment

Is chapter ka maqsad aap ki **conceptual understanding** aur **practical problem-solving** ko parakhna hai—ratta (memorization) nahi! Pehle tamam sawalat khud hal karne ki koshish karein, aur phir aakhir mein diye gaye **Complete Solutions & Explanations** section se apne jawabat match karein.

---

## 1. Topic-Wise Quick Revision Notes (Modules 1 to 26)
1. **ML Foundations & Splitting:** Features $X$ hamesha 2D `(n_samples, n_features)` aur Target $y$ 1D `(n_samples,)` hota hai. Preprocessing se **pehle** `train_test_split` karna lazmi hai.
2. **Preprocessing & Pipelines:** Missing values ke liye `SimpleImputer` / `KNNImputer`, nominal categories ke liye `OneHotEncoder(handle_unknown='ignore')`, ordinal ke liye `OrdinalEncoder`, aur scaling ke liye `StandardScaler` / `RobustScaler` ko hamesha `ColumnTransformer` aur `Pipeline` ke andar rakhein.
3. **Classification & Regression:** Linear models (`LogisticRegression`, `Ridge`, `Lasso`, `SVC`, `MLP`, `KNN`) ko scaling chahiye; Tree ensembles (`RandomForest`, `ExtraTrees`, `HistGradientBoosting`) ko scaling nahi chahiye. Outliers ke liye `HuberRegressor`, `RANSACRegressor`, ya `QuantileRegressor` use karein.
4. **Unsupervised Learning:** Clustering (`KMeans`, `DBSCAN`, `HDBSCAN`, `GaussianMixture`) aur `PCA` se pehle scaling lazmi hai. `TSNE` sirf 2D/3D visualization ke liye hai, pipeline preprocessing ke liye nahi.
5. **Evaluation, Imbalance & Time Series:** Imbalanced data par Accuracy ke bajaye **PR-AUC (`average_precision_score`), F1-score, Recall, aur `TunedThresholdClassifierCV`** use karein. Time-series par hamesha chronological split (`shuffle=False`) aur `TimeSeriesSplit` lagayein.

---

## 2. Beginner Quizzes (Conceptual Understanding — 10 Questions)

- **Q1.** Agar aap ek employee ki salary ($80,000\text{ PKR}$) par fix $5\%$ bonus calculate karna chahte hain, to kya aap ko Machine Learning model train karna chahiye? Kyun ya kyun nahi?
- **Q2.** Scikit-learn mein `scaler.fit_transform(X_train)` aur `scaler.transform(X_test)` mein kya farq hai? Hum `X_test` par `fit_transform` kyun nahi chalate?
- **Q3.** Kisi fitted Scikit-learn estimator ke andar `max_depth` aur `classes_` mein se kaunsa **Hyperparameter** hai aur kaunsa **Learned Attribute**? Kaise pehchana?
- **Q4.** Agar ek dataset mein `City` column (`"Karachi"`, `"Lahore"`, `"Quetta"`) ko `0, 1, 2` encode kar ke `LinearRegression` mein pass kar diya jaye, to model kya ghalat-fehmi paal lega?
- **Q5.** `RandomForestClassifier` aur `KNeighborsClassifier` mein se kis algorithm ke liye `StandardScaler` lagana 100% zaroori hai, aur kyun?
- **Q6.** Ek Credit Card Fraud dataset mein `99.5%` Normal transactions hain aur `0.5%` Fraud. Ek model `99.5%` Accuracy deta hai. Kya yeh model production mein lagane ke qabil hai?
- **Q7.** `KMeans` aur `DBSCAN` mein se kaunsa algorithm outliers/noise ko `-1` label de kar alag pehchan sakta hai aur `n_clusters` pehle se nahi mangta?
- **Q8.** `PCA(n_components=0.95)` mein `0.95` ka kya matlab hai?
- **Q9.** Semi-supervised learning (`SelfTrainingClassifier`, `LabelSpreading`) mein unlabeled training samples ko `y_train` array mein kis number se zahir kiya jata hai?
- **Q10.** Anomaly detection (`IsolationForest`, `OneClassSVM`) ke `.predict()` output mein `+1` aur `-1` ka kya matlab hota hai?

---

## 3. Intermediate Multiple-Choice Questions (MCQs — 10 Questions)

**MCQ 1.** Aap ek `Pipeline([('scaler', StandardScaler()), ('svm', SVC())])` par `GridSearchCV` chala rahe hain. `SVC` ke parameter `C` ko tune karne ke liye `param_grid` mein key ka sahi naam kya hoga?
- (A) `'C'`
- (B) `'SVC__C'`
- (C) `'svm__C'`
- (D) `'svm.C'`

**MCQ 2.** Agar aap ke numerical feature mein chand bohot bare **Outliers** mojood hain, to kaunsa scaler sab se kam mutasir (robust) hoga?
- (A) `MinMaxScaler`
- (B) `StandardScaler`
- (C) `RobustScaler`
- (D) `MaxAbsScaler`

**MCQ 3.** Kaunsa linear regressor L1 regularization penalty use kar ke ghair-zaroori features ke coefficients (`coef_`) ko bilkul **$0.0$** kar deta hai?
- (A) `Ridge`
- (B) `Lasso`
- (C) `HuberRegressor`
- (D) `LinearRegression`

**MCQ 4.** Sparse text matrix (`TfidfVectorizer` output) par bina dense banaye dimensionality reduction karne ke liye kaunsa algorithm recommended hai?
- (A) `PCA`
- (B) `TruncatedSVD`
- (C) `TSNE`
- (D) `Isomap`

**MCQ 5.** Ek hospital dataset mein 100 marijon (patients) ke 500 X-ray scans hain (har patient ke 5 scans). Cross-validation karte waqt kaunsa splitter use karna chahiye taake ek hi marij ke scans Train aur Validation dono mein na jayein?
- (A) `KFold(shuffle=True)`
- (B) `TimeSeriesSplit`
- (C) `StratifiedGroupKFold` / `GroupKFold`
- (D) `LeaveOneOut`

**MCQ 6.** `LocalOutlierFactor` (LOF) ko clean training data par fit kar ke naye unseen test data (`X_test`) par `.predict(X_test)` chalane ke liye kaunsa parameter set karna lazmi hai?
- (A) `contamination=0.1`
- (B) `novelty=True`
- (C) `warm_start=True`
- (D) `probability=True`

**MCQ 7.** Scikit-learn v1.5+ mein bina data leakage ke classifier ka optimal probability cutoff (`threshold`) cross-validation se dhoondne ke liye kaunsi class use hoti hai?
- (A) `CalibratedClassifierCV`
- (B) `TunedThresholdClassifierCV`
- (C) `FixedThresholdClassifier`
- (D) `SelectFromModel`

**MCQ 8.** Kaunsa Scikit-learn classifier **natively** (bina `SimpleImputer` ke) missing values (`NaN`) ko handle kar sakta hai?
- (A) `LogisticRegression`
- (B) `SVC`
- (C) `HistGradientBoostingClassifier`
- (D) `MLPClassifier`

**MCQ 9.** 50 GB CSV dataset jo RAM mein fit nahi aata, us par incremental learning (`.partial_fit()`) ke liye kaunsa classifier pair munasib hai?
- (A) `HashingVectorizer` + `SGDClassifier`
- (B) `TfidfVectorizer` + `SVC(kernel='rbf')`
- (C) `CountVectorizer` + `RandomForestClassifier`
- (D) `PCA` + `GradientBoostingClassifier`

**MCQ 10.** Tree-based `.feature_importances_` (MDI) ke muqable mein `permutation_importance(model, X_test, y_test)` kyun behtar mana jata hai?
- (A) Kyunki woh sirf training set par chalta hai
- (B) Kyunki woh high-cardinality noisy features ki taraf biased nahi hota aur unseen test set par generalization importance naapta hai
- (C) Kyunki woh causation sabit karta hai
- (D) Kyunki usay target `y_test` nahi chahiye hota

---

## 4. Advanced Scenario-Based Questions (Model Selection & Evaluation Scenarios)

- **Scenario 1 (Medical Cancer Screening vs Email Spam Filter):**
  - Case A: Shaukat Khanum / Hospital breast cancer screening tool.
  - Case B: Personal email spam folder filter (jo spam emails ko automatically delete kar deta hai).
  - **Sawal:** Case A aur Case B mein se kis mein **Recall** sab se ahem hai aur kis mein **Precision**? Dono ke liye probability threshold (`0.50` se kam ya zyada) kaise adjust karenge?
- **Scenario 2 (High-Dimensional Genomics Dataset):**
  - Aap ke paas sirf `n_samples = 120` marijon ka data hai lekin `n_features = 15,000` gene expression columns hain, جن mein se sirf ~20 genes bemari se talluq rakhte hain.
  - **Sawal:** Kya aap `KNN` use karenge ya `Lasso` / `LogisticRegression(penalty='l1', solver='saga')`? Kyun?
- **Scenario 3 (Daily Bakery Sales Forecasting):**
  - Ek Junior Data Scientist ne 2 saal ke daily bakery sales data par `RandomForestRegressor` ko `KFold(n_splits=5, shuffle=True)` se evaluate kiya aur `R2 = 0.96` report kiya, lekin production mein agle hafte model buri tarah fail ho gaya.
  - **Sawal:** Junior Data Scientist ne kaunsi 2 bari ghaltiyan kin aur aap isay kaise theek karenge?

---

## 5. Debugging Challenges (Spot and Fix the Bugs!)

### Challenge 1: Find the 3 Severe Bugs in This Pipeline Code
```python
# BUGGY CODE — DO NOT COPY WITHOUT FIXING!
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Bug 1: Scaling before splitting!
X_scaled = StandardScaler().fit_transform(X_num)
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2)

clf = LogisticRegression()
clf.fit(X_tr, y_tr)

# Bug 2: Passing 1D single sample to predict!
single_pred = clf.predict([1.2, -0.5, 0.3])

# Bug 3: Tuning hyperparameter C directly on X_te (Test Set Leakage!)
best_c = max([0.1, 1.0, 10.0], key=lambda c: LogisticRegression(C=c).fit(X_tr, y_tr).score(X_te, y_te))
```

### Challenge 2: Fix the Broken Time-Series Rolling Feature
```python
# BUGGY TIME-SERIES CODE (Target Leakage!)
df["rolling_3d_sales"] = df["sales"].rolling(3).mean()
```

---

## 6. Practical Coding Assignments & Final Capstone Assessment

### Assignment 1: Custom Outlier-Resistant Regression Pipeline
Ek synthetic dataset (`n=300, n_features=5`) banayein jismein 10% `NaN` values aur 5 extreme target outliers hon. `Pipeline` (`SimpleImputer(strategy='median')` + `RobustScaler()` + `HuberRegressor()`) ko standard `LinearRegression()` se Test MAE par compare karein.

### Assignment 2: Semi-Supervised Document Classifier
200 text sentences mein se sirf 20% labels rakhein (`80%` ko `-1` karein) aur `TfidfVectorizer` + `SelfTrainingClassifier(LogisticRegression())` chala kar supervised baseline se accuracy compare karein.

### Final Capstone Assessment Specification
Ek end-to-end **Hospital Patient Readmission Risk (30-Day Readmission = 1)** pipeline likhein jismein:
1. Mixed numerical (`age`, `days_in_hospital`, `num_lab_procedures`) aur categorical (`admission_type`, `primary_diagnosis`) columns hon (with missing values).
2. `StratifiedKFold(5)` ke sath `ColumnTransformer` + `HistGradientBoostingClassifier` train ho.
3. `TunedThresholdClassifierCV(scoring='f1')` se decision threshold optimize ho.
4. `permutation_importance` se top 3 features report kiye jayein.

---

## 7. Complete Solutions & Explanations

### Solutions to Beginner Quizzes (Q1–Q10)
1. **Q1:** Nahi! Jahan exact mathematical rule (`bonus = salary * 0.05`) maloom ho, wahan ML use nahi karte.
2. **Q2:** `fit_transform(X_train)` training data ka mean/std seekhta hai aur badalta hai; `transform(X_test)` usi training mean/std se test data ko scale karta hai taake **Data Leakage** na ho.
3. **Q3:** `max_depth` **Hyperparameter** hai (hum `__init__` mein dete hain); `classes_` **Learned Attribute** hai (kyunki is ke aakhir mein trailing underscore `_` hai jo `.fit()` ke baad banta hai).
4. **Q4:** Model samjhega ke `Quetta (2) > Lahore (1) > Karachi (0)` mein numerical darja-bandi aur faasla hai. Nominal categories ke liye `OneHotEncoder` use karna chahiye.
5. **Q5:** `KNeighborsClassifier` ke liye `StandardScaler` 100% zaroori hai kyunki woh Euclidean distance naapta hai; `RandomForest` single-feature threshold splits karta hai is liye scale-invariant hai.
6. **Q6:** Hargiz nahi! Model har transaction ko `Normal` keh kar 99.5% accuracy le aaya hai jabke Fraud Recall `0%` hai.
7. **Q7:** `DBSCAN` (aur `HDBSCAN`) noise points ko `-1` label deta hai aur `n_clusters` pehle se nahi mangta.
8. **Q8:** Iska matlab hai ke PCA utne Principal Components chune jo original data ki **95% variance (maloomat)** ko mehfooz rakhen.
9. **Q9:** `-1` (Unlabeled samples ko `y_train` mein `-1` se mark kiya jata hai).
10. **Q10:** `+1` = Normal (Inlier), aur `-1` = Anomaly (Outlier/Novelty).

### Solutions to Intermediate MCQs (1–10)
- **MCQ 1:** **(C) `'svm__C'`** (Step name + double underscore + parameter name).
- **MCQ 2:** **(C) `RobustScaler`** (Median aur IQR use karta hai).
- **MCQ 3:** **(B) `Lasso`** (L1 penalty creates sparse zero coefficients).
- **MCQ 4:** **(B) `TruncatedSVD`** (Works directly on sparse matrices without centering).
- **MCQ 5:** **(C) `StratifiedGroupKFold` / `GroupKFold`** (Prevents patient-level group leakage).
- **MCQ 6:** **(B) `novelty=True`**.
- **MCQ 7:** **(B) `TunedThresholdClassifierCV`**.
- **MCQ 8:** **(C) `HistGradientBoostingClassifier`**.
- **MCQ 9:** **(A) `HashingVectorizer` + `SGDClassifier`**.
- **MCQ 10:** **(B)**.

### Solutions to Advanced Scenarios
- **Scenario 1:** Cancer Screening (Case A) mein **Recall** sab se ahem hai (False Negative jaan-lewa hai $\rightarrow$ threshold `0.50` se kam kar ke `0.15`–`0.25` rakhein). Spam Auto-Delete (Case B) mein **Precision** sab se ahem hai (zaroori email delete nahi honi chahiye $\rightarrow$ threshold `0.50` se barha kar `0.85`–`0.90` rakhein).
- **Scenario 2:** $P=15,000 \gg N=120$ mein `KNN` Curse of Dimensionality ka shikar ho kar fail ho jayega. **`LogisticRegression(penalty='l1', solver='saga')` ya `Lasso`** 14,980 faltu genes ka weight `0` kar ke sirf relevant genes chun lega!
- **Scenario 3:** Ghalti 1: Time-series par `KFold(shuffle=True)` lagane se **Future-Data Leakage** hua (kal aur agle din ki sales dekh kar aaj ki sales predict ki!). Ghalti 2: Tree models (`RandomForestRegressor`) training range se bahar **upward trend extrapolate nahi kar sakte**! Ilaj: `TimeSeriesSplit` use karein aur lag/rolling features ke sath `Ridge` ya detrended boosting use karein.

### Solutions to Debugging Challenges & Capstone Code
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# FIXED CHALLENGE 1 CODE:
X_num, y = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=0, random_state=42)

# Fix 1: Split FIRST before scaling!
X_tr, X_te, y_tr, y_te = train_test_split(X_num, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.linear_model import LogisticRegression
pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])

# Fix 3: Tune C using GridSearchCV on X_tr ONLY!
grid = GridSearchCV(pipe, {"clf__C": [0.1, 1.0, 10.0]}, cv=5).fit(X_tr, y_tr)

# Fix 2: Pass 2D array [[1.2, -0.5, 0.3]] to predict!
single_pred = grid.predict([[1.2, -0.5, 0.3]])
print("Fixed Pipeline Test Score:", grid.score(X_te, y_te), "| 2D Single Pred:", single_pred[0])

# FIXED CHALLENGE 2 CODE (Shift by 1 before rolling!):
s = pd.Series([100, 120, 130, 150, 160])
rolling_leak_free = s.shift(1).rolling(3).mean()
print("Leak-Free Rolling 3D Feature:\n", rolling_leak_free.values)
```

---
**Next Chapter:** [26_INTERVIEW_PREPARATION.md](26_INTERVIEW_PREPARATION.md)
