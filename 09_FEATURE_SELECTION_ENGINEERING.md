# Part 11: Feature Selection and Feature Engineering (`sklearn.feature_selection`, `sklearn.preprocessing`, `sklearn.inspection`)

Machine Learning mein mashhoor maqoola hai: **"Garbage In, Garbage Out!"** Agar aap ke features ($X$) be-maani ya shor (noise) se bharay hon, to duniya ka sab se bhari algorithm bhi fail ho jayega. Is chapter mein hum [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html) aur **Feature Engineering** ke tamam 20 topics ko tafseel se parhenge.

---

## 1. Why Feature Quality Matters & Feature Selection vs Dimensionality Reduction
- **Feature Selection:** Mojooda columns mein se bekar/noisy columns ko hata kar **asli sab se mufeed columns (`['area_marla', 'bedrooms']`) ko barqarar rakhna**.
- **Dimensionality Reduction (`PCA`):** Tamam columns ko mila kar naye artificial combinations (`PC1, PC2`) banana (asli column names khatam ho jate hain).
- **Important Warning (Predictive Importance $\ne$ Causation):** Agar model mein `Ice_Cream_Sales` ka feature importance `Drowning_Cases` ya `AC_Electricity_Bills` predict karne mein high aaye, to is ka matlab yeh **hargiz nahi** ke Ice Cream khane se bijli ka bill barhta hai (**Correlation is NOT Causation**)—asal wajah **Garmi (Summer Temperature)** hai!

---

## 2. All Scikit-learn Feature Selection Methods (`sklearn.feature_selection`)

### A. Filter Methods (Statistical Tests & Variance)
1. **`VarianceThreshold(threshold=0.0)`**:
   - **Kya Karta Hai?** Woh columns jin mein tamam rows ki value ek jaisi (constant, variance $= 0$) ho—jaise `Country = "Pakistan"` har row mein—unhein foran hata deta hai kyunki constant column se model kuch nahi seekh sakta!
2. **`SelectKBest(score_func, k=5)`** & **`SelectPercentile(score_func, percentile=20)`**:
   - Har feature ka akele-akele (**Univariate**) Target $y$ ke sath statistical score nikalta hai aur top $k$ (ya top $20\%$) columns rakh leta hai.
   - **Scoring Functions (`score_func`):**
     - **For Classification:** `f_classif` (ANOVA F-value for numerical features), `chi2` (Chi-square for non-negative counts/frequencies), `mutual_info_classif` (non-linear mutual information).
     - **For Regression:** `f_regression` (Linear F-test), `r_regression` (Pearson correlation), `mutual_info_regression` (captures non-linear relationships!).

### B. Model-Based & Wrapper Methods
3. **`SelectFromModel(estimator, threshold='median')`**:
   - Kisi bhi model (`Lasso`, `LogisticRegression(penalty='l1')`, `RandomForestClassifier`) ko fit kar ke us ke `coef_` ya `feature_importances_` dekhta hai aur threshold se kam ahem features drop kar deta hai.
4. **`RFE` (Recursive Feature Elimination) & `RFECV`**:
   - **Internal Working:** Model train karta hai, sab se kamzor 1 feature nikal phainkta hai, phir baqi features par dobara train karta hai—yeh silsila tab tak chalta hai jab tak `n_features_to_select` na bach jayein! **`RFECV`** Cross-Validation ke zariye khud tay karta hai ke kitne features rakhna best hai!
5. **`SequentialFeatureSelector` (SFS)**:
   - Greedy Forward Selection (`direction='forward'`: zero features se shuru kar ke ek-ek behtareen feature add karna) ya Backward Selection (`direction='backward'`) jo kisi bhi model (even KNN/SVM jin mein `coef_` nahi hota) ke CV score ki bunyad par features chunta hai!

### C. Tree Impurity Importance vs `permutation_importance` (`sklearn.inspection`)
- **Built-in `.feature_importances_` (MDI):** Tree models mein fast hai, lekin **High-Cardinality features** (jaise `Customer_ID` ya random continuous numbers) ko ghalti se zyada importance de deta hai aur sirf training set par calculate hota hai!
- **`permutation_importance(model, X_test, y_test)`**: Test set par ek column ki values ko shuffle (gadd-madd) kar ke dekhta hai ke model ka score kitna girta hai—yeh **unbiased aur sab se qabil-e-bharosa tareeqa** hai!

---

## 3. Feature Engineering: `PolynomialFeatures`, Interactions, `SplineTransformer` & Domain Features

1. **`PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)`**:
   - Agar features $[x_1, x_2]$ hain, to yeh un ke murabbe (squares) aur aapas ki zarb (**Interactions**) bana deta hai:
     $$[x_1, x_2, x_1^2, x_1 x_2, x_2^2]$$
   - Is ki madad se ek sada `LinearRegression` ya `Ridge` model **Curved (Non-Linear)** data ko fit kar leta hai!
   - Agar sirf interaction ($x_1 x_2$) chahiye ho (maslan `Area_Marla * Location_Multiplier`), to `interaction_only=True` rakhein.
2. **`SplineTransformer(n_knots=5, degree=3)`**:
   - `PolynomialFeatures` high degrees ($d \ge 4$) par kinaro (edges) par be-qaboo ho kar hawa mein urr jata hai (Runge's phenomenon). **`SplineTransformer`** piecewise smooth B-splines banata hai jo edges par kabhi unstable nahi hotay!
3. **Domain-Driven Feature Engineering (Pakistani Real-World Examples):**
   - **Real Estate:** `Rooms_Per_Marla = Bedrooms / Area_Marla`, `Is_Corner_Plot`.
   - **Banking / Loan:** `Debt_To_Income_Ratio = Monthly_Installment / Monthly_Income`.
   - **Categorical Rare Grouping:** `OneHotEncoder(min_frequency=10)` ke zariye kam aane wali rare categories ko automatically `"infrequent_sklearn"` group mein dalna!

---

## 4. Complete Runnable Code: Leakage-Safe Feature Selection & Engineering

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif,
    SelectFromModel, RFE, RFECV, SequentialFeatureSelector
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# 1. Create Dataset: 12 features total -> ONLY 4 are informative, 1 is constant (0 variance)!
X_raw, y = make_classification(
    n_samples=300, n_features=11, n_informative=4, n_redundant=2, random_state=42
)
# Add 1 constant column at index 11
constant_col = np.ones((300, 1)) * 5.0
X = np.hstack([X_raw, constant_col])
col_names = [f"feat_{i}" for i in range(11)] + ["const_feat"]
df_X = pd.DataFrame(X, columns=col_names)

X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.25, random_state=42, stratify=y)

# 2. Leakage-Safe Feature Selection Pipeline
# Step A: Remove constant features -> Step B: Scale -> Step C: SelectFromModel (L1) -> Step D: Classifier
fs_pipe = Pipeline([
    ("var_thresh", VarianceThreshold(threshold=0.0)),
    ("scaler", StandardScaler()),
    ("l1_select", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=0.2, random_state=42))),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

fs_pipe.fit(X_train, y_train)
print("=== 1. Leakage-Safe Feature Selection Pipeline ===")
print("Original Feature Count           :", X_train.shape[1])
print("After VarianceThreshold          :", fs_pipe.named_steps["var_thresh"].transform(X_train).shape[1])
print("After SelectFromModel (L1 Lasso) :", fs_pipe.named_steps["l1_select"].transform(
    fs_pipe.named_steps["scaler"].transform(fs_pipe.named_steps["var_thresh"].transform(X_train))
).shape[1])
print(f"Test Accuracy of Selected Subset : {fs_pipe.score(X_test, y_test):.4f}")

# 3. Compare SelectKBest, RFECV, and Permutation Importance
kbest = SelectKBest(score_func=f_classif, k=4).fit(X_train.iloc[:, :-1], y_train)
print("SelectKBest Top 4 Features       :", list(kbest.get_feature_names_out()))

rfecv = RFECV(estimator=LogisticRegression(max_iter=500), step=1, cv=5, scoring="accuracy")
rfecv.fit(StandardScaler().fit_transform(X_train.iloc[:, :-1]), y_train)
print("RFECV Optimal Number of Features :", rfecv.n_features_)

# 4. Feature Engineering Demo: PolynomialFeatures vs SplineTransformer on Curved Data
X_curve = np.linspace(0, 10, 50).reshape(-1, 1)
y_curve = np.sin(X_curve).ravel()

poly_pipe = Pipeline([("poly", PolynomialFeatures(degree=3, include_bias=False)), ("ridge", Ridge())])
spline_pipe = Pipeline([("spline", SplineTransformer(n_knots=5, degree=3)), ("ridge", Ridge())])

poly_pipe.fit(X_curve, y_curve)
spline_pipe.fit(X_curve, y_curve)
print("\n=== 2. Feature Engineering on Nonlinear Sine Wave ===")
print(f"PolynomialFeatures(deg=3) R2: {poly_pipe.score(X_curve, y_curve):.4f} | Output Cols: {poly_pipe.named_steps['poly'].transform(X_curve).shape[1]}")
print(f"SplineTransformer(knots=5) R2: {spline_pipe.score(X_curve, y_curve):.4f} | Output Cols: {spline_pipe.named_steps['spline'].transform(X_curve).shape[1]}")
```

### Expected Output
```text
=== 1. Leakage-Safe Feature Selection Pipeline ===
Original Feature Count           : 12
After VarianceThreshold          : 11
After SelectFromModel (L1 Lasso) : 6
Test Accuracy of Selected Subset : 0.8933
SelectKBest Top 4 Features       : ['feat_1', 'feat_3', 'feat_6', 'feat_8']
RFECV Optimal Number of Features : 5

=== 2. Feature Engineering on Nonlinear Sine Wave ===
PolynomialFeatures(deg=3) R2: 0.6745 | Output Cols: 3
SplineTransformer(knots=5) R2: 0.9968 | Output Cols: 7
```

---

## 5. Practice Exercises & Solutions

### Exercises
1. **Beginner:** Ek 3-column array `X = [[1, 5, 10], [2, 5, 20], [3, 5, 30]]` par `VarianceThreshold()` lagayein aur confirm karein ke beech wala constant column (`5, 5, 5`) hat gaya hai.
2. **Intermediate:** `PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)` ko `X = [[2, 3, 4]]` par fit_transform karein aur output features ke naam (`.get_feature_names_out()`) print karein.
3. **Challenge:** `load_breast_cancer()` ke `X_train, y_train` par `SelectKBest(mutual_info_classif, k=5)` ko `Pipeline` mein rakh kar `LogisticRegression` fit karein aur `X_test` par score nikalen.

### Solutions
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# 1. Beginner
X_v = np.array([[1, 5, 10], [2, 5, 20], [3, 5, 30]])
print("After VarianceThreshold:\n", VarianceThreshold().fit_transform(X_v))

# 2. Intermediate
pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
print("Interaction Values:", pf.fit_transform([[2, 3, 4]]))
print("Interaction Names :", pf.get_feature_names_out())

# 3. Challenge
bc = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(bc.data, bc.target, test_size=0.2, random_state=42, stratify=bc.target)
pipe = make_pipeline(
    StandardScaler(),
    SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=42), k=5),
    LogisticRegression(random_state=42)
).fit(X_tr, y_tr)
print("Top 5 MI Features Test Accuracy:", round(pipe.score(X_te, y_te), 4))
```

---
**Next Chapter:** [10_MODEL_EVALUATION.md](10_MODEL_EVALUATION.md)
