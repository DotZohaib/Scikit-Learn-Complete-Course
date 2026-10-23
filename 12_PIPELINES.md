# Part 14: Pipelines and Advanced Preprocessing (`sklearn.pipeline`, `sklearn.compose`, `sklearn.base`)

Real-world production Machine Learning mein hum kabhi bhi `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, aur `Classifier` ko alag-alag bikhray hue variables mein nahi chalate! Hum un sab ko ek single **`Pipeline`** mein bandh dete hain ([Scikit-learn Pipelines & Composite Estimators](https://scikit-learn.org/stable/modules/compose.html)). Is chapter mein hum tamam 18 pipeline topics aur ek full Mixed-Data Classification Project seekhenge.

---

## 1. Why Pipelines Are Necessary (3 Huge Benefits)
1. **Zero Data Leakage in Cross-Validation:** Agar aap `GridSearchCV(pipe, ...)` chalate hain, to Scikit-learn har CV fold ke andar `imputer` aur `scaler` ko **sirf training fold par `fit()`** karta hai aur validation fold par sirf `transform()` karta hai!
2. **Single-Object Deployment:** Aap ko 5 alag files (`imputer.joblib`, `ohe.joblib`, `scaler.joblib`, `model.joblib`) save nahi karni partin—sirf ek `pipeline.joblib` save karein!
3. **Hyperparameter Tuning Across Preprocessing & Model Together:** Aap ek hi `GridSearchCV` mein yeh bhi tune kar sakte hain ke `SimpleImputer` mein `'mean'` behtar hai ya `'median'`, aur `PCA` mein `5` components behtar hain ya `10`!

---

## 2. Core Pipeline Tools Explained

1. **`Pipeline(steps=[('name1', trans1), ('name2', model)], memory=...)`**:
   - Har step ko aap apna pasandida naam dete hain (jaise `('scaler', StandardScaler())`).
   - **Double-Underscore (`__`) Parameter Syntax:** GridSearchCV mein kisi step ke andar ka parameter badalna ho to `<step_name>__<param_name>` likhte hain (maslan `model__C` ya `preprocessor__num__imputer__strategy`).
   - **`memory='cache_dir'` (Pipeline Caching):** Jab GridSearchCV mein model ke parameters badal rahe hon lekin heavy preprocessing bar-bar wohi chal rahi ho, to `memory` dene se Scikit-learn preprocessing ka result cache kar leta hai aur tuning 3x–5x tez ho jati hai!
2. **`make_pipeline(*steps)`**:
   - Naam khud rakhne ki zaroorat nahi—class ka lowercase naam khud-ba-khud rakh deta hai (maslan `StandardScaler()` ka naam `'standardscaler'` ban jata hai).
3. **`ColumnTransformer` & `make_column_selector`**:
   - **`make_column_selector(dtype_include=np.number)`** aur **`make_column_selector(dtype_include=object)`** DataFrame mein se numerical aur categorical columns khud pehchan lete hain—aap ko haath se column names ki lambi list bhi nahi likhni parti!
4. **`FeatureUnion` & `make_union`**:
   - Ek hi data par 2 transformers **parallel** chala kar un ke outputs ko sath-sath (`hstack`) jorne ke liye (maslan `PCA(n_components=2)` + `SelectKBest(k=2)` $\rightarrow$ total 4 columns).
5. **`TransformedTargetRegressor(regressor=..., func=np.log1p, inverse_func=np.expm1)`**:
   - Jab regression target $y$ (maslan House Prices) bohot skewed ho, to yeh training ke waqt $y$ par automatically $\log(1+y)$ lagata hai aur `.predict()` karte waqt wapas $\exp(\hat{y})-1$ kar ke asli Lakhs PKR mein jawab deta hai!
6. **Custom Transformers (`FunctionTransformer` vs `BaseEstimator, TransformerMixin`)**:
   - Stateless math ke liye `FunctionTransformer` use karein.
   - Agar transformer ko training data se kuch seekhna (`fit`) ho (maslan IQR outlier clipping bounds), to `BaseEstimator` aur `TransformerMixin` inherit kar ke apni class banayein!
7. **Pandas DataFrame Output (`set_output(transform="pandas")`)**:
   - Scikit-learn v1.2+ mein kisi bhi transformer ya pipeline par `.set_output(transform="pandas")` lagane se NumPy array ke bajaye column names ke sath saaf Pandas DataFrame return hota hai!
8. **Metadata Routing (`sklearn.set_config(enable_metadata_routing=True)`)**:
   - Scikit-learn v1.3+ mein `sample_weight` ya `groups` ko pipeline ke andar specific steps tak pohanchane ka modern nizam.

---

## 3. Complete Runnable Project: Full Mixed-Data Classification Pipeline with Custom Transformer, `make_column_selector`, `TransformedTargetRegressor` & `GridSearchCV`

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, make_column_selector, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# =====================================================================
# 1. Custom Transformer: IQR Outlier Clipper (Inherits BaseEstimator, TransformerMixin)
# =====================================================================
class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        q25 = np.nanpercentile(X_arr, 25, axis=0)
        q75 = np.nanpercentile(X_arr, 75, axis=0)
        iqr = q75 - q25
        self.lower_bounds_ = q25 - self.factor * iqr
        self.upper_bounds_ = q75 + self.factor * iqr
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)

# =====================================================================
# 2. Create Mixed Numerical/Categorical Dataset with Missing Values & Outliers
# =====================================================================
rng = np.random.RandomState(42)
n = 240
df = pd.DataFrame({
    "age": rng.randint(21, 65, size=n).astype(float),
    "monthly_income_k": rng.lognormal(mean=4.5, sigma=0.6, size=n),
    "credit_score": rng.normal(650, 70, size=n),
    "city": rng.choice(["Karachi", "Lahore", "Islamabad", "Peshawar"], size=n),
    "employment": rng.choice(["Salaried", "Self-Employed", "Contract"], size=n),
})

# Target: Loan Approval (1 = Approved, 0 = Rejected)
logits = 0.02 * df["credit_score"] + 0.015 * df["monthly_income_k"] - 14.2 + (df["city"] == "Karachi") * 0.5
y = (logits + rng.normal(0, 0.8, size=n) > 0).astype(int)

# Inject Missing Values & Extreme Outlier
df.loc[rng.choice(n, 15, replace=False), "monthly_income_k"] = np.nan
df.loc[rng.choice(n, 12, replace=False), "city"] = np.nan
df.loc[0, "monthly_income_k"] = 99999.0  # Extreme outlier!

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42, stratify=y)

# =====================================================================
# 3. Build Production Pipeline with make_column_selector & Custom Transformer
# =====================================================================
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clipper", IQRClipper(factor=1.5)),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object))
])

full_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=500, random_state=42))
])

# =====================================================================
# 4. Tune Preprocessing, Custom Clipper & Model Together via GridSearchCV!
# =====================================================================
param_grid = [
    {
        "preprocess__num__imputer__strategy": ["mean", "median"],
        "preprocess__num__clipper__factor": [1.5, 3.0],
        "model": [LogisticRegression(max_iter=500, random_state=42)],
        "model__C": [0.1, 1.0, 10.0]
    },
    {
        "preprocess__num__imputer__strategy": ["median"],
        "model": [RandomForestClassifier(random_state=42)],
        "model__n_estimators": [50, 100],
        "model__max_depth": [4, 8]
    }
]

search = GridSearchCV(full_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
search.fit(X_train, y_train)

print("=== 1. Mixed-Data Pipeline GridSearchCV Results ===")
print("Best Pipeline Configuration:\n ", search.best_params_)
print(f"Best 5-Fold CV F1-Score: {search.best_score_:.4f}")

y_pred = search.predict(X_test)
print("\n=== 2. Hold-Out Test Set Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))

# =====================================================================
# 5. Bonus Demos: FeatureUnion & TransformedTargetRegressor
# =====================================================================
union = FeatureUnion([
    ("pca", PCA(n_components=2)),
    ("kbest", SelectKBest(f_classif, k=2))
])
X_num_clean = num_pipeline.fit_transform(X_train[["age", "monthly_income_k", "credit_score"]], y_train)
print("FeatureUnion Output Shape (2 PCA + 2 KBest = 4 cols):", union.fit_transform(X_num_clean, y_train).shape)

# TransformedTargetRegressor (Log-transforming skewed target automatically)
tt_reg = TransformedTargetRegressor(
    regressor=Ridge(), func=np.log1p, inverse_func=np.expm1
)
tt_reg.fit(X_num_clean, np.exp(y_train * 5.0))
print("TransformedTargetRegressor Sample Prediction:", np.round(tt_reg.predict(X_num_clean[:2]), 2))
```

### Expected Output
```text
=== 1. Mixed-Data Pipeline GridSearchCV Results ===
Best Pipeline Configuration:
  {'model': LogisticRegression(max_iter=500, random_state=42), 'model__C': 1.0, 'preprocess__num__clipper__factor': 1.5, 'preprocess__num__imputer__strategy': 'median'}
Best 5-Fold CV F1-Score: 0.8924

=== 2. Hold-Out Test Set Classification Report ===
              precision    recall  f1-score   support

Rejected (0)       0.89      0.86      0.88        29
Approved (1)       0.88      0.90      0.89        31

    accuracy                           0.88        60
   macro avg       0.88      0.88      0.88        60
weighted avg       0.88      0.88      0.88        60

FeatureUnion Output Shape (2 PCA + 2 KBest = 4 cols): (180, 4)
TransformedTargetRegressor Sample Prediction: [  1.72 135.4 ]
```

---

## 4. Practice Exercises & Solutions

### Exercises
1. **Beginner:** `make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression())` banayein aur us ke `.named_steps` print karein.
2. **Intermediate:** Ek `Pipeline` par `.set_output(transform="pandas")` lagayein (sirf preprocessing steps par) aur verify karein ke `.fit_transform()` ka output `pd.DataFrame` hai.
3. **Challenge:** Ek custom estimator `MeanBaselineRegressor(BaseEstimator, RegressorMixin)` likhein jo `.fit(X, y)` mein `y` ka mean (`self.mean_ = np.mean(y)`) save kare aur `.predict(X)` mein har row ke liye wohi `self.mean_` return kare.

### Solutions
```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Beginner
p1 = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression())
print("Named Steps:", list(p1.named_steps.keys()))

# 2. Intermediate
df_s = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [10.0, 20.0, 30.0]})
prep = make_pipeline(SimpleImputer(), StandardScaler()).set_output(transform="pandas")
out_df = prep.fit_transform(df_s)
print("Output Type:", type(out_df).__name__, "| Columns:", list(out_df.columns))

# 3. Challenge
class MeanBaselineRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean_)

reg = MeanBaselineRegressor().fit([[1], [2], [3]], [10, 20, 30])
print("Custom Regressor Predictions:", reg.predict([[4], [5]]))  # [20., 20.]
```

---
**Next Chapter:** [13_IMBALANCED_LEARNING.md](13_IMBALANCED_LEARNING.md)
