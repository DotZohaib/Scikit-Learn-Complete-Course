# Part 26: Common Errors, Pitfalls, and Troubleshooting Guide (`sklearn.exceptions`)

Jab aap real-world datasets par Scikit-learn code likhte hain, to errors aur warnings ka aana bilkul fitri (natural) baat hai! Ek beginner error dekh kar ghabra jata hai, jabke ek professional Engineer traceback ki aakhri line parh kar 10 seconds mein masla hal kar leta hai. Is chapter mein hum [Scikit-learn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) aur **21 sab se aam Errors, Warnings, aur Conceptual Mistakes** un ke galat code, wajah, aur sahi fix ke sath seekhenge.

---

## 1. Complete Troubleshooting Reference Table (21 Common Issues)

| # | Error / Warning / Issue | Root Cause (Asal Wajah) | Instant Fix (Fori Hal) |
| :--- | :--- | :--- | :--- |
| 1 | **`ModuleNotFoundError: No module named 'sklearn'`** | Environment mein `scikit-learn` install nahi hai ya ghalat Jupyter kernel select hai. | Terminal mein `pip install -U scikit-learn` chalayein. |
| 2 | **`ImportError: cannot import name '...'`** | Class ghalat submodule se import ki gayi hai ya experimental enabler missing hai. | `HalvingGridSearchCV` ke liye `from sklearn.experimental import enable_halving_search_cv` pehle likhein; sahi submodule check karein. |
| 3 | **`ValueError: Expected 2D array, got 1D array instead`** | `X` ko 2D matrix `(n_samples, n_features)` ke bajaye 1D vector `(n_samples,)` de diya gaya! | Single feature ke liye `X.reshape(-1, 1)` ya `df[['col']]` (double brackets!) use karein; single sample ke liye `[[val1, val2]]` dein. |
| 4 | **`ValueError: X has 4 features, but ... is expecting 5 features`** | Training (`fit`) mein 5 columns diye thay aur `.predict(X_test)` mein 4 columns diye! | Hamesha `ColumnTransformer` aur `Pipeline` use karein taake train aur test columns 100% match karein. |
| 5 | **`ValueError: Found input variables with inconsistent numbers of samples: [100, 80]`** | `fit(X, y)` mein `X` ki rows (`100`) aur `y` ki rows (`80`) barabar nahi hain! | `X` aur `y` par rows drop karne ka kaam alag-alag na karein; `df.dropna()` pehle poore DataFrame par karein phir `X, y` alag karein. |
| 6 | **`DataConversionWarning: A column-vector y was passed when a 1d array was expected`** | Target `y` ko `(n_samples, 1)` 2D column بنا kar de diya gaya. | `y.ravel()` ya `df['target']` (single bracket Series) pass karein. |
| 7 | **`ValueError: could not convert string to float: 'Lahore'`** | Categorical string column bina encode kiye numerical model ko de diya gaya! | `ColumnTransformer` mein `OneHotEncoder(handle_unknown='ignore')` ya `OrdinalEncoder` lagayein. |
| 8 | **`ValueError: Input X contains NaN`** | Dataset mein missing values (`NaN`) hain aur estimator natively `NaN` support nahi karta (`LogisticRegression`, `SVC`, `MLP`). | Pipeline mein `SimpleImputer` ya `KNNImputer` lagayein, ya `HistGradientBoostingClassifier` use karein. |
| 9 | **`ValueError: Found unknown categories ['Multan'] in column 0 during transform`** | Test set mein naya shehar aa gaya jo `X_train` mein nahi tha! | `OneHotEncoder(handle_unknown="ignore")` ya `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` use karein! |
| 10 | **`ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT`** | Optimizer مقررہ `max_iter` mein minimum loss tak nahi pohanch saka (aksar unscaled features ki wajah se!). | 1. **`StandardScaler()`** lagayein! 2. `max_iter=1000` barha dein. |
| 11 | **`NotFittedError: This ... instance is not fitted yet`** | `.fit(X_train, y_train)` chalaye baghair `.predict()` ya `.transform()` call kar liya! | Pehle `model.fit(X_train, y_train)` run karein. |
| 12 | **`TypeError: Sparse data was passed, but dense data is required`** | `OneHotEncoder` ya `TfidfVectorizer` ka sparse CSR matrix kisi aise step (`GaussianNB` ya `PCA`) ko de diya jo dense array mangta hai. | `OneHotEncoder(sparse_output=False)` use karein ya sparse matrix ke liye `TruncatedSVD` / `ComplementNB` chunein. |
| 13 | **`UserWarning: X does not have valid feature names, but ... was fitted with feature names`** | Fit karte waqt Pandas `DataFrame` diya tha aur `.predict()` karte waqt NumPy array de diya! | Dono waqt Pandas DataFrame (with same column names) ya dono waqt NumPy array use karein. |
| 14 | **`UndefinedMetricWarning: Precision is ill-defined and being set to 0.0`** | Imbalanced data par model ne kisi class ke liye ek bhi positive (`1`) predict nahi kiya ($TP + FP = 0$). | `precision_score(..., zero_division=0)`, `class_weight='balanced'`, ya `TunedThresholdClassifierCV` use karein. |
| 15 | **Data Leakage (Preprocessing before Split)** | `StandardScaler().fit_transform(X)` poore data par chala kar baad mein `train_test_split` karna! | Pehle `train_test_split` karein aur transformers ko `Pipeline` mein rakhein. |
| 16 | **Overfitting vs Underfitting** | Overfitting: Train `99%`, Test `65%`. Underfitting: Train `60%`, Test `58%`. | Overfitting ke liye Regularization (`alpha` barhayein, `C` kam karein, `max_depth` ghataayein); Underfitting ke liye complex model/features dein. |
| 17 | **Wrong Metric Selection (Accuracy on 99:1 Imbalance)** | 99% accuracy dekh kar khush hona jabke minority class recall `0%` hai. | Imbalanced classification mein `average_precision_score` (PR-AUC), `f1_score`, ya `balanced_accuracy_score` use karein. |
| 18 | **Wrong Train/Test Splitting (Random Split on Time Series or Groups)** | Time-series par `shuffle=True` ya ek hi marij ke records train aur test dono mein dalna. | Time-series ke liye `TimeSeriesSplit` / chronological split; groups ke liye `GroupKFold` use karein. |
| 19 | **Inconsistent Feature Order in Production** | Training mein `[age, salary]` tha aur production mein `[salary, age]` pass kar diya! | DataFrame column selector `df[expected_columns]` aur `ColumnTransformer` use karein. |
| 20 | **Deprecated / Removed API Usage** | Purane tutorials se `from sklearn.externals import joblib`, `Boston` dataset, ya `OneHotEncoder(sparse=False)` copy karna. | Direct `import joblib`, `fetch_california_housing`, aur `OneHotEncoder(sparse_output=False)` use karein! |
| 21 | **MemoryError / Freezing on Large Datasets** | $N=100,000$ rows par `SVC(kernel='rbf')` ya `OneHotEncoder` par `.toarray()` chala dena! | `HistGradientBoosting`, `SGDClassifier`, aur sparse matrices (`sparse_output=True`) use karein. |

---

## 2. Complete Runnable Code: Demonstrating Typical Errors & Their Exact Fixes

```python
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError, ConvergenceWarning
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

print("=== Troubleshooting Live Demonstrations ===\n")

# ---------------------------------------------------------------------
# Issue 1: ValueError: Expected 2D array, got 1D array instead
# ---------------------------------------------------------------------
x_1d = np.array([10.0, 20.0, 30.0, 40.0])
y = np.array([0, 0, 1, 1])

try:
    LogisticRegression().fit(x_1d, y)
except ValueError as e:
    print("1. [CAUGHT 1D Array Error]:", str(e).splitlines()[0])
    # FIX: Reshape 1D vector to 2D column matrix (n_samples, 1)
    x_2d = x_1d.reshape(-1, 1)
    clf_fixed = LogisticRegression().fit(x_2d, y)
    print("   [FIXED]: Reshaped from", x_1d.shape, "to", x_2d.shape, "-> Fit succeeded!\n")

# ---------------------------------------------------------------------
# Issue 2: ValueError: could not convert string to float + Unknown Category
# ---------------------------------------------------------------------
X_train_str = pd.DataFrame({"city": ["Karachi", "Lahore", "Karachi", "Islamabad"]})
X_test_str = pd.DataFrame({"city": ["Multan"]})  # 'Multan' was never seen in training!

try:
    LogisticRegression().fit(X_train_str, y)
except ValueError as e:
    print("2. [CAUGHT String Feature Error]:", str(e))
    # FIX: Wrap OneHotEncoder(handle_unknown='ignore') in a Pipeline
    pipe_str = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression())
    pipe_str.fit(X_train_str, y)
    print("   [FIXED]: Predicted unseen 'Multan' safely -> Class:", pipe_str.predict(X_test_str)[0], "\n")

# ---------------------------------------------------------------------
# Issue 3: ValueError: Input X contains NaN
# ---------------------------------------------------------------------
X_nan = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0], [6.0, 7.0]])
try:
    LogisticRegression().fit(X_nan, y)
except ValueError as e:
    print("3. [CAUGHT NaN Error]:", str(e).splitlines()[0])
    # FIX: Add SimpleImputer in Pipeline
    pipe_nan = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression()).fit(X_nan, y)
    print("   [FIXED]: Imputed NaN inside Pipeline -> Predictions:", pipe_nan.predict(X_nan), "\n")

# ---------------------------------------------------------------------
# Issue 4: NotFittedError
# ---------------------------------------------------------------------
unfitted_clf = LogisticRegression()
try:
    unfitted_clf.predict([[1.0, 2.0]])
except NotFittedError as e:
    print("4. [CAUGHT NotFittedError]:", str(e).splitlines()[0])
    unfitted_clf.fit([[1.0, 2.0], [3.0, 4.0]], [0, 1])
    print("   [FIXED]: Called .fit() before .predict() ->", unfitted_clf.predict([[1.0, 2.0]]))
```

### Expected Output
```text
=== Troubleshooting Live Demonstrations ===

1. [CAUGHT 1D Array Error]: Expected a 2-dimensional container but got <class 'numpy.ndarray'> instead.
   [FIXED]: Reshaped from (4,) to (4, 1) -> Fit succeeded!

2. [CAUGHT String Feature Error]: could not convert string to float: 'Karachi'
   [FIXED]: Predicted unseen 'Multan' safely -> Class: 0 

3. [CAUGHT NaN Error]: Input X contains NaN.
   [FIXED]: Imputed NaN inside Pipeline -> Predictions: [0 0 1 1] 

4. [CAUGHT NotFittedError]: This LogisticRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
   [FIXED]: Called .fit() before .predict() -> [0]
```

---

## 3. Practice Exercises & Solutions

### Exercises
1. **Beginner:** Ek code line `df['area_marla']` aur `df[['area_marla']]` mein kya farq hai aur Scikit-learn ke `X` ke liye kaunsa chahiye?
2. **Intermediate:** Neeche diye gaye leaky code mein ghalti pehchanein aur usay theek karein:
   ```python
   # LEAKY CODE:
   X_scaled = StandardScaler().fit_transform(X)
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
   ```
3. **Challenge:** Agar `OneHotEncoder(handle_unknown='error')` (default) laga ho aur test data mein naya category aa jaye to woh crash ho jata hai. Agar aap `OrdinalEncoder` use kar rahe hon to unseen categories ko `-1` code dene ke liye kaunse parameters set karenge?

### Solutions
```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# 1. Beginner Answer:
# df['area_marla'] 1D Pandas Series (shape: (n,)) deta hai jo X mein ValueError dega.
# df[['area_marla']] 2D Pandas DataFrame (shape: (n, 1)) deta hai jo Scikit-learn ke X ke liye bilkul durust hai!

# 2. Intermediate Fixed Code:
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# 3. Challenge Solution:
ord_safe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
ord_safe.fit(pd.DataFrame({"grade": ["A", "B", "C"]}))
print("Unseen Grade 'Z' encoded safely as:", ord_safe.transform(pd.DataFrame({"grade": ["Z"]}))[0, 0])
```

---
**Next Chapter:** [24_PRACTICAL_PROJECTS.md](24_PRACTICAL_PROJECTS.md)
