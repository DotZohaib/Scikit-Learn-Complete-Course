# Part 4: Introduction to Scikit-learn

## A. Topic Introduction: What is Scikit-learn?
- **Simple Definition:** "Scikit-learn aik Python library hai jo Machine Learning models banane, train karne aur evaluate karne ke liye use hoti hai."
- **Technical Definition:** [Scikit-learn](https://scikit-learn.org/stable/) (`sklearn`) is an open-source, BSD-licensed Python library built on top of NumPy, SciPy, and Matplotlib that provides a unified, consistent object-oriented API for classical supervised and unsupervised machine learning, data preprocessing, model selection, and evaluation.

## B. Why Do We Use It & What Problems Does It Solve?
- **Unified API Design:** Chahe aap simple `LinearRegression` lagayein ya complex `RandomForestClassifier`, har algorithm ke methods ka naam bilkul ek jaisa hai: `.fit()`, `.predict()`, aur `.score()`! Aap ko har algorithm ke liye naya syntax yaad nahi karna parta.
- **End-to-End Tooling:** Sirf models hi nahi, balke missing values bharne (`SimpleImputer`), scaling (`StandardScaler`), cross-validation (`GridSearchCV`), aur leakage-free `Pipeline` banane ke tamam tools ek hi jagah mil jate hain.

## C. Scikit-learn vs NumPy vs Pandas vs TensorFlow vs PyTorch

| Tool / Library | Primary Role | Kab Use Karein? |
| :--- | :--- | :--- |
| **NumPy** | Fast N-dimensional numerical arrays & math | Matrix operations aur array storage ke liye |
| **Pandas** | Tabular DataFrames, CSV/Excel I/O, EDA | Data load karne, saaf karne aur explore karne ke liye |
| **Scikit-learn** | Classical ML (Tabular & basic Text data) | Classification, Regression, Clustering, Preprocessing & Pipelines ke liye |
| **TensorFlow / PyTorch** | Deep Learning on GPUs/TPUs | Images (CNNs), Audio, Large Language Models (LLMs/Transformers) ke liye |

### Strengths and Limitations of Scikit-learn
- **Strengths (Khubiyan):**
  1. Tabular data (CSV/SQL tables) ke liye sab se behtareen aur tez library.
  2. World-class documentation ([User Guide](https://scikit-learn.org/stable/user_guide.html) aur [API Reference](https://scikit-learn.org/stable/api/index.html)).
  3. Zero-leakage `Pipeline` aur `ColumnTransformer` architecture.
- **Limitations (Hudood):**
  1. GPU acceleration native taur par support nahi karta (sirf CPU aur multi-core `n_jobs` use karta hai).
  2. Deep Neural Networks (CNNs, RNNs, Transformers) ya Reinforcement Learning ke liye nahi bana.

---

## D. Import Conventions & Built-in Datasets (`sklearn.datasets`)

### 1. Golden Import Rule
Kabhi bhi `import sklearn` likh kar poori library se classes call na karein (`sklearn.linear_model.LogisticRegression` ghalat aadat hai). Hamesha submodule se specific class import karein:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```

### 2. Datasets Included in Scikit-learn ([`sklearn.datasets`](https://scikit-learn.org/stable/api/sklearn.datasets.html))
Scikit-learn mein 3 tarah ke dataset functions hote hain:
1. **Toy Datasets (`load_*`)**: Library ke andar pehle se mojood chote datasets (internet ki zaroorat nahi hoti):
   - `load_iris()`: 150 flowers, 4 features, 3 classes (Classification).
   - `load_wine()`: 178 wine samples, 13 chemical features, 3 classes (Classification).
   - `load_breast_cancer()`: 569 medical samples, 30 features, 2 classes (Binary Classification).
   - `load_diabetes()`: 442 patients, 10 features, continuous disease progression target (Regression).
   - `load_digits()`: 1,797 8x8 pixel images of handwritten digits 0–9.
2. **Real-World Downloadable Datasets (`fetch_*`)**: Pehli dafa internet se download hote hain:
   - `fetch_california_housing()`: 20,640 California districts house prices (Regression).
   - `fetch_openml()`: [OpenML.org](https://www.openml.org/) se hazaron datasets (jaise Titanic, MNIST) load karne ke liye.
   - `fetch_20newsgroups()`: Text classification ke liye 20 topics ke news articles.
3. **Synthetic Data Generators (`make_*`)**: Practice aur testing ke liye naqli (synthetic) data banane wale functions:
   - `make_classification()`, `make_regression()`, `make_blobs()`, `make_moons()`, `make_circles()`.

---

## E. Core Architecture: Estimators, Transformers, Predictors, and Scorers

Scikit-learn ke har object ko samajhne ke liye yeh 4 bunyadi pillars zahen-nasheen kar lein ([Glossary of Common Terms and API Elements](https://scikit-learn.org/stable/glossary.html)):

1. **Estimator (Har Woh Object Jo Data Se Seekhta Hai):**
   - Har class jismein **`.fit(X, y)`** (supervised) ya **`.fit(X)`** (unsupervised) method mojood ho, woh ek **Estimator** hai.
   - *Examples:* `StandardScaler`, `SimpleImputer`, `LogisticRegression`, `KMeans`.
2. **Transformer (Jo Data Ko Badal Kar Saaf Karta Hai):**
   - Woh Estimator jismein `.fit()` ke sath **`.transform(X)`** aur **`.fit_transform(X)`** methods hon.
   - *Examples:* `StandardScaler`, `OneHotEncoder`, `PCA`, `SimpleImputer`.
3. **Predictor (Jo Prediction Deta Hai):**
   - Woh Estimator jismein **`.predict(X)`** (aur aksar `.score(X, y)`) method ho.
   - *Examples:* `LogisticRegression`, `RandomForestRegressor`, `SVC`.
4. **Scorer / Metric (Jo Performance Naapta Hai):**
   - Functions jo `y_true` aur `y_pred` ka mawazna (comparison) karte hain, jaise `accuracy_score`, `f1_score`, `r2_score`.

### What Do `fit()`, `transform()`, `fit_transform()`, `predict()`, `predict_proba()`, and `score()` Do?

| Method | Kis Par Lagta Hai? | Roman Urdu Explanation |
| :--- | :--- | :--- |
| **`fit(X_train, y_train)`** | Estimators | Training data se statistics (jaise mean/std) ya model weights seekhta hai. **Sirf Training Data par lagayein!** |
| **`transform(X)`** | Transformers | `fit()` mein seekhe gaye hisab ke mutabiq data ko badalta (scale/encode) hai. |
| **`fit_transform(X_train)`** | Transformers | Pehle `fit()` aur phir `transform()` ek sath karta hai. **Sirf `X_train` par use karein, kabhi `X_test` par nahi!** |
| **`predict(X_test)`** | Predictors | Trained model se naye features `X_test` ke liye class label ya continuous value predict karwata hai. |
| **`predict_proba(X_test)`** | Classifiers | Har class ki probability (0.0 se 1.0 ke darmiyan ehtimal) deta hai (maslan `[0.15, 0.85]`). |
| **`score(X_test, y_test)`** | Predictors | Quick evaluation: Classifiers ke liye default **Accuracy** aur Regressors ke liye **$R^2$ Score** return karta hai. |

---

## F. Hyperparameters vs Learned Attributes (Trailing Underscore `_` Convention)

Scikit-learn mein ek bohat khoobsurat qanoon (convention) hai:
1. **Estimator Parameters (Hyperparameters):** Woh settings jo **hum khud** model banate waqt `__init__()` mein dete hain (maslan `C=1.0`, `n_neighbors=5`, `random_state=42`). In ke aakhir mein underscore (`_`) **nahi** hota. Aap `model.get_params()` se inhein dekh sakte hain.
2. **Learned Attributes (Trailing `_`):** Woh cheezein jo model **`.fit()` chalne ke baad data se khud seekhta hai**, un ke naam ke aakhir mein hamesha ek underscore **`_`** hota hai!
   - `model.classes_`: Classification mein kaun-kaun si unique classes seekhi gayin.
   - `model.coef_` aur `model.intercept_`: Linear models ke seekhe gaye weights.
   - `model.n_features_in_`: Training ke waqt kitne columns pass kiye gaye thay.
   - `model.feature_names_in_`: Agar `X` ek Pandas DataFrame tha, to us ke columns ke naam.

---

## G. Complete Working Code: Exploring Estimators, Learned Attributes & Reproducibility

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Load Built-in Dataset as a Pandas DataFrame (as_frame=True)
cancer = load_breast_cancer(as_frame=True)
X = cancer.data[["mean radius", "mean texture", "mean perimeter"]]
y = cancer.target

print(f"Dataset Loaded -> X shape: {X.shape}, Target classes: {list(cancer.target_names)}")

# 2. Split Data (Reproducible with random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Transformer Workflow: fit_transform on Train, ONLY transform on Test!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Transformer Learned Attributes (Ending in '_') ---")
print("Features seen by scaler (feature_names_in_):", scaler.feature_names_in_)
print("Learned Means (mean_)                      :", scaler.mean_.round(2))
print("Learned Scales/Std (scale_)                :", scaler.scale_.round(2))

# 4. Predictor Workflow: fit, predict, predict_proba, score
clf = LogisticRegression(C=1.0, random_state=42)
clf.fit(X_train_scaled, y_train)

print("\n--- Predictor Learned Attributes & Outputs ---")
print("Learned Classes (classes_)     :", clf.classes_)
print("Learned Coefficients (coef_)   :", clf.coef_.round(3))
print("Learned Intercept (intercept_) :", clf.intercept_.round(3))

# First 3 test predictions and probabilities
preds = clf.predict(X_test_scaled[:3])
probas = clf.predict_proba(X_test_scaled[:3])
print("First 3 Predictions            :", preds)
print("First 3 Probabilities [P(0), P(1)]:\n", probas.round(3))
print(f"Test Accuracy via .score()     : {clf.score(X_test_scaled, y_test):.4f}")
```

### Code Explanation in Roman Urdu
1. `load_breast_cancer(as_frame=True)`: `as_frame=True` dene se Scikit-learn NumPy array ke bajaye seedha Pandas DataFrame return karta hai.
2. `scaler.fit_transform(X_train)`: `StandardScaler` ne `X_train` ke 3 columns ka mean (`mean_`) aur standard deviation (`scale_`) seekha aur `X_train` ko scale kar diya.
3. `scaler.transform(X_test)`: Test data par hum ne **`fit` nahi kiya**, balke training data ke mean/std se hi `X_test` ko transform kiya.
4. `clf.predict_proba(X_test_scaled[:3])`: Har sample ke liye do probabilities din—pehla column Class `0` (malignant) ka chance aur doosra column Class `1` (benign) ka chance. Dono ka sum hamesha `1.0` (100%) hota hai.

### Expected Output
```text
Dataset Loaded -> X shape: (569, 3), Target classes: ['malignant', 'benign']

--- Transformer Learned Attributes (Ending in '_') ---
Features seen by scaler (feature_names_in_): ['mean radius' 'mean texture' 'mean perimeter']
Learned Means (mean_)                      : [14.09 19.29 91.74]
Learned Scales/Std (scale_)                : [ 3.5  4.32 24.16]

--- Predictor Learned Attributes & Outputs ---
Learned Classes (classes_)     : [0 1]
Learned Coefficients (coef_)   : [[-1.631 -1.168 -1.874]]
Learned Intercept (intercept_) : [0.69]
First 3 Predictions            : [0 1 0]
First 3 Probabilities [P(0), P(1)]:
 [[0.999 0.001]
 [0.025 0.975]
 [0.916 0.084]]
Test Accuracy via .score()     : 0.8860
```

---

## H. Common Errors & Common Mistakes
1. **`NotFittedError: This LogisticRegression instance is not fitted yet.`**
   - **Ghalti:** `.fit(X_train, y_train)` chalaye baghair `.predict()` ya `.coef_` call kar lena.
   - **Ilaj:** Pehle `.fit()` run karein.
2. **Data Leakage Mistake (`scaler.fit_transform(X_test)`):**
   - **Ghalat:** `X_test_scaled = scaler.fit_transform(X_test)`
   - **Sahi:** `X_test_scaled = scaler.transform(X_test)`

---

## I. Practice Exercises & Solutions

### Exercises
1. **Beginner:** `sklearn.datasets` se `load_wine(as_frame=True)` load karein aur batayein ke is mein kitne samples (`n_samples`), kitne features (`n_features`), aur kaunse target class names (`target_names`) hain.
2. **Intermediate:** Wine dataset ko `train_test_split` (`test_size=0.25, random_state=42`) se split karein, `StandardScaler` fit karein, aur scaler ka `n_features_in_` attribute print karein.
3. **Challenge:** `LogisticRegression` ko scaled Wine dataset par fit karein aur `clf.get_params()` ke zariye uske tamam hyperparameters ki dictionary print karein.

### Solutions
```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Beginner
wine = load_wine(as_frame=True)
print("Shape:", wine.data.shape, "| Targets:", list(wine.target_names))

# 2. Intermediate
X_tr, X_te, y_tr, y_te = train_test_split(wine.data, wine.target, test_size=0.25, random_state=42)
scaler = StandardScaler().fit(X_tr)
print("n_features_in_:", scaler.n_features_in_)

# 3. Challenge
clf = LogisticRegression(max_iter=500, random_state=42).fit(scaler.transform(X_tr), y_tr)
print("Test Score:", round(clf.score(scaler.transform(X_te), y_te), 4))
print("Hyperparameters C and max_iter:", clf.get_params()["C"], clf.get_params()["max_iter"])
```

---
**Next Chapter:** [03_DATA_PREPROCESSING.md](03_DATA_PREPROCESSING.md)
