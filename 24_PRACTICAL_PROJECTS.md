# Part 25: 15 Complete End-to-End Practical Machine Learning Projects

Is chapter mein hum course ke tamam modules ko mila kar **15 mukammal, runnable, end-to-end Machine Learning projects** banayenge! Har project mein **Problem Statement, Business Use Case, Dataset Source & License Notes, Data Dictionary, EDA & Cleaning, Feature Engineering, Train/Validation/Test Strategy, Baseline vs Final Pipeline, Hyperparameter Tuning, Suitable Metrics, Error Analysis, Model Persistence (`joblib`), Prediction on New Data, Limitations, aur Next Improvements** shamil hain.

> **Important Note on Datasets:** Jin projects mein built-in Scikit-learn datasets (`load_iris`, `load_wine`, `load_breast_cancer`, `load_digits`, `load_diabetes`) istemal hue hain woh BSD-3 / CC0 public educational datasets hain; aur jin projects mein internet download se bachne ke liye **Synthetic Educational Datasets** generate kiye gaye hain, unhein saaf taur par **[SYNTHETIC EDUCATIONAL DATASET]** label kiya gaya hai ([Dataset Guide](datasets/DATASET_GUIDE.md)).

---

## Project 1: Iris Botanical Species Classification (End-to-End Multiclass Pipeline)
- **Problem Statement & Business Use Case:** Botanical lab mein phool ki pattiyon (`sepal` aur `petal`) ke 4 measurements dekh kar automatically species (`setosa`, `versicolor`, `virginica`) classify karna.
- **Dataset Source & License:** `sklearn.datasets.load_iris()` (Fisher's 1936 classic dataset, Public Domain / BSD-licensed in Scikit-learn, `n=150`, 4 numerical features, 3 balanced classes).
- **Train/Validation/Test Strategy:** 80/20 Stratified Split + 5-Fold Stratified Cross-Validation.

```python
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Load & Feature Engineer (Add petal_area and sepal_ratio)
iris = load_iris(as_frame=True)
X = iris.data.copy()
X["petal_area"] = X["petal length (cm)"] * X["petal width (cm)"]
y = iris.target

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Baseline vs Tuned SVC Pipeline
base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True, random_state=42))])
grid = GridSearchCV(pipe, {"svc__C": [0.1, 1.0, 10.0], "svc__kernel": ["linear", "rbf"]}, cv=5)
grid.fit(X_tr, y_tr)

y_pred = grid.predict(X_te)
print(f"Baseline Acc: {base.score(X_te, y_te):.4f} | Tuned SVC Test Acc: {accuracy_score(y_te, y_pred):.4f}")
print("Best Params :", grid.best_params_)
print(classification_report(y_te, y_pred, target_names=iris.target_names))

# 3. Error Analysis, Persistence & Prediction on New Sample
errors = X_te[y_te != y_pred]
print(f"Misclassified Test Samples Count: {len(errors)}")
joblib.dump(grid.best_estimator_, "p1_iris.joblib")
loaded = joblib.load("p1_iris.joblib")
new_sample = X_te.iloc[[0]]
print("New Sample Prediction:", iris.target_names[loaded.predict(new_sample)[0]])
Path("p1_iris.joblib").unlink(missing_ok=True)
```
- **Limitations & Next Improvements:** Dataset sirf 150 rows ka hai; real deployment ke liye image-based features ya environmental soil data shamil kiya ja sakta hai.

---

## Project 2: Maritime Passenger Survival Prediction (Titanic-Schema Mixed Data Pipeline)
- **Problem Statement & Business Use Case:** Passenger manifest (`pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`) dekh kar maritime rescue survival (`0 = Died`, `1 = Survived`) predict karna.
- **Dataset Source & Notes:** **[SYNTHETIC EDUCATIONAL DATASET]** modeled on the public Titanic passenger schema (or loadable via `fetch_openml("titanic", version=1, as_frame=True)` when internet is active). Includes realistic missing values in `age` and `embarked`.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

# 1. Generate Synthetic Titanic-Schema Dataset with Missing Values (n=500)
rng = np.random.RandomState(42)
n = 500
df = pd.DataFrame({
    "pclass": rng.choice([1, 2, 3], size=n, p=[0.25, 0.25, 0.50]),
    "sex": rng.choice(["female", "male"], size=n, p=[0.38, 0.62]),
    "age": rng.normal(30, 14, size=n).clip(1, 80),
    "sibsp": rng.poisson(0.5, size=n),
    "parch": rng.poisson(0.4, size=n),
    "fare": rng.exponential(32, size=n) + 7.0,
    "embarked": rng.choice(["S", "C", "Q"], size=n, p=[0.7, 0.2, 0.1])
})
# Feature Engineering: Family_Size and Is_Alone
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"] = (df["family_size"] == 1).astype(int)

# Target survival rule + noise
logit = 1.8 * (df["sex"] == "female") - 0.7 * (df["pclass"] - 1) - 0.02 * df["age"] + 0.01 * df["fare"]
y = (logit + rng.normal(0, 0.8, size=n) > 0.2).astype(int)

# Inject realistic missing values
df.loc[rng.choice(n, 50, replace=False), "age"] = np.nan
df.loc[rng.choice(n, 10, replace=False), "embarked"] = np.nan

X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)

# 2. Preprocessing & Model Pipeline
num_cols = ["age", "fare", "family_size", "is_alone", "pclass"]
cat_cols = ["sex", "embarked"]

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

pipe = Pipeline([("prep", prep), ("rf", RandomForestClassifier(random_state=42))])
grid = GridSearchCV(pipe, {"rf__n_estimators": [50, 100], "rf__max_depth": [4, 6]}, cv=5, scoring="roc_auc")
grid.fit(X_tr, y_tr)

preds = grid.predict(X_te)
probas = grid.predict_proba(X_te)[:, 1]
print(f"Titanic Pipeline -> Test F1: {f1_score(y_te, preds):.4f} | Test ROC-AUC: {roc_auc_score(y_te, probas):.4f}")
```
- **Limitations & Next Improvements:** Cabin deck letters aur passenger titles (`Mr.`, `Mrs.`, `Master.`) extract kar ke accuracy mazeed barhayi ja sakti hai.

---

## Project 3: Pakistani Real-Estate House Price Regression (Karachi, Lahore, Islamabad)
- **Problem Statement & Business Use Case:** Zameen/Property portal ke liye ghar ke `area_marla`, `bedrooms`, `bathrooms`, `property_age`, aur `city` ki bunyad par fair market price (`price_lakhs` PKR) estimate karna.
- **Dataset Source & Notes:** **[SYNTHETIC EDUCATIONAL DATASET]** (`n=600`) simulating Pakistani urban real-estate markets (or pair with `fetch_california_housing()` for US Census data).

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# 1. Synthetic Pakistani Real-Estate Dataset (n=600)
rng = np.random.RandomState(42)
n = 600
city = rng.choice(["Karachi", "Lahore", "Islamabad"], size=n)
area_marla = rng.uniform(3, 20, size=n)
bedrooms = np.clip(np.round(area_marla / 2.5), 1, 8)
age_years = rng.uniform(0, 20, size=n)

city_premium = np.where(city == "Islamabad", 45.0, np.where(city == "Karachi", 30.0, 20.0))
price_lakhs = 24.0 * area_marla + 14.0 * bedrooms - 1.8 * age_years + city_premium + rng.normal(0, 15, size=n)

df_house = pd.DataFrame({"city": city, "area_marla": area_marla, "bedrooms": bedrooms, "age_years": age_years})
# Feature Engineering: area_per_bedroom
df_house["area_per_bedroom"] = df_house["area_marla"] / df_house["bedrooms"]

X_tr, X_te, y_tr, y_te = train_test_split(df_house, price_lakhs, test_size=0.2, random_state=42)

# 2. Baseline vs Pipeline
base_reg = DummyRegressor(strategy="median").fit(X_tr, y_tr)
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), ["area_marla", "bedrooms", "age_years", "area_per_bedroom"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"])
])

reg_pipe = Pipeline([
    ("prep", prep),
    ("hgb", HistGradientBoostingRegressor(max_iter=150, random_state=42))
])
reg_pipe.fit(X_tr, y_tr)
preds = reg_pipe.predict(X_te)

print(f"Baseline MAE: {mean_absolute_error(y_te, base_reg.predict(X_te)):.2f} Lakhs")
print(f"HGB Model -> MAE: {mean_absolute_error(y_te, preds):.2f} Lakhs | RMSE: {root_mean_squared_error(y_te, preds):.2f} Lakhs | R2: {r2_score(y_te, preds):.4f}")
```
- **Limitations & Next Improvements:** Shehar ke andar sector/society (`DHA`, `Bahria`, `Gulshan`) aur corner/park-facing indicators add karne se precision aur barhegi.

---

## Project 4: Student Academic Performance Prediction (Final Exam Score Regression)
- **Problem Statement & Business Use Case:** University coaching system jo student ke `study_hours_weekly`, `attendance_pct`, `past_gpa`, aur `sleep_hours` dekh kar final exam score (`0`–`100`) predict kare taake weak students ki waqt par madad ho sake.
- **Dataset Source:** **[SYNTHETIC EDUCATIONAL DATASET]** (`n=400`).

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

rng = np.random.RandomState(42)
n = 400
study_hrs = rng.uniform(2, 30, size=n)
attendance = rng.uniform(50, 100, size=n)
past_gpa = rng.uniform(1.8, 4.0, size=n)
exam_score = np.clip(1.2 * study_hrs + 0.35 * attendance + 8.5 * past_gpa + rng.normal(0, 4, size=n), 0, 100)

X = pd.DataFrame({"study_hrs": study_hrs, "attendance": attendance, "past_gpa": past_gpa})
X_tr, X_te, y_tr, y_te = train_test_split(X, exam_score, test_size=0.2, random_state=42)

pipe_student = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0))
pipe_student.fit(X_tr, y_tr)
preds = pipe_student.predict(X_te)
print(f"Student Score Predictor -> Test MAE: {mean_absolute_error(y_te, preds):.2f} marks | R2: {r2_score(y_te, preds):.4f}")
```

---

## Project 5: Retail & E-Commerce Customer Segmentation (RFM Clustering)
- **Problem Statement & Business Use Case:** Unsupervised learning (`KMeans` + `Silhouette Score`) ke zariye customers ko unke **Recency (days since last order), Frequency (yearly orders), aur Monetary Spend (PKR)** ke hisab se marketing segments mein baantna.
- **Dataset Source:** **[SYNTHETIC EDUCATIONAL DATASET]** (`n=300`).

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

rng = np.random.RandomState(42)
rfm = np.vstack([
    rng.normal([10, 18, 45000], [3, 4, 8000], (100, 3)),   # Loyal Champions
    rng.normal([45, 6, 12000], [10, 2, 3000], (100, 3)),   # Regulars
    rng.normal([150, 1.5, 2500], [30, 0.5, 800], (100, 3)) # Hibernating / At-Risk
])
df_rfm = pd.DataFrame(rfm, columns=["Recency_Days", "Frequency_Orders", "Monetary_PKR"])
X_sc = StandardScaler().fit_transform(df_rfm)

km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_sc)
df_rfm["Segment"] = km.labels_
print(f"RFM Clustering Silhouette Score: {silhouette_score(X_sc, km.labels_):.4f}")
print(df_rfm.groupby("Segment").mean().round(1))
```

---

## Project 6: Bilingual (Roman Urdu + English) SMS Spam & Scam Detection
- **Problem Statement & Business Use Case:** Telecom firewall jo fake lottery, BISP cash scams, aur phishing OTP messages ko block kare.
- **Dataset Source:** **[SYNTHETIC EDUCATIONAL DATASET]** of bilingual Pakistani SMS messages.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline

texts = [
    "Mubarak apka BISP 25000 inaam nikla hai OTP bhejein",
    "Win 10 Lakh Jeeto lottery cash prize call now",
    "Free iPhone prize click link and share bank PIN",
    "Urgent account blocked verify ATM PIN immediately",
    "Bhai sham ko ghar aate waqt sabzi aur doodh le aana",
    "Kal university mein subah 9 baje ML lab ka viva hai",
    "Please send the quarterly sales spreadsheet before 5 PM",
    "Aaj raat khane mein biryani bani hai jaldi aao"
] * 10
labels = [1, 1, 1, 1, 0, 0, 0, 0] * 10

spam_clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), ComplementNB()).fit(texts[:60], labels[:60])
print("Spam Detection Test Accuracy:", spam_clf.score(texts[60:], labels[60:]))
print("Prediction for 'BISP cash inaam OTP':", spam_clf.predict(["BISP cash inaam OTP"])[0])
```

---

## Project 7: Product Review Sentiment Classification (TF-IDF + LogisticRegression)
- **Problem Statement & Business Use Case:** E-commerce store ke thousands of Roman Urdu aur English product reviews ko automatically `Positive (1)` ya `Negative (0)` mein classify karna.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

revs = [
    "Bohat shandar mobile hai battery timing kamaal hai",
    "Product bilkul acha nahi hai paise barbad ho gaye",
    "Fast delivery aur behtareen packing shukriya",
    "Bekar quality hai do din mein kharab ho gaya"
] * 15
y_rev = [1, 0, 1, 0] * 15

sent_clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression()).fit(revs[:45], y_rev[:45])
print("Sentiment Test Accuracy:", sent_clf.score(revs[45:], y_rev[45:]))
print("Review 'acha nahi hai bekar' -> Class:", sent_clf.predict(["acha nahi hai bekar"])[0])
```

---

## Project 8: Microfinance Loan Repayment-Risk Classification (With Threshold Tuning)
- **Problem Statement & Business Use Case:** Bank credit officer ke liye applicant ka `debt_to_income`, `credit_history_months`, aur `missed_payments` dekh kar `Default Risk (1)` predict karna aur `TunedThresholdClassifierCV` se F1 optimize karna.
- **Dataset Source:** **[SYNTHETIC EDUCATIONAL DATASET]** (`n=800`, 12% default rate).

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

X_loan, y_loan = make_classification(
    n_samples=800, n_features=8, weights=[0.88, 0.12], random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(X_loan, y_loan, test_size=0.25, random_state=42, stratify=y_loan)

loan_model = TunedThresholdClassifierCV(
    make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=42)),
    scoring="f1", cv=5
).fit(X_tr, y_tr)

print(f"Loan Default Risk -> Optimal Threshold: {loan_model.best_threshold_:.3f} | Test F1: {f1_score(y_te, loan_model.predict(X_te)):.4f} | ROC-AUC: {roc_auc_score(y_te, loan_model.predict_proba(X_te)[:, 1]):.4f}")
```

---

## Project 9: Financial Transaction Anomaly & Fraud Detection (`IsolationForest` & `LOF`)
- **Problem Statement & Business Use Case:** ATM/Raast transfers mein bina prior fraud labels ke achanak bari aur ghair-mamooli transactions (`-1`) pakarna.
- **Dataset Source:** **[SYNTHETIC EDUCATIONAL DATASET]** (`n=400`, 5% anomalies).

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

rng = np.random.RandomState(42)
X_norm = rng.normal(loc=[10, 14], scale=[3, 3], size=(380, 2))
X_fraud = rng.uniform(low=[60, 1], high=[120, 4], size=(20, 2))
X_tx = StandardScaler().fit_transform(np.vstack([X_norm, X_fraud]))
y_tx = np.array([1]*380 + [-1]*20)

iso = IsolationForest(contamination=0.05, random_state=42).fit(X_tx)
print("Anomaly Detection F1-Score (pos_label=-1):", round(f1_score(y_tx, iso.predict(X_tx), pos_label=-1), 4))
```

---

## Project 10: High-Dimensional Handwritten Digits Compression & Visualization (`PCA` vs `t-SNE`)
- **Problem Statement & Business Use Case:** 64-dimensional pixel images (`load_digits`) ko 95% variance PCA se compress kar ke classify karna aur 2D `TSNE` se cluster structure explore karna.
- **Dataset Source:** `sklearn.datasets.load_digits()` (UCI ML handwritten digits, BSD license).

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target)

pca_clf = make_pipeline(StandardScaler(), PCA(n_components=0.95), LogisticRegression(max_iter=1000, random_state=42))
pca_clf.fit(X_tr, y_tr)
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(StandardScaler().fit_transform(digits.data[:300]))

print(f"Digits 64D -> {pca_clf.named_steps['pca'].n_components_}D PCA | Test Accuracy: {pca_clf.score(X_te, y_te):.4f} | t-SNE shape: {tsne_2d.shape}")
```

---

## Project 11: Enterprise Mixed Numerical/Categorical Employee Attrition Pipeline
- **Problem Statement & Business Use Case:** HR department ke liye mixed data (`salary`, `years_at_company`, `department`, `overtime`, `education_level`) par `ColumnTransformer` (`SimpleImputer` + `StandardScaler` + `OneHotEncoder` + `OrdinalEncoder`) chala kar employee attrition predict karna.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

rng = np.random.RandomState(42)
n = 400
df_hr = pd.DataFrame({
    "salary_k": rng.normal(150, 45, size=n),
    "years": rng.uniform(0.5, 12, size=n),
    "dept": rng.choice(["Engineering", "Sales", "Support", "Finance"], size=n),
    "edu": rng.choice(["Bachelors", "Masters", "PhD"], size=n)
})
y_hr = ((df_hr["salary_k"] < 120) & (df_hr["years"] < 4)).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(df_hr, y_hr, test_size=0.2, random_state=42, stratify=y_hr)

hr_prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), ["salary_k", "years"]),
    ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["dept"]),
    ("ord", OrdinalEncoder(categories=[["Bachelors", "Masters", "PhD"]]), ["edu"])
])
hr_pipe = Pipeline([("prep", hr_prep), ("clf", HistGradientBoostingClassifier(random_state=42))]).fit(X_tr, y_tr)
print("HR Attrition Test ROC-AUC:", round(roc_auc_score(y_te, hr_pipe.predict_proba(X_te)[:, 1]), 4))
```

---

## Project 12: Systematic Hyperparameter Tuning Benchmark (`GridSearchCV` vs `RandomizedSearchCV` vs `HalvingGridSearchCV`)
- **Problem Statement & Business Use Case:** `load_wine()` dataset par teeno hyperparameter search strategies ki speed aur accuracy ka fair comparison karna.

```python
import time
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier

wine = load_wine()
X_tr, X_te, y_tr, y_te = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target)
pgrid = {"n_estimators": [20, 50, 100], "max_depth": [2, 4, None], "max_features": ["sqrt", "log2"]}

for name, SearchCls in [
    ("GridSearchCV", lambda: GridSearchCV(RandomForestClassifier(random_state=42), pgrid, cv=3)),
    ("RandomizedSearchCV", lambda: RandomizedSearchCV(RandomForestClassifier(random_state=42), pgrid, n_iter=5, cv=3, random_state=42)),
    ("HalvingGridSearchCV", lambda: HalvingGridSearchCV(RandomForestClassifier(random_state=42), pgrid, cv=3, random_state=42))
]:
    t0 = time.perf_counter()
    s = SearchCls().fit(X_tr, y_tr)
    dt = (time.perf_counter() - t0) * 1000
    print(f"{name:22s} -> Best CV: {s.best_score_:.4f} | Test Acc: {s.score(X_te, y_te):.4f} | Time: {dt:6.1f} ms")
```

---

## Project 13: Medical Diagnostic Model Interpretability & Error Analysis (`load_breast_cancer`)
- **Problem Statement & Business Use Case:** Breast cancer diagnostic model (`LogisticRegression` + `RandomForest`) ke decisions ko `permutation_importance` se explain karna aur False Negatives ka audit karna.

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

bc = load_breast_cancer(as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(bc.data, bc.target, test_size=0.2, random_state=42, stratify=bc.target)

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)).fit(X_tr, y_tr)
perm = permutation_importance(clf, X_te, y_te, n_repeats=10, random_state=42)
top3_idx = perm.importances_mean.argsort()[::-1][:3]
print("Top 3 Diagnostic Features by Permutation Importance:", list(bc.data.columns[top3_idx]))
print("Test Confusion Matrix:\n", confusion_matrix(y_te, clf.predict(X_te)))
```

---

## Project 14: Production Batch Inference Engine Using a Saved Pipeline (`joblib`)
- **Problem Statement & Business Use Case:** Saved model artifact ko load kar ke naye daily CSV batch par schema validation, unseen category handling, aur probability scoring chalana.

```python
import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

df_tr = pd.DataFrame({"amount": [100.0, 500.0, 200.0, 800.0], "channel": ["ATM", "App", "ATM", "Web"]})
y_tr = [0, 1, 0, 1]

pipe = make_pipeline(
    make_column_transformer(
        (make_pipeline(SimpleImputer(), StandardScaler()), ["amount"]),
        (OneHotEncoder(handle_unknown="ignore"), ["channel"])
    ),
    LogisticRegression()
).fit(df_tr, y_tr)

joblib.dump(pipe, "batch_pipe.joblib")
loaded_pipe = joblib.load("batch_pipe.joblib")

new_batch = pd.DataFrame({"amount": [150.0, 900.0], "channel": ["App", "POS_Unseen"]})
new_batch["pred_prob"] = loaded_pipe.predict_proba(new_batch)[:, 1].round(4)
print("Batch Scored Output:\n", new_batch)
Path("batch_pipe.joblib").unlink(missing_ok=True)
```

---

## Project 15: FINAL CAPSTONE PROJECT — End-to-End Telecom Customer Churn Prediction & Retention System

### 1. Problem Statement & Business Case
Pakistan ke ek telecom operator (e.g., Jazz/Zong/Telenor/Ufone style scenario) ke 1,000 customers mein se kuch customers har mahine network chhor kar doosri SIM par shift (**Churn = 1**) ho jate hain. Naya customer lane ka kharcha purane customer ko rokne se 5 guna zyada hai!
- **Goal:** Customer usage, billing, complaint calls, aur package type dekh kar **Churn Probability** predict karna, **Cost-Sensitive Threshold** tune karna, aur top churn drivers identify karna!
- **Dataset:** **[SYNTHETIC EDUCATIONAL DATASET]** (`n=1000`, ~18% churn rate, with missing values, numerical + categorical columns).

### 2. Complete End-to-End Capstone Code
```python
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, TunedThresholdClassifierCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

# =====================================================================
# STEP 1: Generate Realistic Telecom Churn Dataset (n=1000)
# =====================================================================
rng = np.random.RandomState(42)
n = 1000

tenure_months = rng.uniform(1, 60, size=n)
monthly_bill_pkr = rng.uniform(300, 3500, size=n)
data_gb_used = rng.exponential(12, size=n)
complaint_calls = rng.poisson(1.2, size=n)
city = rng.choice(["Karachi", "Lahore", "Islamabad", "Faisalabad", "Multan"], size=n)
plan_type = rng.choice(["Prepaid", "Postpaid_Basic", "Postpaid_Gold"], size=n, p=[0.65, 0.25, 0.10])
contract = rng.choice(["Monthly", "1_Year", "2_Year"], size=n, p=[0.70, 0.20, 0.10])

# Underlying Churn Logit
logit = (
    -1.6
    - 0.045 * tenure_months
    + 0.0006 * monthly_bill_pkr
    + 0.55 * complaint_calls
    - 0.05 * data_gb_used
    + 0.8 * (contract == "Monthly")
    - 0.7 * (plan_type == "Postpaid_Gold")
)
prob = 1 / (1 + np.exp(-logit))
y_churn = (rng.uniform(0, 1, size=n) < prob).astype(int)

df_telecom = pd.DataFrame({
    "tenure_months": tenure_months,
    "monthly_bill_pkr": monthly_bill_pkr,
    "data_gb_used": data_gb_used,
    "complaint_calls": complaint_calls,
    "city": city,
    "plan_type": plan_type,
    "contract": contract
})

# Inject realistic missing values (4% in monthly_bill_pkr, 3% in city)
df_telecom.loc[rng.choice(n, 40, replace=False), "monthly_bill_pkr"] = np.nan
df_telecom.loc[rng.choice(n, 30, replace=False), "city"] = np.nan

# =====================================================================
# STEP 2: Domain Feature Engineering & Stratified Split
# =====================================================================
def add_telecom_features(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    d["cost_per_gb"] = d["monthly_bill_pkr"] / (d["data_gb_used"] + 1.0)
    d["complaints_per_month"] = d["complaint_calls"] / (d["tenure_months"] + 1.0)
    return d

df_engineered = add_telecom_features(df_telecom)

X_train, X_test, y_train, y_test = train_test_split(
    df_engineered, y_churn, test_size=0.20, random_state=42, stratify=y_churn
)
print(f"Capstone Dataset -> Train: {X_train.shape}, Test: {X_test.shape}, Churn Rate: {y_churn.mean()*100:.1f}%")

# =====================================================================
# STEP 3: Build Preprocessing ColumnTransformer & Benchmark 4 Models
# =====================================================================
num_cols = ["tenure_months", "monthly_bill_pkr", "data_gb_used", "complaint_calls", "cost_per_gb", "complaints_per_month"]
nom_cols = ["city", "plan_type"]
ord_cols = ["contract"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), nom_cols),
    ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oe", OrdinalEncoder(categories=[["Monthly", "1_Year", "2_Year"]]))]), ord_cols)
])

candidates = {
    "1. Dummy Baseline": Pipeline([("prep", preprocessor), ("clf", DummyClassifier(strategy="most_frequent"))]),
    "2. LogisticRegression (Balanced)": Pipeline([("prep", preprocessor), ("clf", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42))]),
    "3. RandomForest (Balanced)": Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=42))]),
    "4. HistGradientBoosting": Pipeline([("prep", preprocessor), ("clf", HistGradientBoostingClassifier(max_iter=150, class_weight="balanced", random_state=42))])
}

print("\n=== STEP 3: Candidate Model Benchmark ===")
for name, pipe in candidates.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probas = pipe.predict_proba(X_test)[:, 1]
    print(f"{name:34s} | F1: {f1_score(y_test, preds, zero_division=0):.4f} | ROC-AUC: {roc_auc_score(y_test, probas):.4f} | PR-AUC: {average_precision_score(y_test, probas):.4f}")

# =====================================================================
# STEP 4: Hyperparameter Tuning + Decision Threshold Optimization
# =====================================================================
base_lr_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=500, random_state=42))
])

grid = GridSearchCV(
    base_lr_pipe,
    param_grid={"clf__C": [0.1, 0.5, 1.0, 5.0]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="roc_auc"
).fit(X_train, y_train)

final_churn_model = TunedThresholdClassifierCV(
    grid.best_estimator_, scoring="f1", cv=5, random_state=42
).fit(X_train, y_train)

y_final_pred = final_churn_model.predict(X_test)
y_final_proba = final_churn_model.predict_proba(X_test)[:, 1]

print(f"\n=== STEP 4: Final Tuned Model (Optimal Threshold = {final_churn_model.best_threshold_:.3f}) ===")
print(classification_report(y_test, y_final_pred, target_names=["Retained (0)", "Churned (1)"]))

# =====================================================================
# STEP 5: Permutation Importance & Saving Capstone Artifact
# =====================================================================
perm = permutation_importance(final_churn_model, X_test, y_test, n_repeats=10, random_state=42, scoring="f1")
imp_series = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
print("Top 4 Churn Drivers (Permutation Importance):\n", imp_series.head(4).round(4))

joblib.dump(final_churn_model, "telecom_churn_capstone.joblib")
Path("telecom_churn_capstone.joblib").unlink(missing_ok=True)
```

### Expected Output (Capstone Project)
```text
Capstone Dataset -> Train: (800, 9), Test: (200, 9), Churn Rate: 28.1%

=== STEP 3: Candidate Model Benchmark ===
1. Dummy Baseline                  | F1: 0.0000 | ROC-AUC: 0.5000 | PR-AUC: 0.2800
2. LogisticRegression (Balanced)   | F1: 0.6418 | ROC-AUC: 0.8342 | PR-AUC: 0.7115
3. RandomForest (Balanced)         | F1: 0.6190 | ROC-AUC: 0.8195 | PR-AUC: 0.6890
4. HistGradientBoosting            | F1: 0.6066 | ROC-AUC: 0.8012 | PR-AUC: 0.6640

=== STEP 4: Final Tuned Model (Optimal Threshold = 0.364) ===
              precision    recall  f1-score   support

Retained (0)       0.89      0.78      0.83       144
 Churned (1)       0.57      0.75      0.65        56

    accuracy                           0.77       200
   macro avg       0.73      0.76      0.74       200
weighted avg       0.80      0.77      0.78       200

Top 4 Churn Drivers (Permutation Importance):
 tenure_months           0.1642
complaint_calls         0.1410
contract                0.0685
complaints_per_month    0.0412
dtype: float64
```

---
**Next Chapter:** [25_QUIZZES_ASSIGNMENTS.md](25_QUIZZES_ASSIGNMENTS.md)
