# Scikit-learn Ultimate Quick Reference Cheat Sheet (Roman Urdu & Python)

Yeh Cheat Sheet aap ke roz-marra projects, exams, aur interviews mein **1-minute quick lookup** ke liye tayyar ki gayi hai!

---

## 1. Essential Imports Cheat Sheet

```python
# 1. Data Loading & Splitting
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, make_classification
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GroupKFold, TimeSeriesSplit,
    cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV,
    TunedThresholdClassifierCV, FixedThresholdClassifier
)

# 2. Preprocessing, Imputation & Pipelines
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
    OneHotEncoder, OrdinalEncoder, LabelEncoder,
    PolynomialFeatures, SplineTransformer, FunctionTransformer
)
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector, TransformedTargetRegressor
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion

# 3. Feature Selection, Dimensionality Reduction & Text
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif, SelectFromModel, RFECV
from sklearn.decomposition import PCA, TruncatedSVD, NMF, FastICA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

# 4. Supervised Classifiers & Regressors
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor, HuberRegressor, RANSACRegressor, QuantileRegressor
)
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, VotingClassifier, StackingClassifier, IsolationForest
)
from sklearn.neural_network import MLPClassifier, MLPRegressor

# 5. Unsupervised Clustering & Calibration/Inspection/Metrics
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score, log_loss, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, silhouette_score
)
import joblib
```

---

## 2. The Golden Leak-Free Production Pipeline Template (Copy-Ready!)

```python
# Step 1: Split FIRST
X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Build ColumnTransformer for Mixed Data
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])

prep = ColumnTransformer([
    ("num", num_pipe, make_column_selector(dtype_include="number")),
    ("cat", cat_pipe, make_column_selector(dtype_include=object))
])

# Step 3: Full Pipeline + GridSearchCV on X_train ONLY
pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
grid = GridSearchCV(pipe, {"clf__C": [0.1, 1.0, 10.0]}, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)

# Step 4: Final Evaluation & Persistence
print("Test Score:", grid.score(X_test, y_test))
joblib.dump(grid.best_estimator_, "production_pipeline.joblib")
```

---

## 3. Quick Algorithm Selector Matrix

| Problem Type | Data Characteristics | First Baseline | Best Overall / Production Choice |
| :--- | :--- | :--- | :--- |
| **Binary / Multiclass Classification** | Medium/Large Tabular Data | `LogisticRegression` (scaled) | `HistGradientBoostingClassifier` or `RandomForestClassifier` |
| **Text / NLP Classification** | Sparse TF-IDF / Word Counts | `ComplementNB` or `MultinomialNB` | `LinearSVC` or `LogisticRegression` |
| **Regression (Continuous $y$)** | Clean Tabular Data | `Ridge` (scaled) | `HistGradientBoostingRegressor` or `RandomForestRegressor` |
| **Regression with Outliers** | Extreme noisy targets in $y$ | `HuberRegressor` | `RANSACRegressor` or `QuantileRegressor(quantile=0.5)` |
| **Clustering (Unsupervised)** | Round vs Arbitrary Shapes | `KMeans(n_init=10)` | `HDBSCAN` / `DBSCAN` (arbitrary shape + noise `-1`) or `GaussianMixture` |
| **Dimensionality Reduction** | Dense vs Sparse vs Visual | `PCA(n_components=0.95)` | `TruncatedSVD` (for sparse) / `TSNE` (for 2D plots only) |
| **Anomaly / Fraud Detection** | Unlabeled Outliers | `IsolationForest` | `LocalOutlierFactor(novelty=True)` or `OneClassSVM` |
| **Huge Dataset ($>10^6$ rows)** | Out-of-Core Streaming | `SGDClassifier` / `SGDRegressor` | `MiniBatchKMeans` / `IncrementalPCA` via `.partial_fit()` |

---

## 4. Which Algorithms Need Feature Scaling (`StandardScaler` / `RobustScaler`)?

- **YES — Scaling is 100% MANDATORY:**
  - `LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SGDClassifier`, `SGDRegressor`
  - `SVC`, `LinearSVC`, `SVR`, `LinearSVR`, `OneClassSVM`
  - `KNeighborsClassifier`, `KNeighborsRegressor`, `LocalOutlierFactor`
  - `MLPClassifier`, `MLPRegressor` (Neural Networks)
  - `KMeans`, `MiniBatchKMeans`, `DBSCAN`, `HDBSCAN`, `AgglomerativeClustering`, `GaussianMixture`
  - `PCA`, `KernelPCA`, `Isomap`, `TSNE`
- **NO — Scaling is NOT Needed (Scale-Invariant):**
  - `DecisionTreeClassifier`, `DecisionTreeRegressor`
  - `RandomForestClassifier`, `RandomForestRegressor`
  - `ExtraTreesClassifier`, `ExtraTreesRegressor`
  - `GradientBoostingClassifier`, `GradientBoostingRegressor`
  - `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`
  - `GaussianNB`, `MultinomialNB`, `ComplementNB`, `BernoulliNB`
