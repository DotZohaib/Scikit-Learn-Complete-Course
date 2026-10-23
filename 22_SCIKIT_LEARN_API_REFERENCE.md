# Part 24: Scikit-learn API and Estimator Ecosystem (`sklearn.*` Complete Reference)

Scikit-learn mein 31 se zyada specialized submodules hain ([Official Scikit-learn API Reference](https://scikit-learn.org/stable/api/index.html)). Ek professional ML Engineer ko pata hona chahiye ke kaunsa tool kis submodule mein rehta hai, naye estimators kaise dhoonde jate hain (`all_estimators`), aur official documentation mein deprecation warnings aur version notes kaise parhe jate hain.

---

## 1. Complete Guide to All 31 Scikit-learn Submodules

| # | Submodule Name | Primary Purpose (Roman Urdu & English) | Major Classes / Functions | Official Docs Link |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **`sklearn.base`** | Custom Estimators aur Transformers banane ki base classes aur `clone()`. | `BaseEstimator`, `TransformerMixin`, `ClassifierMixin`, `RegressorMixin`, `clone` | [sklearn.base](https://scikit-learn.org/stable/api/sklearn.base.html) |
| 2 | **`sklearn.datasets`** | Built-in toy datasets, OpenML/real-world downloaders, aur synthetic generators. | `load_iris`, `load_breast_cancer`, `fetch_openml`, `make_classification`, `make_regression` | [sklearn.datasets](https://scikit-learn.org/stable/api/sklearn.datasets.html) |
| 3 | **`sklearn.model_selection`** | Train/test splitting, Cross-Validation, Hyperparameter tuning, aur Threshold tuning. | `train_test_split`, `StratifiedKFold`, `TimeSeriesSplit`, `GridSearchCV`, `TunedThresholdClassifierCV` | [sklearn.model_selection](https://scikit-learn.org/stable/api/sklearn.model_selection.html) |
| 4 | **`sklearn.preprocessing`** | Feature scaling, normalization, encoding, aur non-linear polynomial/spline features. | `StandardScaler`, `RobustScaler`, `OneHotEncoder`, `OrdinalEncoder`, `PolynomialFeatures`, `SplineTransformer` | [sklearn.preprocessing](https://scikit-learn.org/stable/api/sklearn.preprocessing.html) |
| 5 | **`sklearn.impute`** | Missing values (`NaN`) bharne wale transformers. | `SimpleImputer`, `KNNImputer`, `MissingIndicator`, `IterativeImputer` *(experimental)* | [sklearn.impute](https://scikit-learn.org/stable/api/sklearn.impute.html) |
| 6 | **`sklearn.compose`** | Mixed column types (`ColumnTransformer`) aur target transformation (`TransformedTargetRegressor`). | `ColumnTransformer`, `make_column_transformer`, `make_column_selector`, `TransformedTargetRegressor` | [sklearn.compose](https://scikit-learn.org/stable/api/sklearn.compose.html) |
| 7 | **`sklearn.pipeline`** | Preprocessing aur model steps ko ek leakage-free zanjeer mein jorna. | `Pipeline`, `make_pipeline`, `FeatureUnion`, `make_union` | [sklearn.pipeline](https://scikit-learn.org/stable/api/sklearn.pipeline.html) |
| 8 | **`sklearn.feature_selection`** | Sab se ahem columns chunna aur bekar columns hatana. | `VarianceThreshold`, `SelectKBest`, `SelectFromModel`, `RFE`, `RFECV`, `SequentialFeatureSelector` | [sklearn.feature_selection](https://scikit-learn.org/stable/api/sklearn.feature_selection.html) |
| 9 | **`sklearn.feature_extraction`** | Text aur Dictionaries se numerical matrices banana (`text`, `image`). | `CountVectorizer`, `TfidfVectorizer`, `HashingVectorizer`, `DictVectorizer` | [sklearn.feature_extraction](https://scikit-learn.org/stable/api/sklearn.feature_extraction.html) |
| 10 | **`sklearn.linear_model`** | Linear & Logistic models, L1/L2 regularization, SGD, aur Robust regressors. | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression`, `SGDClassifier`, `HuberRegressor`, `RANSACRegressor`, `QuantileRegressor` | [sklearn.linear_model](https://scikit-learn.org/stable/api/sklearn.linear_model.html) |
| 11 | **`sklearn.neighbors`** | Nearest Neighbors classification, regression, density, aur LOF anomaly detection. | `KNeighborsClassifier`, `KNeighborsRegressor`, `NearestNeighbors`, `LocalOutlierFactor`, `KernelDensity` | [sklearn.neighbors](https://scikit-learn.org/stable/api/sklearn.neighbors.html) |
| 12 | **`sklearn.svm`** | Support Vector Machines for classification, regression, and novelty detection. | `SVC`, `LinearSVC`, `SVR`, `LinearSVR`, `OneClassSVM` | [sklearn.svm](https://scikit-learn.org/stable/api/sklearn.svm.html) |
| 13 | **`sklearn.tree`** | Decision Trees aur tree visualization (`plot_tree`, `export_text`). | `DecisionTreeClassifier`, `DecisionTreeRegressor`, `ExtraTreeClassifier`, `plot_tree`, `export_text` | [sklearn.tree](https://scikit-learn.org/stable/api/sklearn.tree.html) |
| 14 | **`sklearn.ensemble`** | Bagging, Random Forests, Gradient Boosting, HistGradientBoosting, Voting, Stacking, IsolationForest. | `RandomForestClassifier`, `HistGradientBoostingClassifier`, `GradientBoostingRegressor`, `VotingClassifier`, `StackingClassifier`, `IsolationForest` | [sklearn.ensemble](https://scikit-learn.org/stable/api/sklearn.ensemble.html) |
| 15 | **`sklearn.naive_bayes`** | Probabilistic Naive Bayes classifiers for tabular, count, and text data. | `GaussianNB`, `MultinomialNB`, `ComplementNB`, `BernoulliNB`, `CategoricalNB` | [sklearn.naive_bayes](https://scikit-learn.org/stable/api/sklearn.naive_bayes.html) |
| 16 | **`sklearn.discriminant_analysis`** | Linear aur Quadratic Discriminant Analysis classifiers & dimensionality reduction. | `LinearDiscriminantAnalysis` (LDA), `QuadraticDiscriminantAnalysis` (QDA) | [sklearn.discriminant_analysis](https://scikit-learn.org/stable/api/sklearn.discriminant_analysis.html) |
| 17 | **`sklearn.neural_network`** | Multi-Layer Perceptron (MLP) feedforward neural networks. | `MLPClassifier`, `MLPRegressor`, `BernoulliRBM` | [sklearn.neural_network](https://scikit-learn.org/stable/api/sklearn.neural_network.html) |
| 18 | **`sklearn.cluster`** | Unsupervised clustering algorithms. | `KMeans`, `MiniBatchKMeans`, `DBSCAN`, `HDBSCAN`, `AgglomerativeClustering`, `SpectralClustering`, `FeatureAgglomeration` | [sklearn.cluster](https://scikit-learn.org/stable/api/sklearn.cluster.html) |
| 19 | **`sklearn.mixture`** | Gaussian Mixture Models for soft clustering & density estimation. | `GaussianMixture`, `BayesianGaussianMixture` | [sklearn.mixture](https://scikit-learn.org/stable/api/sklearn.mixture.html) |
| 20 | **`sklearn.decomposition`** | Matrix factorization aur linear/kernel dimensionality reduction. | `PCA`, `IncrementalPCA`, `KernelPCA`, `TruncatedSVD`, `NMF`, `FastICA`, `FactorAnalysis` | [sklearn.decomposition](https://scikit-learn.org/stable/api/sklearn.decomposition.html) |
| 21 | **`sklearn.manifold`** | Non-linear manifold learning aur high-dimensional visualization. | `TSNE`, `Isomap`, `LocallyLinearEmbedding`, `MDS`, `SpectralEmbedding` | [sklearn.manifold](https://scikit-learn.org/stable/api/sklearn.manifold.html) |
| 22 | **`sklearn.metrics`** | Classification, Regression, Clustering, aur Ranking evaluation metrics & Displays. | `accuracy_score`, `f1_score`, `roc_auc_score`, `root_mean_squared_error`, `silhouette_score`, `ConfusionMatrixDisplay` | [sklearn.metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html) |
| 23 | **`sklearn.inspection`** | Model interpretability aur black-box explanation tools. | `permutation_importance`, `PartialDependenceDisplay`, `DecisionBoundaryDisplay` | [sklearn.inspection](https://scikit-learn.org/stable/api/sklearn.inspection.html) |
| 24 | **`sklearn.semi_supervised`** | Labeled + Unlabeled (`-1`) data par semi-supervised learning. | `SelfTrainingClassifier`, `LabelPropagation`, `LabelSpreading` | [sklearn.semi_supervised](https://scikit-learn.org/stable/api/sklearn.semi_supervised.html) |
| 25 | **`sklearn.covariance`** | Robust covariance estimation aur Gaussian outlier detection. | `EllipticEnvelope`, `EmpiricalCovariance`, `LedoitWolf`, `MinCovDet` | [sklearn.covariance](https://scikit-learn.org/stable/api/sklearn.covariance.html) |
| 26 | **`sklearn.calibration`** | Classifier probabilities ko true confidence mein calibrate karna. | `CalibratedClassifierCV`, `calibration_curve`, `CalibrationDisplay` | [sklearn.calibration](https://scikit-learn.org/stable/api/sklearn.calibration.html) |
| 27 | **`sklearn.multiclass`** | Binary classifiers ko multiclass/multilabel banana. | `OneVsRestClassifier`, `OneVsOneClassifier`, `OutputCodeClassifier` | [sklearn.multiclass](https://scikit-learn.org/stable/api/sklearn.multiclass.html) |
| 28 | **`sklearn.multioutput`** | Multiple targets ek sath predict karna. | `MultiOutputClassifier`, `MultiOutputRegressor`, `ClassifierChain`, `RegressorChain` | [sklearn.multioutput](https://scikit-learn.org/stable/api/sklearn.multioutput.html) |
| 29 | **`sklearn.dummy`** | Sanity-check baseline models (bina pattern seekhe). | `DummyClassifier`, `DummyRegressor` | [sklearn.dummy](https://scikit-learn.org/stable/api/sklearn.dummy.html) |
| 30 | **`sklearn.utils`** | Developer utilities, resampling, class weights, estimator discovery. | `resample`, `shuffle`, `compute_class_weight`, `discovery.all_estimators`, `check_X_y` | [sklearn.utils](https://scikit-learn.org/stable/api/sklearn.utils.html) |
| 31 | **`sklearn.exceptions`** | Scikit-learn specific warnings aur errors. | `NotFittedError`, `ConvergenceWarning`, `DataConversionWarning`, `UndefinedMetricWarning`, `InconsistentVersionWarning` | [sklearn.exceptions](https://scikit-learn.org/stable/api/sklearn.exceptions.html) |

---

## 2. How to Read Official Scikit-learn Documentation Like a Pro
Jab aap [Scikit-learn API Reference](https://scikit-learn.org/stable/api/index.html) mein kisi class (maslan `RandomForestClassifier`) ka page kholte hain, to wahan 5 hissay hote hain:
1. **Class Signature & Default Values:** Sab se upar default hyperparameters dikhte hain (`n_estimators=100, criterion='gini'`).
2. **Parameters Section:** Woh inputs jo aap `__init__()` mein pass karte hain. Har parameter ke sath uski data type (`int`, `float`, `{'gini', 'entropy'}`) aur **`Added in version 1.x` / `Deprecated since version 1.x`** ke badges lage hote hain!
3. **Attributes Section (Ending in `_`):** Woh variables jo `.fit()` chalne ke baad paida hote hain (`classes_`, `feature_importances_`).
4. **Methods Table:** `.fit()`, `.predict()`, `.predict_proba()`, `.set_output()` waghaira.
5. **Examples at the Bottom:** Har page ke bilkul neeche copy-paste runnable code aur gallery links hote hain!

---

## 3. Complete Runnable Code: Discovering Estimators Programmatically with `all_estimators` & `clone`

```python
from sklearn.utils.discovery import all_estimators, all_displays, all_functions
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

# 1. Programmatically count all Classifiers, Regressors, Clusters & Transformers in installed sklearn!
classifiers = all_estimators(type_filter="classifier")
regressors = all_estimators(type_filter="regressor")
clusters = all_estimators(type_filter="cluster")
transformers = all_estimators(type_filter="transformer")

print("=== 1. Installed Scikit-learn Estimator Ecosystem Inventory ===")
print(f"Total Classifiers  : {len(classifiers)}")
print(f"Total Regressors   : {len(regressors)}")
print(f"Total Clusterers   : {len(clusters)}")
print(f"Total Transformers : {len(transformers)}")
print(f"Total Display Tools: {len(all_displays())}")

print("\nFirst 5 Classifiers:", [name for name, _ in classifiers[:5]])
print("First 5 Regressors :", [name for name, _ in regressors[:5]])

# 2. Using sklearn.base.clone() to copy an estimator's hyperparameters WITHOUT copying fitted weights
orig_clf = LogisticRegression(C=0.5, max_iter=300, random_state=42)
orig_clf.fit([[1, 2], [3, 4], [5, 6]], [0, 1, 1])

fresh_copy = clone(orig_clf)
print("\n=== 2. Cloning an Estimator with sklearn.base.clone() ===")
print("Original Fitted has classes_? :", hasattr(orig_clf, "classes_"))
print("Cloned Copy has classes_?     :", hasattr(fresh_copy, "classes_"))
print("Cloned Copy kept C=0.5?       :", fresh_copy.get_params()["C"])
```

### Expected Output
```text
=== 1. Installed Scikit-learn Estimator Ecosystem Inventory ===
Total Classifiers  : 44
Total Regressors   : 56
Total Clusterers   : 11
Total Transformers : 85
Total Display Tools: 7

First 5 Classifiers: ['AdaBoostClassifier', 'BaggingClassifier', 'BernoulliNB', 'CalibratedClassifierCV', 'CategoricalNB']
First 5 Regressors : ['ARDRegression', 'AdaBoostRegressor', 'BaggingRegressor', 'BayesianRidge', 'CCA']

=== 2. Cloning an Estimator with sklearn.base.clone() ===
Original Fitted has classes_? : True
Cloned Copy has classes_?     : False
Cloned Copy kept C=0.5?       : 0.5
```

---

## 4. Practice Exercises & Solutions

### Exercises
1. **Beginner:** `sklearn.utils.class_weight.compute_class_weight` use kar ke `y = np.array([0, 0, 0, 0, 1])` ke liye `'balanced'` weights calculate karein.
2. **Intermediate:** `sklearn.tree.export_text` use kar ke Iris dataset par fit kiye gaye `DecisionTreeClassifier(max_depth=2)` ke `if/else` rules text format mein print karein!
3. **Challenge:** `all_estimators(type_filter="classifier")` par loop chala kar woh tamam classifiers dhoondein jin ke andar `.partial_fit` method mojood hai!

### Solutions
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.discovery import all_estimators

# 1. Beginner
y_imb = np.array([0, 0, 0, 0, 1])
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_imb)
print("Balanced Weights for [0, 1]:", weights)  # [0.625, 2.5]

# 2. Intermediate
iris = load_iris()
dt = DecisionTreeClassifier(max_depth=2, random_state=42).fit(iris.data, iris.target)
print("Decision Tree Rules:\n", export_text(dt, feature_names=list(iris.feature_names)))

# 3. Challenge: Find all incremental classifiers supporting .partial_fit()
online_clfs = [name for name, Cls in all_estimators(type_filter="classifier") if hasattr(Cls, "partial_fit")]
print("Classifiers with .partial_fit():", online_clfs)
```

---
**Next Chapter:** [23_COMMON_ERRORS.md](23_COMMON_ERRORS.md)
