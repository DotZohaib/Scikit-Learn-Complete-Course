# Part 27B: Machine Learning & Scikit-learn Interview Preparation + Production Workflow Checklist

Is chapter mein Junior se Senior Machine Learning Engineer aur Data Scientist interviews mein poochay jane wale **30+ sab se ahem Scikit-learn aur ML Interview Questions & Answers** (Roman Urdu + English technical precision ke sath) aur ek **Professional ML Production Checklist** di gayi hai.

---

## 1. Fundamental & API Design Interview Questions (Q1–Q10)

### Q1. Estimator, Transformer, aur Predictor mein kya farq hai?
- **Answer:**
  - **Estimator:** Har woh Scikit-learn object jo data se parameters seekhta hai aur jismein `.fit(X, y)` method hota hai (e.g., `StandardScaler`, `RandomForestClassifier`).
  - **Transformer:** Woh estimator jo data ko transform/preprocess karta hai aur jismein `.transform(X)` aur `.fit_transform(X)` methods hote hain (e.g., `SimpleImputer`, `OneHotEncoder`, `PCA`).
  - **Predictor:** Woh estimator jo target predict karta hai aur jismein `.predict(X)` aur `.score(X, y)` methods hote hain (e.g., `LogisticRegression`, `KMeans`).

### Q2. What is Data Leakage and what are its 3 most common causes in Scikit-learn?
- **Answer:** **Data Leakage** tab hota hai jab training ke dauran Test Set ya Future ki maloomat model ya preprocessor mein shamil ho jaye, jis se validation score jhoota (overly optimistic) aata hai aur production mein model fail ho jata hai.
  1. **Preprocessing before splitting:** Poore `X` par `StandardScaler().fit_transform(X)` ya `SimpleImputer` chala kar baad mein `train_test_split` karna.
  2. **Feature selection before cross-validation:** Poore dataset par `SelectKBest` chala kar phir `cross_val_score` lagana (is ka hal `Pipeline` hai!).
  3. **Time-series random splitting / unshifted rolling windows:** Time-series par `shuffle=True` rakhna ya `.shift(1)` ke baghair rolling mean nikalna.

### Q3. Why do learned attributes in Scikit-learn end with an underscore (`_`), like `coef_` or `classes_`?
- **Answer:** Yeh Scikit-learn ka API design convention hai taake user ko saaf farq pata chale ke kaunse attributes **Hyperparameters** hain (jo user ne `__init__()` mein set kiye, jaise `C`, `max_depth`) aur kaunse **Learned Parameters** hain (jo `.fit()` chalne ke baad data se calculate hue, jaise `coef_`, `mean_`, `n_features_in_`).

### Q4. `OneHotEncoder` mein `handle_unknown='ignore'` kyun zaroori hai?
- **Answer:** Agar production ya test set mein koi aisi nayi category (maslan naya shehar `"Gwadar"`) aa jaye jo training data mein nahi thi, to default `handle_unknown='error'` par pipeline crash (`ValueError`) ho jata hai. `handle_unknown='ignore'` us naye shehar ke tamam dummy columns ko `[0, 0, ..., 0]` set kar ke pipeline ko crash hone se bachata hai.

### Q5. When should you use `LabelEncoder` vs `OrdinalEncoder` vs `OneHotEncoder`?
- **Answer:**
  - `LabelEncoder`: **Sirf 1D target vector `y`** ke liye!
  - `OrdinalEncoder`: **2D feature matrix `X`** ke un categorical columns ke liye jin mein natural order ho (`["Small", "Medium", "Large"]`).
  - `OneHotEncoder`: **2D feature matrix `X`** ke nominal categorical columns ke liye jin mein koi order na ho (`["Karachi", "Lahore", "Islamabad"]`).

---

## 2. Algorithms & Mathematical Intuition Questions (Q6–Q18)

### Q6. L1 (`Lasso`) aur L2 (`Ridge`) Regularization mein kya farq hai? L1 weights ko exact zero kyun kar deta hai?
- **Answer:**
  - **L2 (`Ridge`)** loss mein weights ke murabbe ($\alpha \sum w_j^2$) jama karta hai. Iska derivative $2\alpha w_j$ hota hai jo $w_j$ chota hone par khud chota ho jata hai, is liye weights $0$ ke qareeb jate hain magar exact $0$ nahi bante.
  - **L1 (`Lasso`)** absolute values ($\alpha \sum |w_j|$) jama karta hai. Iska sub-gradient constant $\pm \alpha$ rehta hai aur constraint diamond-shaped hota hai jiske kinaray (corners) axes par hote hain, is liye kam ahem features ke weights exact **$0.0$** ho jate hain (built-in feature selection!).

### Q7. Why do Tree-Based Models (`RandomForest`, `HistGradientBoosting`) NOT require Feature Scaling?
- **Answer:** Decision Trees har node par ek waqt mein ek hi feature ka order-based threshold (`if income > 50000`) check karte hain. Agar aap `income` ko 1000 se divide kar dein ya `StandardScaler` laga dein, tab bhi split point aur Gini impurity/MSE reduction bilkul wohi rehti hai (monotonic transformation invariant). Jabke `KNN`, `SVM`, `PCA`, aur `Ridge/Lasso` distances aur weight magnitudes par depend karte hain, is liye unhein scaling lazmi chahiye.

### Q8. Bagging (`RandomForest`) aur Boosting (`GradientBoosting` / `HistGradientBoosting`) mein kya farq hai?
- **Answer:**
  - **Bagging (`RandomForest`):** Gehray (deep) trees **parallel** mein random bootstrap samples par bante hain aur aakhir mein average/vote liya jata hai. Yeh मुख्य रूप से **Variance (Overfitting) kam** karta hai.
  - **Boosting (`GradientBoosting`):** Chote (shallow, `max_depth=3`) trees **ek ke baad ek (sequentially)** bante hain jahan har naya tree pichle trees ke **Residuals (ghaltiyon)** ko predict karta hai. Yeh मुख्य रूप से **Bias kam** karta hai.

### Q9. `HistGradientBoostingClassifier` aam `GradientBoostingClassifier` se 10x–50x tez kyun hai?
- **Answer:** Aam `GradientBoosting` har continuous column ki har unique value ko sort kar ke split dhoondta hai ($O(N \log N)$). Jabke `HistGradientBoosting` (LightGBM ki tarah) pehle continuous features ko **255 integer bins (Histograms)** mein bucket kar leta hai, jis se split finding sirf $O(255)$ reh jati hai! Mazeed yeh ke is mein native `NaN` aur categorical support mojood hai.

### Q10. What is the "Kernel Trick" in `SVC`? And why shouldn't you use `SVC(kernel='rbf')` on 500,000 rows?
- **Answer:** Kernel Trick data ko asal mein high-dimensional coordinates mein transform kiye baghair sirf pairwise inner products $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ calculate kar ke non-linear boundaries bana leta hai. Lekin is ke liye $N \times N$ Gram matrix banana parta hai jiski complexity $O(N^2)$ se $O(N^3)$ hoti hai—500,000 rows par yeh سینکڑوں GB RAM maang kar freeze ho jayega! Bare data par `Nystroem` + `SGDClassifier` ya `HistGradientBoosting` use karein.

### Q11. What is the difference between `LinearSVC` and `SVC(kernel='linear')`?
- **Answer:** `LinearSVC` `liblinear` library par bana hai jo linear classification ke liye optimize hai aur bare datasets par $O(N)$ mein tez chalta hai (aur default taur par intercept ko भी regularize karta hai aur `.predict_proba()` nahi deta), jabke `SVC(kernel='linear')` `libsvm` solver use karta hai jo $O(N^2)$ scale karta hai.

### Q12. When should you use `PCA` vs `TruncatedSVD` vs `TSNE`?
- **Answer:**
  - `PCA`: Dense tabular numerical data (scaled with `StandardScaler`) ki preprocessing/compression ke liye.
  - `TruncatedSVD`: **Sparse matrices** (jaise `TfidfVectorizer` text counts) ke liye kyunki yeh data ko center kiye baghair seedha SVD lagata hai.
  - `TSNE`: **Sirf 2D/3D Visualization** ke liye! Is mein `.transform(X_test)` nahi hota, is liye isay ML pipeline mein preprocessor ke taur par use nahi kar sakte.

---

## 3. Evaluation, Imbalance, Time Series & Production Questions (Q13–Q22)

### Q13. ROC-AUC vs Precision-Recall AUC (`average_precision_score`) — Imbalanced data par kaunsa behtar hai?
- **Answer:** Jab dataset bohot imbalanced ho (maslan `99% Negative (TN)` aur `1% Positive (TP)`), to ROC curve ka X-axis $\text{FPR} = \frac{FP}{FP + TN}$ hota hai. Kyunki $TN$ bohot bara (`9900`) hai, is liye $100$ False Positives aane par bhi $\text{FPR}$ sirf `0.01` rehta hai aur ROC-AUC `0.98` dikhata hai! Jabke **Precision** ($\frac{TP}{TP + FP}$) mein $TN$ shamil nahi hota, woh foran pakar leta hai ke False Positives barh gaye hain. Is liye rare events mein **PR-AUC (`average_precision_score`)** hamesha zyada sacha metric hai!

### Q14. `cross_val_score` aur `cross_val_predict` mein kya farq hai?
- **Answer:** `cross_val_score` har fold par metric calculate kar ke $K$ scores return karta hai. `cross_val_predict` har sample ke liye out-of-fold prediction return karta hai (jo residual visualization ya model stacking ke kaam aata hai).

### Q15. Probability Calibration (`CalibratedClassifierCV`) kya hai aur kab zaroori hoti hai?
- **Answer:** Agar model kisi marij ke liye `predict_proba = 0.80` de, to calibrated model ka matlab hai ke aise 100 marijon mein se waqai ~80 marij positive honge. `RandomForest`, `NaiveBayes`, aur `SVM` ke scores calibrated nahi hote. Jab probabilities ko seedha expected financial cost ($\text{Prob of Default} \times \text{Loan Amount}$) ya medical risk decision mein use karna ho, to `CalibratedClassifierCV` zaroori hota hai.

### Q16. Why should you NEVER load an untrusted `.pkl` or `.joblib` file?
- **Answer:** Python ka `pickle` protocol (jise `joblib` bhi andar se use karta hai) deserialization ke waqt `__reduce__` method ke zariye arbitrary shell commands ya malicious code execute kar sakta hai. Sirf apne banaye hue aur hash-verified `.joblib` models load karein, ya untrusted model sharing ke liye `ONNX` / `skops` use karein.

### Q17. What is the difference between Data Drift and Concept Drift in production?
- **Answer:**
  - **Data Drift (Covariate Shift):** Input features $P(X)$ ka distribution badal gaya (maslan inflation ki wajah se customers ki average salary aur bill barh gaye).
  - **Concept Drift:** Features aur Target ka talluq $P(y \mid X)$ hi badal gaya (maslan naye tax qanoon ya pandemic ke baad wohi customer profile ab mukhtalif faisla le rahi hai).

### Q18. How do you tune a threshold in Scikit-learn without leaking test data?
- **Answer:** `TunedThresholdClassifierCV(estimator, scoring='f1', cv=5)` ko `X_train, y_train` par fit kar ke! Yeh internal cross-validation folds par out-of-fold probabilities nikal kar optimal threshold (`best_threshold_`) chunta hai.

---

## 4. Professional Machine Learning Workflow Checklist (Production Sign-Off)

Apne kisi bhi real-world project ko complete ya deploy karne se pehle is **15-Point Professional Checklist** par tick lagayein:

- [ ] **1. Problem & Metric Alignment:** Kya business goal ke mutabiq sahi metric (`PR-AUC`, `F1`, `RMSE`, `MAE`, ya custom cost scorer) chuna gaya hai?
- [ ] **2. Leak-Free Splitting First:** Kya koi bhi imputer, scaler, encoder, ya feature selector lagane se **pehle** `train_test_split` kiya gaya hai?
- [ ] **3. Domain-Appropriate Splitter:** Classification ke liye `StratifiedKFold`, grouped records ke liye `GroupKFold`, aur temporal data ke liye `TimeSeriesSplit` (`shuffle=False`) use hua hai?
- [ ] **4. Baseline Comparison:** Kya `DummyClassifier` ya `DummyRegressor` chala kar confirm kiya gaya hai ke hamara ML model naive baseline se waqai behtar hai?
- [ ] **5. Unified `Pipeline` & `ColumnTransformer`:** Kya tamam preprocessing steps (`SimpleImputer`, `OneHotEncoder`, `StandardScaler`) aur estimator ek single `Pipeline` mein bandhe hue hain?
- [ ] **6. Unseen Category & NaN Safety:** Kya `OneHotEncoder(handle_unknown='ignore')` aur `SimpleImputer` laga hai taake production mein naya shehar ya missing value aane par server crash na ho?
- [ ] **7. Feature Scaling Check:** Kya linear, SVM, KNN, MLP, aur PCA models ke liye `StandardScaler` / `RobustScaler` lagaya gaya hai?
- [ ] **8. Time-Series Lag Safety:** Agar rolling window features hain, to kya `.rolling()` se pehle `.shift(1)` lagaya gaya hai?
- [ ] **9. Hyperparameter Tuning on Train Only:** Kya `GridSearchCV` / `RandomizedSearchCV` sirf `X_train, y_train` par chalaya gaya hai aur `X_test` ko bilkul aakhir mein sirf ek dafa dekha gaya hai?
- [ ] **10. Threshold & Calibration Audit:** Imbalanced classification mein kya `TunedThresholdClassifierCV` ya `class_weight='balanced'` check kiya gaya hai?
- [ ] **11. Interpretability & Sanity Check:** Kya `permutation_importance` chala kar verify kiya gaya hai ke top features domain sense banate hain (aur koi ID column ya target leakage feature top par nahi hai)?
- [ ] **12. Subgroup & Slice Error Analysis:** Kya Confusion Matrix aur worst residuals dekh kar check kiya gaya hai ke model kisi specific shehar ya subgroup par zyada ghalti to nahi kar raha?
- [ ] **13. Reproducibility:** Kya tamam random splitters aur estimators mein `random_state=42` fix kiya gaya hai?
- [ ] **14. Versioned Artifact Bundling:** Kya `joblib.dump()` ke sath `sklearn.__version__`, `expected_columns`, aur training summary stats save kiye gaye hain?
- [ ] **15. Security & Model Card:** Kya untrusted pickle loading se bachaao aur `MODEL_CARD` documentation mukammal hai?

---
**Next Resource:** [GLOSSARY.md](GLOSSARY.md) | [CHEAT_SHEET.md](CHEAT_SHEET.md) | [REFERENCES.md](REFERENCES.md)
