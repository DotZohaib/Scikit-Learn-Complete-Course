# Part 7: Supervised Learning — Classification (`sklearn.dummy`, `sklearn.linear_model`, `sklearn.neighbors`, `sklearn.tree`, `sklearn.ensemble`, `sklearn.naive_bayes`, `sklearn.svm`, `sklearn.discriminant_analysis`, `sklearn.multiclass`)

Is chapter mein hum **Supervised Classification** ko zero se advanced level tak aur Scikit-learn ke **tamam 25 major classifiers** ko step-by-step Roman Urdu mein seekhenge ([Scikit-learn Supervised Learning Guide](https://scikit-learn.org/stable/supervised_learning.html)).

---

## 1. Classification Fundamentals: Binary, Multiclass, and Multilabel

### A. Topic Introduction
- **Classification Kya Hai?** Jab hamara target $y$ ek **Category (Class)** ho (maslan `Spam` vs `Not Spam`, ya `Diabetes` vs `Healthy`).
- **Teen Bari Qismein ([Multiclass & Multilabel Algorithms](https://scikit-learn.org/stable/modules/multiclass.html)):**
  1. **Binary Classification:** Sirf **2 classes** (`0` ya `1`). *Misaal:* Loan applicant `Default (1)` karega ya `Repay (0)`?
  2. **Multiclass Classification:** **2 se zyada mutually exclusive classes** (har sample ki sirf ek class ho sakti hai). *Misaal:* Iris phool `Setosa (0)`, `Versicolor (1)`, ya `Virginica (2)` mein se koi ek hoga.
  3. **Multilabel Classification:** Ek hi sample ke **ek se zyada labels** ho sakte hain (`y` ek 2D binary matrix hota hai). *Misaal:* Ek news article ek hi waqt mein `["Politics", "Economy", "Pakistan"]` teeno topics par ho sakta hai!

---

## 2. Progression of All 25 Scikit-learn Classifiers

Hum in 25 classifiers ko 7 logical families mein taqseem kar ke parhenge:

### FAMILY 1: Baseline Classifier
#### 1. `DummyClassifier` (`sklearn.dummy.DummyClassifier`)
- **What & Why:** Yeh koi asli pattern nahi seekhta! Yeh sirf ek **Baseline (Benchmark)** banane ke kaam aata hai. Agar aap ke dataset mein 90% `Healthy` log hain, to `DummyClassifier(strategy="most_frequent")` har kisi ko `Healthy` keh kar 90% accuracy le aayega. Aap ke asli ML model ko `DummyClassifier` se lazmi behtar hona chahiye!
- **Key Hyperparameter:** `strategy` (`"most_frequent"`, `"prior"`, `"stratified"`, `"uniform"`).

---

### FAMILY 2: Linear Classifiers & Discriminant Analysis
#### 2. `LogisticRegression` (`sklearn.linear_model.LogisticRegression`)
- **What & Why:** Naam mein "Regression" hone ke bawajood yeh **Classification** ka sab se mashhoor linear algorithm hai!
- **Internal Working & Math Intuition:**
  1. Pehle har feature ko weight ($w$) se multiply kar ke jama karta hai: $z = w_1 x_1 + w_2 x_2 + \dots + b$.
  2. Phir $z$ ko **Sigmoid (Logistic) Function** se guzarta hai jo kisi bhi number ko **$0.0$ aur $1.0$ ke darmiyan probability** mein badal deta hai:
     $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
  3. Agar probability $\ge 0.5$ ho to Class `1`, warna Class `0`!
- **Suitable Data & Scaling:** Numerical ya One-Hot encoded tabular aur text data. **Feature Scaling (`StandardScaler`) LAZMI hai** kyunki is mein default L2 regularization lagi hoti hai.
- **Important Hyperparameters:**
  - `C` (default `1.0`): **Inverse Regularization Strength**. Chota `C` (jaise `0.01`) = Sakht regularization (simple model, less overfitting). Bara `C` (jaise `100.0`) = Kamzor regularization.
  - `penalty`: `'l2'` (default), `'l1'`, `'elasticnet'`, ya `None`.
  - `solver`: `'lbfgs'` (default), `'liblinear'` (chote datasets ke liye), `'saga'` (bare datasets aur L1/ElasticNet ke liye).
  - `max_iter`: Default `100` (agar `ConvergenceWarning` aaye to `1000` kar dein).

#### 3. `LinearSVC` (`sklearn.svm.LinearSVC`) & 4. `SVC` (`sklearn.svm.SVC`)
- **What & Why:** **Support Vector Machines (SVM)** do classes ke darmiyan ek aisi lakeer (hyperplane) khinchte hain jis ka dono classes ke qareebi points (**Support Vectors**) se faasla (**Margin**) sab se zyada ho (**Maximum Margin Classifier**).
- **`LinearSVC` vs `SVC`:**
  - `LinearSVC`: Sirf seedhi boundary banata hai, lekin bare datasets (aur text classification) par `SVC(kernel='linear')` se kahin zyada tez hai! (Note: `LinearSVC` mein `.predict_proba()` nahi hota, sirf `.decision_function()` hota hai; probabilities ke liye isay `CalibratedClassifierCV` mein wrap karein).
  - `SVC`: **Kernel Trick** (`kernel='rbf'`, `'poly'`, `'sigmoid'`) ke zariye tirchay aur gol (non-linear) patterns ko high dimension mein le ja kar alag kar sakta hai. (`probability=True` karne par `.predict_proba()` deta hai).
- **Important Hyperparameters:** `C` (margin hardness), `kernel` (`'rbf'`, `'linear'`, `'poly'`), `gamma` (`'scale'`, `'auto'`, ya float—RBF curve ka gherao control karta hai). **Scaling (`StandardScaler`) LAZMI hai!**

#### 5. `SGDClassifier` (`sklearn.linear_model.SGDClassifier`)
- **What & Why:** **Stochastic Gradient Descent** classifier jo ek waqt mein ek sample dekh kar linear model (`loss='hinge'` = Linear SVM, ya `loss='log_loss'` = Logistic Regression) train karta hai.
- **When to Choose:** Jab dataset bohot bara ho ($> 100,000$ samples) ya memory mein fit na aata ho (supports `.partial_fit()`). **Scaling LAZMI hai!**

#### 6. `LinearDiscriminantAnalysis` (LDA) & 7. `QuadraticDiscriminantAnalysis` (QDA) (`sklearn.discriminant_analysis`)
- **What & Why:** Bayes' Theorem aur Gaussian (Normal) distribution par مبنی classifiers.
- **Internal Working:**
  - **LDA:** Maanta hai ke har class ka bell-curve (Gaussian) alag mean rakhta hai lekin sab ki **Covariance Matrix ek jaisi (shared)** hai $\rightarrow$ is liye **Linear (seedhi) boundary** banti hai. Yeh `.transform(X)` ke zariye supervised dimensionality reduction bhi kar sakta hai!
  - **QDA:** Har class ko apni alag Covariance Matrix rakhne ki ijazat deta hai $\rightarrow$ is liye **Curved (Quadratic) boundary** banti hai.

---

### FAMILY 3: Instance-Based (Lazy) Learning
#### 8. `KNeighborsClassifier` (KNN) (`sklearn.neighbors.KNeighborsClassifier`)
- **What & Why:** Sab se sada algorithm! Training (`fit`) ke waqt yeh koi formula nahi banata, bas training data ko yaad (save) kar leta hai.
- **Internal Working:** Jab naya point predict karna ho, to us ke sab se qareebi **`k` (`n_neighbors`)** parosiyon ka faasla (Euclidean distance) naapta hai aur aksariyat (majority vote) se faisla karta hai.
- **Real-World Analogy:** "Insaan apni sohbat (doston) se pehchana jata hai." Agar aap ke 5 sab se qareebi doston mein se 4 Cricket khelte hain, to model predict karega ke aap bhi Cricket pasand karte hain.
- **Hyperparameters:** `n_neighbors` (default `5`), `weights` (`'uniform'` ya `'distance'`), `metric` (`'minkowski'`, `'euclidean'`, `'manhattan'`). **Scaling LAZMI hai!**

---

### FAMILY 4: Naive Bayes Family (`sklearn.naive_bayes`)
Naive Bayes classifiers **Bayes' Theorem** par kaam karte hain aur "Naive" (bhola) is liye kehlate hain kyunki yeh farz karte hain ke tamam features ek doosre se bilkul azaad (conditionally independent) hain:
$$P(y \mid x_1, \dots, x_p) \propto P(y) \prod_{j=1}^{p} P(x_j \mid y)$$

| Classifier | Kis Data Type Ke Liye Bana Hai? | Key Hyperparameter |
| :--- | :--- | :--- |
| **9. `GaussianNB`** | Continuous numerical features (jin ka distribution Normal/bell-shaped ho) | `var_smoothing=1e-9` |
| **10. `MultinomialNB`** | **Text Classification** (Word counts ya TF-IDF scores; non-negative values) | `alpha=1.0` (Laplace smoothing) |
| **11. `BernoulliNB`** | Binary (`0`/`1`) features (maslan lafz email mein aaya ya nahi) | `alpha=1.0`, `binarize=0.0` |
| **12. `ComplementNB`** | **Imbalanced Text Classification** (`MultinomialNB` ka behtar version jab classes barabar na hon) | `alpha=1.0`, `norm=False` |

---

### FAMILY 5: Decision Trees & Tree Ensembles (`sklearn.tree`, `sklearn.ensemble`)
Tree models tabular data ke be-taaj badshah hain aur inhein **Feature Scaling ki zaroorat NAHI hoti**!

#### 13. `DecisionTreeClassifier` (`sklearn.tree.DecisionTreeClassifier`)
- **Internal Working:** Data ko step-by-step `if/else` sawalon (`income > 80000?`) ke zariye is tarah kaatta (split karta) hai ke har pattay (**Leaf Node**) mein ek hi class ke khalis (pure) samples reh jayein. Impurity naapne ke liye **Gini Impurity** ($1 - \sum p_i^2$) ya **Entropy** ($-\sum p_i \log_2 p_i$) use hoti hai.
- **Limitation:** Agar `max_depth=None` chor dein to darakht bohat gehra ho kar training data ko ratta (**Overfit**) laga leta hai!
- **Hyperparameters:** `criterion` (`'gini'`, `'entropy'`, `'log_loss'`), `max_depth`, `min_samples_split`, `min_samples_leaf`.

#### 14. `RandomForestClassifier` (`sklearn.ensemble.RandomForestClassifier`)
- **What & Why (Bagging + Feature Randomness):** Ek akele Decision Tree ki overfitting khatam karne ke liye **sainkron (e.g., 100) mukhtalif Decision Trees** banata hai (har tree ko data ka random bootstrap sample aur features ka random subset milta hai) aur aakhir mein sab trees ka **Majority Vote** leta hai!
- **Analogy:** Ek doctor se mashwara lene ke bajaye 100 mukhtalif specialist doctors ke board se vote lena.
- **Hyperparameters:** `n_estimators=100`, `max_depth`, `max_features='sqrt'`, `n_jobs=-1` (tamam CPU cores use karne ke liye), `class_weight='balanced'`.

#### 15. `ExtraTreesClassifier` (`sklearn.ensemble.ExtraTreesClassifier`)
- **What & Why (Extremely Randomized Trees):** `RandomForest` jaisa hi hai, lekin har feature par best split dhoondne mein waqt lagane ke bajaye **random split thresholds** chunta hai. Is se training tez hoti hai aur variance (overfitting) mazeed kam hota hai!

#### 16. `GradientBoostingClassifier` (`sklearn.ensemble.GradientBoostingClassifier`)
- **What & Why (Sequential Boosting):** Random Forest mein darakht ek doosre se azaad (parallel) bante hain. **Gradient Boosting** mein darakht **ek ke baad ek (sequentially)** bante hain—har naya chota tree (`max_depth=3`) pichle trees ki **ghaltiyon (residual errors / pseudo-residuals)** ko theek karne ki koshish karta hai!
- **Hyperparameters:** `n_estimators=100`, `learning_rate=0.1` (chota learning rate + zyada trees = behtar generalization), `max_depth=3`, `subsample=1.0`.

#### 17. `HistGradientBoostingClassifier` (`sklearn.ensemble.HistGradientBoostingClassifier`)
- **What & Why (LightGBM-inspired Fast Boosting):** Scikit-learn ka **sab se powerful aur tez-tareen tabular classifier** ($n > 10,000$ samples par `GradientBoostingClassifier` se 10x–50x tez)!
- **Internal Working:** Continuous numbers ko pehle `255` integer bins (Histograms) mein taqseem kar leta hai.
- **Built-in Superpowers:**
  1. **Native Missing Value (`NaN`) Support:** Isay `SimpleImputer` ki bhi zaroorat nahi!
  2. **Native Categorical Support:** `categorical_features="from_dtype"` dene se bina One-Hot Encoding ke categories handle karta hai!

#### 18. `AdaBoostClassifier` (`sklearn.ensemble.AdaBoostClassifier`)
- **What & Why (Adaptive Boosting):** Pehle ek chota tree (stump, `max_depth=1`) train karta hai. Jin samples par ghalti hoti hai, agli iteration mein un **mushkil samples ka weight barha deta hai** taake agla classifier un par zyada tawajjo de. *(Note: Scikit-learn v1.2+ mein parameter ka naam `base_estimator` se badal kar **`estimator`** ho chuka hai aur default algorithm `'SAMME'` hai).*

---

### FAMILY 6: Neural Network Classifier
#### 19. `MLPClassifier` (`sklearn.neural_network.MLPClassifier`)
- **What & Why:** **Multi-Layer Perceptron** (Feedforward Artificial Neural Network). Input layer, ek ya zyada `hidden_layer_sizes=(100,)` (with ReLU activation), aur output softmax/logistic layer ke zariye non-linear patterns seekhta hai. **Scaling LAZMI hai!** (Tafseel Part 18 mein).

---

### FAMILY 7: Meta-Ensembles, Calibration & Multiclass Wrappers
#### 20. `VotingClassifier` (`sklearn.ensemble.VotingClassifier`)
- **What & Why:** Alag-alag qism ke models (maslan `LogisticRegression` + `RandomForest` + `SVC`) ko mila kar ek team banata hai.
- **`voting='hard'`**: Class predictions ka majority vote.
- **`voting='soft'`**: Har model ki `.predict_proba()` probabilities ka average (aksar behtar result deta hai).

#### 21. `StackingClassifier` (`sklearn.ensemble.StackingClassifier`)
- **What & Why:** Voting mein sab ki barabar suni jati hai; **Stacking** mein base models (`estimators`) ki cross-validated predictions ko ek **Meta-Model (`final_estimator`, default `LogisticRegression`)** ke input ke taur par diya jata hai jo seekhta hai ke kis model par kitna bharosa karna hai!

#### 22. `CalibratedClassifierCV` (`sklearn.calibration.CalibratedClassifierCV`)
- **What & Why:** Kuch models (`LinearSVC`, `GaussianNB`, `RandomForest`) ki probabilities (`predict_proba`) true confidence nahi dikhatin. `CalibratedClassifierCV` (`method='sigmoid'` ya `'isotonic'`) cross-validation ke zariye sachi, calibrated probabilities paida karta hai.

#### 23. `OneVsRestClassifier` (OvR), 24. `OneVsOneClassifier` (OvO), 25. `OutputCodeClassifier` (`sklearn.multiclass`)
- **`OneVsRestClassifier` (OvR):** $K$ classes ke liye $K$ binary classifiers banata hai (har class vs baqi sab). Multilabel classification ke liye bhi use hota hai!
- **`OneVsOneClassifier` (OvO):** Har do classes ke joray (pair) ke darmiyan muqabla karwata hai ($\frac{K(K-1)}{2}$ classifiers).
- **`OutputCodeClassifier`:** Har class ko ek unique binary codeword (Error-Correcting Output Code) deta hai.

---

## 3. Complete Runnable Benchmark Code: Comparing Key Classifiers on Breast Cancer Data

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

# Import Classifiers across all families
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

# 1. Load & Split Dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

# 2. Define Dictionary of Classifiers (Wrapped in StandardScaler where required)
models = {
    "01. Dummy (Baseline)": DummyClassifier(strategy="most_frequent"),
    "02. LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "03. KNeighborsClassifier": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "04. DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "05. RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "06. ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "07. GradientBoosting": GradientBoostingClassifier(random_state=42),
    "08. HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "09. AdaBoost": AdaBoostClassifier(random_state=42),
    "10. GaussianNB": GaussianNB(),
    "11. LinearSVC + Calibrated": CalibratedClassifierCV(make_pipeline(StandardScaler(), LinearSVC(random_state=42))),
    "12. SVC (RBF Kernel)": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
    "13. SGDClassifier": make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", random_state=42)),
    "14. MLPClassifier (Neural Net)": make_pipeline(StandardScaler(), MLPClassifier(max_iter=500, random_state=42)),
    "15. LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    "16. QDA": make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis()),
}

# Add Voting & Stacking Ensembles
models["17. VotingClassifier (Soft)"] = VotingClassifier(
    estimators=[
        ("lr", models["02. LogisticRegression"]),
        ("rf", models["05. RandomForest"]),
        ("svc", models["12. SVC (RBF Kernel)"])
    ],
    voting="soft"
)

models["18. StackingClassifier"] = StackingClassifier(
    estimators=[
        ("lr", models["02. LogisticRegression"]),
        ("rf", models["05. RandomForest"]),
        ("hgb", models["08. HistGradientBoosting"])
    ],
    final_estimator=LogisticRegression()
)

# 3. Train, Predict & Evaluate All Models
results = []
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Test_Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "ROC_AUC": round(roc_auc_score(y_test, y_proba), 4)
    })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
```

### Expected Output
```text
                         Model  Test_Accuracy  ROC_AUC
          01. Dummy (Baseline)         0.6316   0.5000
        02. LogisticRegression         0.9825   0.9954
      03. KNeighborsClassifier         0.9561   0.9788
              04. DecisionTree         0.9386   0.9360
              05. RandomForest         0.9561   0.9937
                06. ExtraTrees         0.9561   0.9957
          07. GradientBoosting         0.9561   0.9907
      08. HistGradientBoosting         0.9737   0.9937
                  09. AdaBoost         0.9561   0.9825
                10. GaussianNB         0.9386   0.9878
    11. LinearSVC + Calibrated         0.9737   0.9960
          12. SVC (RBF Kernel)         0.9825   0.9950
             13. SGDClassifier         0.9649   0.9931
14. MLPClassifier (Neural Net)         0.9649   0.9940
                       15. LDA         0.9561   0.9917
                       16. QDA         0.9561   0.9888
   17. VotingClassifier (Soft)         0.9737   0.9964
        18. StackingClassifier         0.9825   0.9960
```

---

## 4. When to Choose Which Classifier? (Decision Guide)

| Scenario / Problem Type | Recommended Scikit-learn Classifiers |
| :--- | :--- |
| **Sanity Check / Benchmark** | `DummyClassifier(strategy="most_frequent")` |
| **Fast, Interpretable Baseline** | `LogisticRegression` (with `StandardScaler`) |
| **Medium/Large Tabular Data (Best Accuracy)** | `HistGradientBoostingClassifier` or `RandomForestClassifier` |
| **Small/Medium Data with Non-Linear Boundaries** | `SVC(kernel="rbf")` (with `StandardScaler`) |
| **High-Dimensional Text / Bag-of-Words Data** | `ComplementNB`, `MultinomialNB`, `LinearSVC`, or `LogisticRegression` |
| **Huge Dataset ($>10^6$ rows, Out-of-Core)** | `SGDClassifier` (using `.partial_fit()`) |
| **Combining Diverse Models for Maximum Score** | `StackingClassifier` or `VotingClassifier(voting="soft")` |

---

## 5. Practice Exercises & Solutions

### Exercises
1. **Beginner:** `load_iris()` par `GaussianNB` aur `DecisionTreeClassifier(max_depth=3)` ko train kar ke dono ki test accuracy compare karein.
2. **Intermediate:** `OneVsOneClassifier(LinearSVC(random_state=42))` ko scaled Iris dataset par fit karein aur dekhein ke 3 classes ke liye andar kitne binary estimators (`estimators_`) bane hain.
3. **Challenge:** `MultinomialNB` aur `BernoulliNB` ko ek chote synthetic word-count matrix `X = np.array([[2, 1, 0], [3, 0, 0], [0, 1, 4], [0, 0, 3]])`, `y = np.array([0, 0, 1, 1])` par fit karein aur `[1, 0, 2]` ke liye dono ki predicted probabilities print karein.

### Solutions
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier

# 1. Beginner
iris = load_iris()
X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
print("GaussianNB Acc:", round(GaussianNB().fit(X_tr, y_tr).score(X_te, y_te), 4))
print("DecisionTree Acc:", round(DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr).score(X_te, y_te), 4))

# 2. Intermediate
scaler = StandardScaler()
ovo = OneVsOneClassifier(LinearSVC(random_state=42)).fit(scaler.fit_transform(X_tr), y_tr)
print("Number of OvO binary estimators for 3 classes:", len(ovo.estimators_))  # 3*(3-1)/2 = 3

# 3. Challenge
X_w = np.array([[2, 1, 0], [3, 0, 0], [0, 1, 4], [0, 0, 3]])
y_w = np.array([0, 0, 1, 1])
mnb = MultinomialNB().fit(X_w, y_w)
bnb = BernoulliNB().fit(X_w, y_w)
print("MultinomialNB Proba:", mnb.predict_proba([[1, 0, 2]]).round(3))
print("BernoulliNB Proba  :", bnb.predict_proba([[1, 0, 2]]).round(3))
```

---
**Next Chapter:** [06_REGRESSION.md](06_REGRESSION.md)
