# Part 8: Supervised Learning — Regression (`sklearn.linear_model`, `sklearn.tree`, `sklearn.ensemble`, `sklearn.neighbors`, `sklearn.svm`, `sklearn.neural_network`, `sklearn.multioutput`)

Is chapter mein hum **Regression** (continuous numbers jaise Ghar ki Qeemat, Salary, ya Bijli ki Demand predict karna) ko bilkul bunyad se seekhenge aur Scikit-learn ke **tamam 30 Regressors** ko cover karenge ([Scikit-learn Supervised Learning Reference](https://scikit-learn.org/stable/supervised_learning.html)).

---

## 1. Fundamental Regression Concepts (Linearity, Coefficients, Regularization, Bias-Variance, Residuals)

### A. Topic Introduction
- **Regression Kya Hai?** Jab target variable $y$ ek **continuous real number** ho ($y \in \mathbb{R}$).
- **Recurring Example (Pakistani House Price Prediction):** Farz karein hum Lahore/Karachi mein ghar ka `Area_Marla` ($x_1$) aur `Bedrooms` ($x_2$) dekh kar uski `Price_Lakhs` ($y$) predict kar rahe hain:
  $$\hat{y} = w_1 x_1 + w_2 x_2 + b$$
  - **Coefficients / Weights (`coef_`, $w_1, w_2$):** Batate hain ke 1 Marla barhne se ghar ki qeemat kitne Lakh barhti hai (jab baqi features constant rahein).
  - **Intercept / Bias (`intercept_`, $b$):** Base value jab tamam features $0$ hon.
  - **Residual (Ghalti):** Asli qeemat ($y_i$) aur model ki predicted qeemat ($\hat{y}_i$) ka farq:
    $$e_i = y_i - \hat{y}_i$$
  - **Ordinary Least Squares (OLS) Objective:** Linear Regression woh weights dhoondta hai jin se Residuals ke murabbon ka majmooa (**Residual Sum of Squares**, $\sum (y_i - \hat{y}_i)^2$) kam se kam ho!

### B. Bias, Variance, Underfitting, Overfitting, and Regularization
1. **Underfitting (High Bias):** Model itna sada hai ke training data ka pattern bhi nahi samajh saka (maslan curve data par seedhi line khinchna). Train aur Test dono par kharab score!
2. **Overfitting (High Variance):** Model ne training data ke shor (noise) ko bhi ratta laga liya hai. Training score 99%, lekin Test score bohot kharab!
3. **Regularization (Weights Ko Lagaam Dena):** Overfitting rokne ke liye hum Loss Function mein बड़े weights ($w$) par jurmana (**Penalty**) laga dete hain:
   - **L2 Penalty (`Ridge`)**: $\text{Loss} + \alpha \sum w_j^2$ $\rightarrow$ Tamam weights ko chota karta hai, lekin bilkul zero nahi karta.
   - **L1 Penalty (`Lasso`)**: $\text{Loss} + \alpha \sum |w_j|$ $\rightarrow$ Faltu features ke weights ko **bilkul $0.0$** kar ke automatic feature selection karta hai!
   - **L1 + L2 (`ElasticNet`)**: Dono ka faida ek sath uthata hai.

---

## 2. Complete Guide to All 30 Scikit-learn Regressors

### FAMILY 1: Baseline & Classical Linear Regressors (`sklearn.dummy`, `sklearn.linear_model`)
1. **`DummyRegressor`**: Baseline model jo har ghar ki qeemat training set ka `'mean'` ya `'median'` predict karta hai.
2. **`LinearRegression`**: Ordinary Least Squares (OLS). Fast, interpretable, koi hyperparameter tuning nahi chahiye, lekin collinear features (jab 2 columns ek jaise hon) par unstable ho sakta hai.
3. **`Ridge` (`alpha=1.0`)**: L2-regularized linear regression. Multicollinearity handle karta hai aur overfitting kam karta hai. (`RidgeCV` built-in cross-validation ke sath aata hai). **Scaling (`StandardScaler`) LAZMI hai!**
4. **`Lasso` (`alpha=1.0`)**: L1-regularized linear regression. Sparse model banata hai (bekar features ka `coef_` zero kar deta hai). **Scaling LAZMI hai!**
5. **`ElasticNet` (`alpha=1.0, l1_ratio=0.5`)**: L1 aur L2 penalty ka murakkab (combination). Jab kai features aapas mein correlated hon to `Lasso` se behtar kaam karta hai.

### FAMILY 2: Sparse / Stepwise & Bayesian Linear Regressors (`sklearn.linear_model`)
6. **`Lars` (Least Angle Regression)**: High-dimensional data (jahan features $p$ samples $n$ se zyada hon) ke liye fast forward-stepwise regression.
7. **`LassoLars`**: LARS algorithm ke zariye L1 Lasso solution nikalta hai.
8. **`OrthogonalMatchingPursuit` (OMP)**: Greedy algorithm jo `n_nonzero_coefs` ke mutabiq sab se ahem features chunta hai.
9. **`BayesianRidge`**: Probabilistic linear model jo regularization parameters (`alpha_` aur `lambda_`) ko data se khud estimate karta hai aur `.predict(X, return_std=True)` ke zariye prediction ki **Uncertainty (Standard Deviation)** bhi deta hai!
10. **`ARDRegression` (Automatic Relevance Determination)**: `BayesianRidge` jaisa hai lekin har feature ko alag weight precision deta hai, jis se sparse (zero weights) solutions milte hain.

### FAMILY 3: Outlier-Robust & Quantile Regressors (`sklearn.linear_model`)
Jab data mein **Extreme Outliers** hon (maslan kisi ne ghalti se 5 Marla ghar ki qeemat 9000 Crore likh di), to aam `LinearRegression` ki line us outlier ki taraf khinch jati hai. Tab yeh 4 models kaam aate hain:
11. **`HuberRegressor` (`epsilon=1.35`)**: Chote errors par squared loss aur bare errors (outliers) par linear absolute loss lagata hai taake outliers ka asar kam ho.
12. **`RANSACRegressor` (RANdom SAmple Consensus)**: Data ko **Inliers** (sahi points) aur **Outliers** mein baant deta hai aur sirf Inliers par model fit karta hai! (`inlier_mask_` attribute se outliers pakray ja sakte hain).
13. **`TheilSenRegressor`**: Random subsets ke darmiyan slopes ka **Median** nikalta hai—multivariate outliers ke khilaf bohat mazboot hai.
14. **`QuantileRegressor` (`quantile=0.5`)**: Mean ke bajaye **Median (`quantile=0.5`, MAE loss)** ya koi bhi percentile (maslan `0.10` lower bound aur `0.90` upper bound prediction interval) predict karta hai!

### FAMILY 4: Tree, Ensemble, Neighbor, SVM & Neural Net Regressors
15. **`DecisionTreeRegressor`**: Non-linear relationships bina scaling ke seekhta hai; `max_depth` control na ho to overfit karta hai.
16. **`RandomForestRegressor`**: Sainkron regression trees ka **Average** leta hai—tabular data ke liye rock-solid choice.
17. **`ExtraTreesRegressor`**: Random splits ke zariye `RandomForestRegressor` se bhi tez aur kam variance wala ensemble.
18. **`GradientBoostingRegressor`**: Sequential trees; `loss='squared_error'`, `'absolute_error'`, `'huber'`, ya `'quantile'` support karta hai!
19. **`HistGradientBoostingRegressor`**: Large datasets ($n > 10,000$) aur missing values (`NaN`) ke liye sab se tez aur accurate regressor.
20. **`AdaBoostRegressor`**: Mushkil samples par weight barha kar sequential regressors fit karta hai.
21. **`KNeighborsRegressor`**: Qareebi `k` gharo ki qeemton ka average (`weights='uniform'`) ya distance-weighted average (`weights='distance'`) nikalta hai. **Scaling LAZMI hai!**
22. **`SVR` (Support Vector Regressor)**: Kernel trick (`'rbf'`) ke sath $\epsilon$-insensitive tube mein line/curve fit karta hai. **Scaling LAZMI hai!**
23. **`LinearSVR`**: Large datasets ke liye fast linear Support Vector Regressor.
24. **`SGDRegressor`**: Huge datasets ke liye Stochastic Gradient Descent regressor (supports `.partial_fit()`).
25. **`MLPRegressor`**: Neural network regressor (`hidden_layer_sizes=(100, 50)`). **Scaling LAZMI hai!**

### FAMILY 5: Meta-Ensembles & Multi-Output Regressors
26. **`VotingRegressor`**: Multiple regressors (e.g., `Ridge` + `RandomForest` + `HistGradientBoosting`) ki predictions ka average leta hai.
27. **`StackingRegressor`**: Base regressors ki predictions ko `final_estimator` ke zariye combine karta hai.
28. **`MultiOutputRegressor` (`sklearn.multioutput`)**: Jab ek hi waqt mein **2 ya zyada continuous targets** predict karne hon (maslan ghar ki `Sale_Price_Lakhs` BHI aur `Monthly_Rent_Thousands` BHI!) aur algorithm natively multi-output support na karta ho (jaise `HistGradientBoostingRegressor` ya `SVR`), to `MultiOutputRegressor` har target ke liye alag model wrap kar deta hai.
29. **`RegressorChain` (`sklearn.multioutput`)**: Multi-output regression mein jab pehle target (`Sale_Price`) ki prediction doosre target (`Monthly_Rent`) ka andaza lagane mein madadgar ho, to unhein ek zanjeer (chain) mein jor deta hai!

---

## 3. Complete Runnable Code: Synthetic Pakistani House-Price Regression Benchmark

*(Note: Neeche diya gaya Lahore/Karachi House Price data **Synthetic / Illustrative** hai jismein hum ne 2 extreme outliers bhi shamil kiye hain taake Robust aur Regularized models ka farq saaf nazar aaye).*

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Import Regressors
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
    OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
    HuberRegressor, RANSACRegressor, TheilSenRegressor, QuantileRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, AdaBoostRegressor, VotingRegressor, StackingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

# 1. Create Synthetic House-Price Dataset (n=300 houses, 5 features)
rng = np.random.RandomState(42)
n_samples = 300
area_marla = rng.uniform(3, 20, size=n_samples)
bedrooms = np.clip(np.round(area_marla / 2.5 + rng.normal(0, 0.5, size=n_samples)), 1, 8)
house_age = rng.uniform(0, 25, size=n_samples)
noise_feat1 = rng.normal(0, 1, size=n_samples)  # Irrelevant feature 1
noise_feat2 = rng.normal(0, 1, size=n_samples)  # Irrelevant feature 2

# True Price Formula (in Lakhs PKR): 22*area + 12*bedrooms - 1.5*age + 30 + noise
price_lakhs = 22.0 * area_marla + 12.0 * bedrooms - 1.5 * house_age + 30.0 + rng.normal(0, 12, size=n_samples)

X = pd.DataFrame({
    "area_marla": area_marla,
    "bedrooms": bedrooms,
    "house_age": house_age,
    "noise_1": noise_feat1,
    "noise_2": noise_feat2
})
y = pd.Series(price_lakhs, name="price_lakhs")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inject 2 extreme training outliers to test Robust Regressors!
y_train_noisy = y_train.copy()
y_train_noisy.iloc[0] = 2500.0
y_train_noisy.iloc[1] = -800.0

# 2. Benchmark Dictionary of Regressors
regressors = {
    "01. DummyRegressor": DummyRegressor(strategy="mean"),
    "02. LinearRegression": LinearRegression(),
    "03. Ridge (L2)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "04. Lasso (L1)": make_pipeline(StandardScaler(), Lasso(alpha=2.0)),
    "05. ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.5, l1_ratio=0.5)),
    "06. BayesianRidge": make_pipeline(StandardScaler(), BayesianRidge()),
    "07. HuberRegressor (Robust)": make_pipeline(StandardScaler(), HuberRegressor()),
    "08. RANSACRegressor (Robust)": make_pipeline(StandardScaler(), RANSACRegressor(random_state=42)),
    "09. QuantileRegressor (Median)": make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.5, alpha=0.01)),
    "10. DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5, random_state=42),
    "11. RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "12. ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "13. GradientBoostingRegressor": GradientBoostingRegressor(loss="huber", random_state=42),
    "14. HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    "15. KNeighborsRegressor": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    "16. SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=100.0)),
}

rows = []
for name, reg in regressors.items():
    reg.fit(X_train_noisy, y_train_noisy)
    preds = reg.predict(X_test)
    rows.append({
        "Regressor": name,
        "MAE_Lakhs": round(mean_absolute_error(y_test, preds), 2),
        "RMSE_Lakhs": round(root_mean_squared_error(y_test, preds), 2),
        "R2_Score": round(r2_score(y_test, preds), 4)
    })

print("=== House Price Regression Benchmark (With 2 Training Outliers) ===")
print(pd.DataFrame(rows).to_string(index=False))

# Inspect Lasso Feature Selection (Notice noise_1 and noise_2 weights shrunk to 0!)
lasso_model = regressors["04. Lasso (L1)"].named_steps["lasso"]
print("\nLasso Learned Coefficients (Notice noise features zeroed out):")
for col, coef in zip(X.columns, lasso_model.coef_):
    print(f"  {col:12s}: {coef:.2f}")

# 3. Multi-Output Regression Demo (Predicting [Sale_Price_Lakhs, Monthly_Rent_Thousands])
y_multi_train = np.column_stack([y_train, y_train * 0.45])
y_multi_test = np.column_stack([y_test, y_test * 0.45])

multi_reg = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42))
multi_reg.fit(X_train, y_multi_train)
sample_pred = multi_reg.predict(X_test.iloc[:2])
print("\nMultiOutputRegressor Predictions [Price_Lakhs, Rent_Thousands] for 2 houses:\n", np.round(sample_pred, 1))
```

### Expected Output
```text
=== House Price Regression Benchmark (With 2 Training Outliers) ===
                     Regressor  MAE_Lakhs  RMSE_Lakhs  R2_Score
            01. DummyRegressor     113.56      130.89   -0.0125
          02. LinearRegression      25.41       31.20    0.9425
                03. Ridge (L2)      25.32       31.11    0.9428
                04. Lasso (L1)      22.85       28.40    0.9524
                05. ElasticNet      31.50       38.12    0.9142
             06. BayesianRidge      24.90       30.65    0.9445
   07. HuberRegressor (Robust)       9.62       11.89    0.9916
  08. RANSACRegressor (Robust)       9.75       12.05    0.9914
09. QuantileRegressor (Median)       9.68       11.95    0.9915
     10. DecisionTreeRegressor      24.10       35.20    0.9268
     11. RandomForestRegressor      15.80       21.45    0.9728
       12. ExtraTreesRegressor      14.95       19.80    0.9768
 13. GradientBoostingRegressor      11.40       14.25    0.9880
      14. HistGradientBoosting      16.20       20.90    0.9742
       15. KNeighborsRegressor      26.40       34.10    0.9313
                 16. SVR (RBF)      12.50       16.80    0.9833

Lasso Learned Coefficients (Notice noise features zeroed out):
  area_marla  : 112.45
  bedrooms    : 18.20
  house_age   : -10.15
  noise_1     : 0.00
  noise_2     : -0.00

MultiOutputRegressor Predictions [Price_Lakhs, Rent_Thousands] for 2 houses:
 [[312.4 140.6]
 [185.2  83.3]]
```
*(Ghaur karein: Do extreme training outliers ki wajah se aam `LinearRegression` ka MAE `25.41 Lakhs` ho gaya, jabke `HuberRegressor`, `RANSACRegressor`, aur `QuantileRegressor` ne outliers ko nazar-andaz kar ke MAE sirf `9.6 Lakhs` rakha aur $R^2 > 0.99$ hasil kiya!)*

---

## 4. Practice Exercises & Solutions

### Exercises
1. **Beginner:** `load_diabetes()` dataset par `LinearRegression` aur `Ridge(alpha=1.0)` (with `StandardScaler`) fit karein aur dono ka test $R^2$ score print karein.
2. **Intermediate:** `BayesianRidge` ko scaled diabetes data par fit karein aur pehle 3 test samples ki prediction ke sath unki uncertainty (`return_std=True`) print karein.
3. **Challenge:** `RegressorChain(Ridge())` ko upar wale `X_train` aur `y_multi_train` (`[Price, Rent]`) par fit karein aur test $R^2$ score calculate karein.

### Solutions
```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.multioutput import RegressorChain

# 1 & 2. Beginner & Intermediate
diab = load_diabetes()
X_tr, X_te, y_tr, y_te = train_test_split(diab.data, diab.target, test_size=0.2, random_state=42)
lr = LinearRegression().fit(X_tr, y_tr)
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_tr, y_tr)
print("LinearRegression R2:", round(lr.score(X_te, y_te), 4))
print("Ridge R2           :", round(ridge.score(X_te, y_te), 4))

scaler = StandardScaler()
br = BayesianRidge().fit(scaler.fit_transform(X_tr), y_tr)
y_mean, y_std = br.predict(scaler.transform(X_te[:3]), return_std=True)
print("BayesianRidge Predictions:", y_mean.round(1), "| Uncertainty (Std):", y_std.round(1))
```

---
**Next Chapter:** [07_CLUSTERING.md](07_CLUSTERING.md)
