# Part 5: Data Preparation and Preprocessing (`sklearn.preprocessing`, `sklearn.impute`, `sklearn.compose`)

Real-world datasets kabhi bhi saaf suthray nahi hote. Un mein **Missing Values (`NaN`)**, **Text/Categorical Columns** (`"Karachi"`, `"Lahore"`), **Alag-Alag Scales** (Age `25` vs Salary `250,000`), aur **Outliers** hote hain. Scikit-learn ke models sirf saaf numerical matrices samajhte hain. Is chapter mein hum [Scikit-learn Preprocessing & Imputation](https://scikit-learn.org/stable/modules/preprocessing.html) ke tamam 29 topics tafseel se seekhenge.

---

## 1. Features ($X$), Target ($y$), and Splitting (`train_test_split`)

### A. Topic Introduction & Why Do We Need Train/Validation/Test Splits?
- **Features ($X$):** Input columns (Numerical: `area_marla`, `income`; Categorical: `city`, `education_level`).
- **Target ($y$):** Output column jise predict karna hai.
- **Three Data Splits:**
  1. **Training Set (`X_train, y_train`)**: Model aur preprocessors ko sikhane (`fit`) ke liye (~70%–80%).
  2. **Validation Set (`X_val, y_val`)**: Best algorithm aur hyperparameters (`GridSearchCV`) chunne ke liye.
  3. **Test Set (`X_test, y_test`)**: Bilkul aakhir mein final model ka imtihan lene ke liye (~20%).
- **Random vs Stratified Splitting (`stratify=y`):**
  - Agar aap ke dataset mein 90% Healthy aur sirf 10% Fraud cases hain, to simple random split mein ho sakta hai saare Fraud cases train mein chale jayein aur test mein zero bachein!
  - **`stratify=y`** is baat ki zamanat deta hai ke Train aur Test dono mein classes ka tanasub (90% / 10% ratio) bilkul barabar rahe ([Cross-validation & Splitting Guide](https://scikit-learn.org/stable/modules/cross_validation.html)).

### B. Complete Working Code (`train_test_split` with `stratify`)
```python
import numpy as np
from sklearn.model_selection import train_test_split

X = np.arange(20).reshape(10, 2)
# Imbalanced target: eight 0s and two 1s (20% class 1)
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

print("Train y distribution:", y_train, "-> Class 1 ratio:", y_train.mean())
print("Test y distribution :", y_test,  "-> Class 1 ratio:", y_test.mean())
```
**Expected Output:**
```text
Train y distribution: [0 0 0 1 0] -> Class 1 ratio: 0.2
Test y distribution : [0 1 0 0 0] -> Class 1 ratio: 0.2
```

---

## 2. Handling Missing Values: `SimpleImputer` and `KNNImputer` (`sklearn.impute`)

### A. Topic Introduction & Why Do We Need It?
- Zyadatar Scikit-learn algorithms (`LogisticRegression`, `SVC`, `LinearRegression`, `MLP`) mein agar ek bhi `NaN` chala jaye to woh foran `ValueError: Input X contains NaN` de kar ruk jate hain. *(Note: `HistGradientBoostingClassifier/Regressor` aur modern `RandomForest`/`DecisionTree` natively `NaN` handle kar sakte hain).*
- **Missing Value Strategies in `SimpleImputer(strategy=...)`:**
  1. `'mean'`: Numerical column ke ausat (average) se bharna. (Jab outliers na hon).
  2. `'median'`: Darmiyani value se bharna. (**Best for numerical data with outliers**, jaise salaries ya house prices).
  3. `'most_frequent'`: Sab se zyada baar aane wali value (Mode) se bharna. (**Best for categorical strings** jaise `"Lahore"`).
  4. `'constant'`: Kisi fix value (`fill_value=0` ya `fill_value="Missing"`) se bharna.
  5. `add_indicator=True`: Ek naya binary column (`0`/`1`) bana deta hai jo model ko yaad dilata hai ke yahan value gayab thi!
- **`KNNImputer(n_neighbors=5)`:** Sab ka ek hi average dalne ke bajaye, milte-julte `k` parosi samples (nearest neighbors) dhoond kar un ke hisab se missing value estimate karta hai!

### B. Real-World Analogy
Farz karein ek school ke form par 10th class ke bache ki `Age` likhi honi reh gayi (`NaN`).
- **`SimpleImputer(strategy='median')`**: Poore school ke bachon ki darmiyani umar (maslan `12 saal`) likh dega.
- **`KNNImputer`**: Dekhega ke is bache ki `Class = 10th` aur `Height = 5.6 ft` hai, to sirf 10th class ke un 5 bachon ki average umar (`15.5 saal`) likhega jin ka qad aur class is se milti hai!

### C. Complete Working Code (Before-and-After Shape & Values)
```python
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# Before Data: [Area_Marla, Price_Lakhs] with NaNs
X_missing = np.array([
    [5.0,  140.0],
    [10.0, 300.0],
    [np.nan, 150.0],  # Area missing, but price is close to 5 Marla!
    [20.0, 600.0]
])

print("BEFORE Imputation (Shape:", X_missing.shape, "):\n", X_missing)

# 1. SimpleImputer with median + missing indicator
sim_imp = SimpleImputer(strategy="median", add_indicator=True)
X_simple = sim_imp.fit_transform(X_missing)
print("\nAFTER SimpleImputer(median, add_indicator=True) (Shape:", X_simple.shape, "):")
print(X_simple)

# 2. KNNImputer (n_neighbors=2)
knn_imp = KNNImputer(n_neighbors=2)
X_knn = knn_imp.fit_transform(X_missing)
print("\nAFTER KNNImputer(n_neighbors=2) (Shape:", X_knn.shape, "):")
print(X_knn)
```

### D. Expected Output
```text
BEFORE Imputation (Shape: (4, 2) ):
 [[  5. 140.]
 [ 10. 300.]
 [ nan 150.]
 [ 20. 600.]]

AFTER SimpleImputer(median, add_indicator=True) (Shape: (4, 3) ):
 [[  5. 140.   0.]
 [ 10. 300.   0.]
 [ 10. 150.   1.]
 [ 20. 600.   0.]]

AFTER KNNImputer(n_neighbors=2) (Shape: (4, 2) ):
 [[  5.  140. ]
 [ 10.  300. ]
 [  7.5 150. ]
 [ 20.  600. ]]
```
*(Explanation: `SimpleImputer` mein `add_indicator=True` ki wajah se shape `(4, 2)` se barh kar `(4, 3)` ho gayi kyunki teesra column batata hai ke Row 2 mein `area` missing tha!)*

---

## 3. Handling Categorical Values: `OneHotEncoder`, `OrdinalEncoder`, and `LabelEncoder`

### A. Why Arbitrary Integer Encoding Misleads a Model!
Agar hamare paas `City` column mein `["Karachi", "Lahore", "Quetta"]` hai aur hum unhein `Karachi = 0`, `Lahore = 1`, `Quetta = 2` de dein, to Linear Regression ya KNN model yeh ghalat-fehmi paal lega ke:
- *"Quetta (2) Karachi (0) se do-guna bara hai!"* ya *"Karachi (0) aur Quetta (2) ka average `(0+2)/2 = 1` yani Lahore hai!"*
Yeh bilkul be-tukki baat hai kyunki shehron ke darmiyan koi mathematical order nahi hota (**Nominal Categories**).

### B. Which Encoder to Use When?
1. **`OneHotHarCategoryKeLiyeAlagBinaryColumn` (`OneHotEncoder`)**:
   - **Kab Use Karein?** **Nominal Features ($X$)** ke liye jin mein koi darja-bandi (order) na ho (jaise `City`: Karachi, Lahore, Islamabad).
   - **Kaise Kaam Karta Hai?** Har unique category ke liye ek naya `0/1` dummy column bana deta hai.
   - **Key Parameters:**
     - `handle_unknown='ignore'`: Bohat zaroori! Agar test data mein koi naya shehar `"Multan"` aa jaye jo training mein nahi tha, to code crash hone ke bajaye uski row mein `[0, 0, 0]` laga dega.
     - `sparse_output=True` (default): Memory bachane ke liye **Compressed Sparse Row (CSR) Matrix** return karta hai (kyunki hazaron columns mein zyadatar entries `0` hoti hain). Agar aam NumPy array chahiye ho to `sparse_output=False` dein.
     - `drop='first'`: Multicollinearity (dummy variable trap) se bachne ke liye $k$ mein se $k-1$ columns rakhta hai (linear models ke liye mufeed).
2. **`OrdinalEncoder`**:
   - **Kab Use Karein?** **Ordinal Features ($X$)** ke liye jin mein qudrati tarteeb (natural order) ho! Maslan `Education`: `["Matric", "Bachelors", "Masters", "PhD"]` $\rightarrow$ `[0, 1, 2, 3]`, ya `Size`: `["Small", "Medium", "Large"]`.
3. **`LabelEncoder`**:
   - **Rule:** Official Scikit-learn documentation ke mutabiq `LabelEncoder` **SIRF 1D Target Labels ($y$)** ko encode karne ke liye bana hai, features ($X$) ke liye **hargiz nahi**!

### C. Complete Working Code (Encoders & Sparse Matrix Demonstration)
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

df_cat = pd.DataFrame({
    "city": ["Lahore", "Karachi", "Islamabad", "Lahore"],
    "size": ["Small", "Large", "Medium", "Small"]
})
y_raw = ["Approve", "Reject", "Approve", "Reject"]

# 1. OneHotEncoder on Nominal column ('city')
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
city_ohe = ohe.fit_transform(df_cat[["city"]])
print("OneHotEncoder Output Shape:", city_ohe.shape)
print("Feature Names:", ohe.get_feature_names_out())
print(city_ohe)

# Testing unseen category ('Multan') without crashing!
unseen_test = pd.DataFrame({"city": ["Multan"]})
print("Unseen 'Multan' Encoded:", ohe.transform(unseen_test))

# 2. OrdinalEncoder with explicit order on 'size'
ord_enc = OrdinalEncoder(categories=[["Small", "Medium", "Large"]])
size_ord = ord_enc.fit_transform(df_cat[["size"]])
print("\nOrdinalEncoder Output (Small=0, Medium=1, Large=2):\n", size_ord.ravel())

# 3. LabelEncoder ONLY for Target y
le = LabelEncoder()
y_enc = le.fit_transform(y_raw)
print("\nLabelEncoder Target y:", y_enc, "| Classes:", le.classes_)
```

### D. Expected Output
```text
OneHotEncoder Output Shape: (4, 3)
Feature Names: ['city_Islamabad' 'city_Karachi' 'city_Lahore']
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]
Unseen 'Multan' Encoded: [[0. 0. 0.]]

OrdinalEncoder Output (Small=0, Medium=1, Large=2):
 [0. 2. 1. 0.]

LabelEncoder Target y: [0 1 0 1] | Classes: ['Approve' 'Reject']
```

---

## 4. Feature Scaling & Normalization: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Normalizer`

### A. Why Do We Need Scaling?
Farz karein Feature 1 `Experience_Years` (`1` se `10` tak) hai aur Feature 2 `Monthly_Salary_PKR` (`50,000` se `500,000` tak) hai.
- Distance-based aur Gradient-based algorithms (**KNN, SVM, Logistic/Linear Regression with Regularization, PCA, Neural Networks/MLP, KMeans**) बड़े numbers (`500,000`) se darr jate hain aur chote numbers (`10`) ko bilkul nazar-andaz kar dete hain!
- **Kab Scaling Zaroori NAHI Hai?** Tree-based models (**Decision Trees, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting**) ko scaling ki zaroorat nahi hoti kyunki woh ek waqt mein ek hi column par threshold split (`salary > 100000`) check karte hain.

### B. Formulas & Comparison of Scalers

| Transformer | Mathematical Formula (per column / row) | Kab Use Karein? |
| :--- | :--- | :--- |
| **`StandardScaler`** | $z = \frac{x - \mu}{\sigma}$ (Mean = $0$, Std = $1$) | Default choice for Linear Models, SVM, PCA, MLP |
| **`MinMaxScaler`** | $x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ (Range $[0, 1]$) | Jab bounded values $[0, 1]$ chahiye hon (Outliers se kharab hota hai!) |
| **`RobustScaler`** | $x_{\text{scaled}} = \frac{x - \text{Median}}{\text{IQR}_{75 - 25}}$ | **Jab data mein extreme Outliers hon!** (Median aur IQR use karta hai) |
| **`Normalizer`** | $x_{\text{norm}} = \frac{x}{\|x\|_2}$ (**Row-wise** unit length $= 1$) | **Column-wise Scaling NAHI, balke Row-wise Normalization** (Text TF-IDF aur cosine similarity ke liye) |

### C. Complete Working Code: Comparing Scalers on Data with an Outlier
```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

# Column 0: Normal experience | Column 1: Salary with 1 extreme billionaire outlier
X_raw = np.array([
    [1.0,  50000.0],
    [2.0,  60000.0],
    [3.0,  70000.0],
    [4.0,  80000.0],
    [5.0, 5000000.0]  # Outlier!
])

print("StandardScaler (Column-wise z-score):\n", np.round(StandardScaler().fit_transform(X_raw), 3))
print("\nMinMaxScaler [0, 1] (Notice first 4 salaries crushed near 0!):\n", np.round(MinMaxScaler().fit_transform(X_raw), 4))
print("\nRobustScaler (Handles the outlier gracefully via Median/IQR):\n", np.round(RobustScaler().fit_transform(X_raw), 2))
print("\nNormalizer (Row-wise L2 norm so each row length = 1):\n", np.round(Normalizer().fit_transform([[3.0, 4.0]]), 2))
```

### D. Expected Output
```text
StandardScaler (Column-wise z-score):
 [[-1.414 -0.506]
 [-0.707 -0.501]
 [ 0.    -0.496]
 [ 0.707 -0.491]
 [ 1.414  1.995]]

MinMaxScaler [0, 1] (Notice first 4 salaries crushed near 0!):
 [[0.     0.    ]
 [0.25   0.002 ]
 [0.5    0.004 ]
 [0.75   0.0061]
 [1.     1.    ]]

RobustScaler (Handles the outlier gracefully via Median/IQR):
 [[-1.  -1. ]
 [-0.5 -0.5]
 [ 0.   0. ]
 [ 0.5  0.5]
 [ 1.  246.5]]

Normalizer (Row-wise L2 norm so each row length = 1):
 [[0.6 0.8]]
```

---

## 5. Additional Transformers (`FunctionTransformer`, `Binarizer`, `QuantileTransformer`, `PowerTransformer`)
- **`FunctionTransformer(np.log1p)`**: Skewed data (jaise house prices ya salaries) par custom mathematical function ($\log(1+x)$) lagane ke liye.
- **`Binarizer(threshold=50.0)`**: Continuous numbers ko threshold ke hisab se seedha `0` ya `1` banane ke liye.
- **`PowerTransformer(method='yeo-johnson')`** & **`QuantileTransformer`**: Tirchay (skewed) distributions ko Gaussian (bell-curve) shakal mein dhalne ke liye.

---

## 6. Combining Mixed Data Without Leakage: `ColumnTransformer` & `make_column_transformer`

### A. Why Do We Need `ColumnTransformer`?
Asli datasets mein kuch columns **Numerical** hote hain (jinhein `SimpleImputer(median)` + `StandardScaler` chahiye) aur kuch columns **Categorical** hote hain (jinhein `SimpleImputer(most_frequent)` + `OneHotEncoder` chahiye).
**`ColumnTransformer`** humein ijazat deta hai ke hum ek hi step mein alag-alag columns par alag-alag transformers lagayein aur un ke outputs ko automatically jor kar ek final feature matrix hasil kar lein!

### B. Complete Working Code: Mixed Numerical & Categorical Preprocessing
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, Binarizer

# 1. Mixed Dataset with Missing Values
df = pd.DataFrame({
    "area_marla": [5.0, 10.0, np.nan, 8.0, 12.0, 6.0],
    "income_k":   [80.0, 250.0, 120.0, np.nan, 300.0, 95.0],
    "city":       ["Lahore", "Karachi", "Lahore", "Islamabad", np.nan, "Karachi"],
    "bought":     [0, 1, 0, 1, 1, 0]
})

X = df[["area_marla", "income_k", "city"]]
y = df["bought"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 2. Define sub-pipelines for Numerical and Categorical features
num_cols = ["area_marla", "income_k"]
cat_cols = ["city"]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# 3. Combine with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ],
    remainder="drop"
)

# 4. Fit ONLY on X_train, transform both X_train and X_test
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Before Shape (X_train):", X_train.shape)
print("After Shape (X_train_processed):", X_train_processed.shape)
print("Output Feature Names:\n", preprocessor.get_feature_names_out())
print("Processed X_train Matrix:\n", np.round(X_train_processed, 2))

# Quick demo of FunctionTransformer and Binarizer
log_transformer = FunctionTransformer(np.log1p)
binarizer = Binarizer(threshold=100.0)
print("\nLog1p of [0, 10, 100]:", np.round(log_transformer.transform([[0, 10, 100]]), 2))
print("Binarized [80, 120, 250] at threshold=100:", binarizer.transform([[80, 120, 250]]))
```

### C. Expected Output
```text
Before Shape (X_train): (4, 3)
After Shape (X_train_processed): (4, 5)
Output Feature Names:
 ['num__area_marla' 'num__income_k' 'cat__city_Islamabad'
 'cat__city_Karachi' 'cat__city_Lahore']
Processed X_train Matrix:
 [[ 0.    0.   0.    0.    1.  ]
 [-0.45 -1.16 0.    1.    0.  ]
 [-0.89  0.   1.    0.    0.  ]
 [ 1.34  1.16 0.    1.    0.  ]]

Log1p of [0, 10, 100]: [[0.   2.4  4.62]]
Binarized [80, 120, 250] at threshold=100: [[0 1 1]]
```

---

## 7. Data Leakage: What It Is and How to Prevent It
- **Data Leakage Kya Hai?** Jab Test Data (`X_test`) ki maloomat ghalti se Training ke waqt model ya preprocessor ko mil jaye.
- **Sab Se Aam Ghalti:** `train_test_split` karne se **pehle** poore dataset `X` par `StandardScaler().fit_transform(X)` ya `SimpleImputer().fit_transform(X)` chala dena! Is se test set ka mean aur median training mein leak ho jata hai aur aap ko jhooti high accuracy milti hai jo production mein fail ho jati hai ([Scikit-learn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)).
- **Golden Rule to Prevent Leakage:**
  1. Pehle `train_test_split` karein.
  2. Tamam transformers ko `Pipeline` aur `ColumnTransformer` ke andar rakhein taake cross-validation ke har fold mein `.fit()` sirf training fold par chale!

---

## 8. Practice Exercises & Solutions

### Exercises
1. **Beginner:** Ek NumPy column `[[10], [20], [np.nan], [40]]` par `SimpleImputer(strategy='mean')` lagayein aur output print karein.
2. **Intermediate:** Ek DataFrame jismein `grade = ['A', 'C', 'B', 'A']` ho, us par `OrdinalEncoder` is tarah lagayein ke `'C' -> 0`, `'B' -> 1`, aur `'A' -> 2` bane.
3. **Challenge:** `make_column_transformer` use kar ke ek mixed DataFrame (`salary` numerical aur `department` categorical) ko preprocess karein aur `.get_feature_names_out()` print karein.

### Solutions
```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

# 1. Beginner
X_b = np.array([[10.0], [20.0], [np.nan], [40.0]])
print("Mean Imputed:", SimpleImputer(strategy="mean").fit_transform(X_b).ravel())

# 2. Intermediate
df_g = pd.DataFrame({"grade": ["A", "C", "B", "A"]})
enc = OrdinalEncoder(categories=[["C", "B", "A"]])
print("Ordinal Grades:", enc.fit_transform(df_g).ravel())

# 3. Challenge
df_m = pd.DataFrame({"salary": [50, 80, 120], "dept": ["IT", "HR", "IT"]})
ct = make_column_transformer(
    (StandardScaler(), ["salary"]),
    (OneHotEncoder(sparse_output=False), ["dept"])
)
res = ct.fit_transform(df_m)
print("Features:", ct.get_feature_names_out())
```

---
**Next Chapter:** [04_FIRST_MACHINE_LEARNING_MODEL.md](04_FIRST_MACHINE_LEARNING_MODEL.md)
