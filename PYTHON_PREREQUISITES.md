# Part 2: Python Prerequisites for Machine Learning & Scikit-learn

Is chapter ka maqsad aap ko poori Python encyclopedia ratwana nahi, balke **bilkul wohi Python concepts** sikhana hai jo Scikit-learn, NumPy, aur Pandas mein har roz istemal hote hain. Agar aap ne aaj se pehle kabhi Python code nahi likha, to yeh chapter aap ko zero se tayyar kar dega!

---

## 1. Variables, Data Types, Numbers, Strings, Booleans, and `None`

### A. Topic Introduction
- **Variable Kya Hai?** Computer ki memory mein kisi value (number ya text) ko naam de kar mehfooz (store) karne wala dabba (container).
- **Basic Data Types:**
  1. `int` (Integer): Baghair point ke poore numbers (jaise `n_Show = 5`, `random_state = 42`).
  2. `float` (Floating-point): Decimal/ashariya wale numbers (jaise `learning_rate = 0.01`, `accuracy = 0.945`).
  3. `str` (String): Quotes ke andar likha gaya text (jaise `city = "Lahore"`, `kernel = 'rbf'`).
  4. `bool` (Boolean): Sirf do values: `True` (Sach) ya `False` (Jhoot). Scikit-learn mein switches on/off karne ke liye use hota hai (jaise `fit_intercept=True`).
  5. `None`: "Kuch bhi nahi" ya "Khali/Default". Scikit-learn mein jab kisi parameter ki limit na lagani ho to `max_depth=None` diya jata hai.

### B. Complete Working Code
```python
# Machine Learning hyper-parameters aur metrics ke variables
model_name = "RandomForest"      # str (String)
n_estimators = 100               # int (Integer)
learning_rate = 0.05             # float (Decimal number)
use_scaling = True               # bool (Boolean: True or False)
max_depth = None                 # NoneType (No limit)

print(f"Model: {model_name} (Type: {type(model_name).__name__})")
print(f"Trees: {n_estimators} (Type: {type(n_estimators).__name__})")
print(f"Learning Rate: {learning_rate} (Type: {type(learning_rate).__name__})")
print(f"Scaling Enabled: {use_scaling} (Type: {type(use_scaling).__name__})")
print(f"Max Depth: {max_depth} (Type: {type(max_depth).__name__})")
```

### C. Expected Output
```text
Model: RandomForest (Type: str)
Trees: 100 (Type: int)
Learning Rate: 0.05 (Type: float)
Scaling Enabled: True (Type: bool)
Max Depth: None (Type: NoneType)
```

---

## 2. Lists, Tuples, Dictionaries, and Sets

### A. Why Do We Need Them in Scikit-learn?
Scikit-learn mein:
- **Lists (`[...]`)**: Feature names (`['age', 'salary']`) ya data rows pass karne ke liye use hoti hain. Lists ko badla (modify) ja sakta hai (**Mutable**).
- **Tuples (`(...)`)**: Array ka `.shape` (maslan `(150, 4)`) ya `Pipeline` ke steps `('scaler', StandardScaler())` hamesha Tuple hote hain. Tuples ko banne ke baad badla nahi ja sakta (**Immutable**).
- **Dictionaries (`{key: value}`)**: `GridSearchCV` mein hyperparameter grids (maslan `{'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}`) hamesha Dictionary mein diye jate hain!
- **Sets (`{...}`)**: Unique classes ya categories dekhne ke liye use hote hain kyunki Set mein duplicate values khatam ho jati hain.

### B. Complete Working Code
```python
# 1. List: Ordered and Mutable (Badla ja sakta hai)
features = ["area_marla", "bedrooms", "bathrooms"]
features.append("location")  # Naya feature add kiya
print("First Feature:", features[0])       # Indexing (0 se shuru hoti hai!)
print("Last Feature :", features[-1])      # -1 ka matlab aakhri item
print("Total Features:", len(features))

# 2. Tuple: Ordered and Immutable (Pipeline steps ya shape ke liye)
dataset_shape = (1000, 4)  # 1000 rows, 4 columns
n_samples, n_features = dataset_shape  # Tuple unpacking
print(f"Unpacked Tuple -> Samples: {n_samples}, Features: {n_features}")

# 3. Dictionary: Key-Value pairs (GridSearchCV parameter grid ke liye)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, None],
    "criterion": ["gini", "entropy"]
}
print("Tuning max_depth options:", param_grid["max_depth"])

# 4. Set: Unique items nikalne ke liye
raw_cities = ["Karachi", "Lahore", "Karachi", "Islamabad", "Lahore"]
unique_cities = set(raw_cities)
print("Unique Cities:", sorted(unique_cities))
```

### C. Expected Output
```text
First Feature: area_marla
Last Feature : location
Total Features: 4
Unpacked Tuple -> Samples: 1000, Features: 4
Tuning max_depth options: [3, 5, None]
Unique Cities: ['Islamabad', 'Karachi', 'Lahore']
```

---

## 3. Operators, Comparisons, `if/else`, Loops, and Comprehensions

### A. Topic Introduction
- **Comparisons (`==`, `!=`, `>`, `<`, `>=`, `<=`):** Data filter karne aur model ki accuracy check karne ke kaam aate hain.
- **`if / elif / else`:** Shart (condition) ke mutabiq faisla lene ke liye.
- **`for` Loop:** Ek list ke tamam models ko bari-bari train aur evaluate karne ke liye.
- **List & Dictionary Comprehensions:** Ek hi line mein loop chala kar nayi list ya dictionary banane ka Pythonic tareeqa (feature cleaning mein bohat use hota hai).

### B. Complete Working Code
```python
# Accuracy scores of 4 models
model_scores = {
    "Dummy": 0.50,
    "LogisticRegression": 0.84,
    "RandomForest": 0.91,
    "SVC": 0.88
}

# 1. For Loop + if/elif/else
for name, score in model_scores.items():
    if score >= 0.90:
        verdict = "Excellent (Production Ready)"
    elif score >= 0.80:
        verdict = "Good Baseline"
    else:
        verdict = "Poor (Reject)"
    print(f"{name:20s} -> Score: {score:.2f} | Status: {verdict}")

# 2. List Comprehension: Sirf woh models chunein jinka score > 0.85 hai
top_models = [name for name, score in model_scores.items() if score > 0.85]
print("Top Models (>85%):", top_models)

# 3. Dictionary Comprehension: Scores ko percentage (%) mein convert karein
score_percent = {name: round(score * 100, 1) for name, score in model_scores.items()}
print("Percentages:", score_percent)
```

### C. Expected Output
```text
Dummy                -> Score: 0.50 | Status: Poor (Reject)
LogisticRegression   -> Score: 0.84 | Status: Good Baseline
RandomForest         -> Score: 0.91 | Status: Excellent (Production Ready)
SVC                  -> Score: 0.88 | Status: Good Baseline
Top Models (>85%): ['RandomForest', 'SVC']
Percentages: {'Dummy': 50.0, 'LogisticRegression': 84.0, 'RandomForest': 91.0, 'SVC': 88.0}
```

---

## 4. Functions, Arguments, Return Values, Scope, and `lambda`

### A. Why Functions Matter in ML
Jab hum data clean karte hain ya 5 models ko evaluate karte hain, to bar-bar wohi 10 lines likhne ke bajaye hum ek reusable **Function (`def`)** bana lete hain. Chote 1-line functions ke liye **`lambda`** use hota hai (jaise `FunctionTransformer` ya Pandas `.apply()` mein).

### B. Complete Working Code
```python
def calculate_accuracy(y_true, y_pred, as_percentage=True):
    '''Do lists ko compare kar ke accuracy nikalne wala function.'''
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = correct / len(y_true)
    if as_percentage:
        return acc * 100.0
    return acc

actual = [1, 0, 1, 1, 0]
predicted = [1, 0, 0, 1, 0]

# Positional aur Default arguments ka istemal
print("Accuracy (%):", calculate_accuracy(actual, predicted))
print("Accuracy (0-1):", calculate_accuracy(actual, predicted, as_percentage=False))

# Lambda Function: Square footage ko Marla mein convert karna (1 Marla ~ 272.25 sq ft)
sqft_to_marla = lambda sqft: round(sqft / 272.25, 2)
print("1361 SqFt in Marla:", sqft_to_marla(1361))
```

### C. Expected Output
```text
Accuracy (%): 80.0
Accuracy (0-1): 0.8
1361 SqFt in Marla: 5.0
```

---

## 5. Classes and Objects (OOP Basics for Scikit-learn)

### A. Why You MUST Understand Classes & Objects for Scikit-learn
Scikit-learn ka poora design **Object-Oriented Programming (OOP)** par bana hai!
- Jab aap `from sklearn.linear_model import LogisticRegression` import karte hain, to `LogisticRegression` ek **Class (Naqsha / Blueprint)** hai.
- Jab aap `model = LogisticRegression(C=1.0)` likhte hain, to `model` us class ka ek **Object (Instance)** ban jata hai.
- Jab aap `model.fit(X, y)` chalate hain, to `.fit()` us object ka ek **Method (Function inside a class)** hai.
- Aur jab aap `model.coef_` dekhte hain, to `.coef_` us object ka ek **Attribute (Variable inside a class)** hai!

### B. Complete Working Code (Mini Custom Estimator Simulation)
```python
class SimpleThresholdClassifier:
    '''Scikit-learn style ki ek simple class jo threshold par predict karti hai.'''
    def __init__(self, threshold=50.0):
        # Constructor: Hyperparameter save karein
        self.threshold = threshold
        self.is_fitted_ = False

    def fit(self, X, y=None):
        # Training simulation: सीखा hua attribute '_' par khatam hota hai
        self.mean_seen_ = sum(X) / len(X)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Model abhi fit nahi hua! Pehle .fit() chalayein.")
        return [1 if x >= self.threshold else 0 for x in X]

# Class se Object (Instance) banayein
clf = SimpleThresholdClassifier(threshold=60.0)
clf.fit([45, 70, 82, 55])

print("Learned Mean Attribute (mean_seen_):", clf.mean_seen_)
print("Predictions for [50, 65, 90]:", clf.predict([50, 65, 90]))
```

### C. Expected Output
```text
Learned Mean Attribute (mean_seen_): 63.0
Predictions for [50, 65, 90]: [0, 1, 1]
```

---

## 6. Modules, Files, CSV Reading, Exceptions & Reading Tracebacks

### A. Topic Introduction
- **Imports (`import ... from ...`):** Scikit-learn bohat bari library hai, is liye hum poori library ek sath load karne ke bajaye specific module se class import karte hain: `from sklearn.metrics import accuracy_score`.
- **`try / except` (Error Handling):** Agar file na mile ya user ghalat input de, to program crash hone se bachane ke liye.
- **`csv` & `pathlib.Path`:** Files aur paths ko Windows (`\`) aur Linux/Mac (`/`) dono par bina error ke handle karne ke liye `pathlib.Path` best practice hai.
- **Reading Tracebacks:** Jab Python mein error aata hai, to ghabrane ke bajaye **Traceback ki sab se aakhri line** parhein—wahan Error ka naam (jaise `ValueError` ya `KeyError`) aur uski wajah saaf likhi hoti hai!

### B. Complete Working Code
```python
import csv
from pathlib import Path

# 1. Cross-platform file path banana aur CSV likhna/parhna
csv_path = Path("sample_houses.csv")

rows = [
    ["area_marla", "bedrooms", "price_lakhs"],
    [5, 3, 140],
    [10, 5, 310],
    [7, 4, 210]
]

# File write karein
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

# File read karein
with csv_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    loaded_data = [
        {"area": int(r["area_marla"]), "price": int(r["price_lakhs"])}
        for r in reader
    ]

print("Loaded CSV Records:", loaded_data)

# 2. Exception Handling demonstration
try:
    bad_number = int("140_Lakhs_Text")
except ValueError as err:
    print(f"Caught Expected Error -> ValueError: {err}")

# Cleanup sample file
if csv_path.exists():
    csv_path.unlink()
```

### C. Expected Output
```text
Loaded CSV Records: [{'area': 5, 'price': 140}, {'area': 10, 'price': 310}, {'area': 7, 'price': 210}]
Caught Expected Error -> ValueError: invalid literal for int() with base 10: '140_Lakhs_Text'
```

---

## 7. Jupyter Notebook vs Python Scripts (`.py`)
- **Jupyter Notebook (`.ipynb`):** Exploratory Data Analysis (EDA), graphs dekhne, aur step-by-step experiments ke liye behtareen hai.
- **Python Scripts (`.py`):** Jab aap ka model final ho jaye aur usay production server, automated pipeline, ya FastAPI mein chalana ho, to `.py` scripts use hoti hain.

---

## 8. Practice Exercises & Solutions

### Exercises
1. **Beginner:** Ek dictionary `student = {"name": "Ali", "marks": [65, 78, 82, 90]}` banayein aur us ke marks ka average calculate kar ke print karein.
2. **Intermediate:** Ek list `prices = [120, -50, 250, 0, 340]` di gayi hai (jismein `-50` aur `0` ghalat entries hain). List comprehension use kar ke sirf positive (`> 0`) prices filter karein.
3. **Challenge:** Ek class `MinMaxNormalizer` likhein jismein `.fit(data)` method list ka `min_` aur `max_` save kare, aur `.transform(data)` method har number ko `(x - min_) / (max_ - min_)` formula se `0` aur `1` ke darmiyan scale kare.

### Solutions
```python
# 1. Beginner Solution
student = {"name": "Ali", "marks": [65, 78, 82, 90]}
avg_marks = sum(student["marks"]) / len(student["marks"])
print(f"{student['name']} Average: {avg_marks}")  # 78.75

# 2. Intermediate Solution
prices = [120, -50, 250, 0, 340]
valid_prices = [p for p in prices if p > 0]
print("Valid Prices:", valid_prices)  # [120, 250, 340]

# 3. Challenge Solution
class MinMaxNormalizer:
    def fit(self, data):
        self.min_ = min(data)
        self.max_ = max(data)
        return self

    def transform(self, data):
        return [(x - self.min_) / (self.max_ - self.min_) for x in data]

scaler = MinMaxNormalizer().fit([10, 20, 30, 50])
print("Scaled:", scaler.transform([10, 30, 50]))  # [0.0, 0.5, 1.0]
```

---
**Next Chapter:** [NUMPY_PANDAS_FOUNDATIONS.md](NUMPY_PANDAS_FOUNDATIONS.md)
