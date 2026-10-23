# Part 3: NumPy, Pandas, and Data Visualization Foundations

Scikit-learn khud CSV files read nahi karta aur na hi charts banata hai. Is liye real-world Machine Learning mein **NumPy**, **Pandas**, aur **Matplotlib/Seaborn** Scikit-learn ke sab se qareebi sathi (companions) hain ([Scikit-learn Getting Started](https://scikit-learn.org/stable/getting_started.html)):
1. **NumPy (`numpy`)**: Tez-tareen 1D/2D numerical arrays aur linear algebra calculations ke liye (Scikit-learn andar se NumPy arrays par hi hisab kitab karta hai).
2. **Pandas (`pandas`)**: CSV/Excel datasets load karne, missing values dhoondne, aur tables (`DataFrame`) saaf karne ke liye.
3. **Matplotlib & Seaborn**: Data ke distributions, correlations, aur outliers ko graphs ke zariye dekhne ke liye.

---

## SECTION 1: NumPy Foundations (`import numpy as np`)

### A. Topic Introduction & Why Do We Need It?
- Python ki aam `list` mein agar 10 lakh numbers hon aur sab ko `2` se multiply karna ho, to `for` loop bohat slow chalta hai.
- **NumPy (`ndarray`)** C-language ki raftaar se **Vectorized Operations** karta hai (bina loop likhe poore array par ek sath math operation!).
- **Scikit-learn Requirement:** Scikit-learn ke har model ko Features **`X` hamesha 2D Shape `(n_samples, n_features)`** mein chahiye hote hain, aur Target **`y` 1D Shape `(n_samples,)`** mein.

### B. Complete Working Code: Arrays, Shape, Reshaping, Stats, NaN & Random Seeds
```python
import numpy as np

# 1. 1D aur 2D Arrays banana
y = np.array([140, 210, 310, 180])  # 1D Target array
X = np.array([
    [5, 3, 1],   # Sample 1: 5 Marla, 3 Bed, Lahore(1)
    [7, 4, 1],   # Sample 2
    [10, 5, 2],  # Sample 3
    [6, 3, 2]    # Sample 4
], dtype=np.float64)

print(f"X dimensions (ndim): {X.ndim}, shape: {X.shape}, size: {X.size}, dtype: {X.dtype}")
print(f"y dimensions (ndim): {y.ndim}, shape: {y.shape}")

# 2. Indexing aur Slicing
print("First Sample (Row 0):", X[0])
print("All Marla Values (Column 0):", X[:, 0])
print("First 2 rows, first 2 cols:\n", X[:2, :2])

# 3. Reshaping: 1D array ko 2D column vector (n_samples, 1) banana
# Bohat zaroori: Single feature ko Scikit-learn mein pass karne ke liye reshape(-1, 1) lagta hai!
single_feature = np.array([5, 7, 10, 6])
single_feature_2d = single_feature.reshape(-1, 1)
print("Reshaped from", single_feature.shape, "to", single_feature_2d.shape)

# 4. Mathematical & Statistical Operations (Mean, Median, Std, Variance)
marla_col = X[:, 0]
print(f"Mean (Ausat)        : {np.mean(marla_col):.2f}")
print(f"Median (Darmiyani)  : {np.median(marla_col):.2f}")
print(f"Std Dev (Phailao)   : {np.std(marla_col):.2f}")
print(f"Variance (Std^2)    : {np.var(marla_col):.2f}")

# 5. Missing Values (np.nan) detection
data_with_nan = np.array([10.0, np.nan, 30.0, np.nan, 50.0])
print("Is NaN mask:", np.isnan(data_with_nan))
print("Total NaNs :", np.isnan(data_with_nan).sum())
print("Mean ignoring NaN (nanmean):", np.nanmean(data_with_nan))

# 6. Random Numbers & Reproducibility (Seed)
rng = np.random.RandomState(42)
print("Reproducible Random Floats:", np.round(rng.uniform(0, 10, size=3), 2))
```

### C. Code Explanation in Roman Urdu
- `X.shape`: Output `(4, 3)` ka matlab hai 4 rows (`n_samples`) aur 3 columns (`n_features`).
- `X[:, 0]`: Colon `:` ka matlab hai "tamam rows", aur `0` ka matlab hai "pehla column".
- `single_feature.reshape(-1, 1)`: `-1` NumPy ko kehta hai ke rows ki ginti khud hisab laga lo, lekin column lazmi `1` rakho. Yeh Scikit-learn ke sab se aam error (`Expected 2D array, got 1D array instead`) ka fori ilaj hai!
- `np.isnan(data_with_nan).sum()`: `True` ko `1` aur `False` ko `0` maan kar ginta hai ke kitni missing values hain.

### D. Expected Output
```text
X dimensions (ndim): 2, shape: (4, 3), size: 12, dtype: float64
y dimensions (ndim): 1, shape: (4,)
First Sample (Row 0): [5. 3. 1.]
All Marla Values (Column 0): [ 5.  7. 10.  6.]
First 2 rows, first 2 cols:
 [[5. 3.]
 [7. 4.]]
Reshaped from (4,) to (4, 1)
Mean (Ausat)        : 7.00
Median (Darmiyani)  : 6.50
Std Dev (Phailao)   : 1.87
Variance (Std^2)    : 3.50
Is NaN mask: [False  True False  True False]
Total NaNs : 2
Mean ignoring NaN (nanmean): 30.0
Reproducible Random Floats: [3.75 9.51 7.32]
```

---

## SECTION 2: Pandas Foundations (`import pandas as pd`)

### A. Topic Introduction
- **Series:** Ek single column (1D labeled array).
- **DataFrame:** Poora Excel-style table (2D labeled table jismein har column ka apna naam aur data type ho sakta hai—kuch numerical aur kuch text/categorical).
- **Good News:** Modern Scikit-learn (`v1.0+`) mein aap seedha Pandas `DataFrame` ko `X` ke taur par aur `Series` ko `y` ke taur par pass kar sakte hain!

### B. Complete Working Code: DataFrame Inspection, Cleaning, Grouping & Feature Engineering
```python
import pandas as pd
import numpy as np

# 1. Synthetic Pakistani Real-Estate Dataset banana
raw_data = {
    "city": ["Lahore", "Karachi", "Lahore", "Islamabad", "Karachi", "Karachi"],
    "area_marla": [5.0, 10.0, np.nan, 8.0, 10.0, 4.0],
    "bedrooms": [3, 5, 3, 4, 5, 2],
    "price_lakhs": [145, 320, 150, 290, 320, 125]
}
df = pd.DataFrame(raw_data)

print("--- 1. Top 3 Rows (head) ---")
print(df.head(3))

print("\n--- 2. Shape & Summary Stats (describe) ---")
print("Shape:", df.shape)
print(df.describe())

# 3. Missing Values & Duplicates check karna
print("\nMissing Values per Column:\n", df.isna().sum())
print("Duplicate Rows Count:", df.duplicated().sum())

# 4. Duplicate drop karna aur Missing Value fill karna (EDA demonstration)
df_clean = df.drop_duplicates().copy()
df_clean["area_marla"] = df_clean["area_marla"].fillna(df_clean["area_marla"].median())

# 5. Filtering, Sorting & Selecting (loc / iloc)
karachi_houses = df_clean[df_clean["city"] == "Karachi"].sort_values(by="price_lakhs", ascending=False)
print("\n--- Filtered Karachi Houses ---")
print(karachi_houses)

# 6. Grouping & Aggregation
city_summary = df_clean.groupby("city")["price_lakhs"].agg(["count", "mean"])
print("\n--- City-wise Price Summary ---")
print(city_summary)

# 7. Basic Feature Engineering (Price per Marla)
df_clean["price_per_marla"] = df_clean["price_lakhs"] / df_clean["area_marla"]

# 8. Separating Features (X) and Target (y) for Scikit-learn
X = df_clean[["city", "area_marla", "bedrooms"]]
y = df_clean["price_lakhs"]
print("\nFinal X shape:", X.shape, "| Final y shape:", y.shape)
```

### C. Expected Output
```text
--- 1. Top 3 Rows (head) ---
      city  area_marla  bedrooms  price_lakhs
0   Lahore         5.0         3          145
1  Karachi        10.0         5          320
2   Lahore         NaN         3          150

--- 2. Shape & Summary Stats (describe) ---
Shape: (6, 4)
       area_marla  bedrooms  price_lakhs
count    5.000000  6.000000     6.000000
mean     7.400000  3.666667   225.000000
...
Missing Values per Column:
city           0
area_marla     1
bedrooms       0
price_lakhs    0
dtype: int64
Duplicate Rows Count: 1

--- Filtered Karachi Houses ---
      city  area_marla  bedrooms  price_lakhs
1  Karachi        10.0         5          320
5  Karachi         4.0         2          125

--- City-wise Price Summary ---
           count   mean
city                   
Islamabad      1  290.0
Karachi        2  222.5
Lahore         2  147.5

Final X shape: (5, 3) | Final y shape: (5,)
```

> **Warning on Missing Values:** Upar hum ne Pandas mein `.fillna()` sirf sikhane ke liye dikhaya hai. Jab aap Machine Learning model banayenge (Part 5), to missing values hamesha `train_test_split` ke **baad** Scikit-learn ke `SimpleImputer` se fill karenge taake **Data Leakage** na ho!

---

## SECTION 3: Data Visualization with Matplotlib & Seaborn

### A. Why Visualize Before Modeling?
Graphs banaye baghair model train karna andheray mein teer chalane jaisa hai. Visualization se humein foran pata chalta hai:
1. **Histogram (`sns.histplot`)**: Kya data bell-curve (Normal) hai ya ek taraf jhuka hua (Skewed)?
2. **Box Plot (`sns.boxplot`)**: Kya data mein extreme **Outliers** (ghalat ya ghair-mamooli values, jaise 5 Marla ghar ki qeemat 99,999 Crore) mojood hain?
3. **Scatter Plot (`sns.scatterplot`)**: Kya Feature $X$ aur Target $y$ mein sidha (linear) talluq hai ya curve?
4. **Correlation Heatmap (`sns.heatmap`)**: Kaunse features target ke sath sab se zyada juray hue hain?

### B. Complete Working Visualization Script
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Synthetic dataset for visualization
rng = np.random.RandomState(42)
n = 100
area = rng.uniform(3, 20, size=n)
price = 25 * area + rng.normal(0, 30, size=n)
# Add 1 extreme outlier
area[0], price[0] = 5.0, 950.0

df_viz = pd.DataFrame({
    "area_marla": area,
    "bedrooms": np.clip(np.round(area / 2.5), 1, 8),
    "price_lakhs": price
})

# Create a 2x2 grid of plots
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. Histogram (Distribution)
sns.histplot(df_viz["price_lakhs"], kde=True, ax=axes[0, 0], color="teal")
axes[0, 0].set_title("1. Histogram: Price Distribution")

# 2. Box Plot (Outlier Detection)
sns.boxplot(x=df_viz["price_lakhs"], ax=axes[0, 1], color="coral")
axes[0, 1].set_title("2. Box Plot: Detecting the 950 Lakh Outlier")

# 3. Scatter Plot (Feature vs Target Relationship)
sns.scatterplot(data=df_viz, x="area_marla", y="price_lakhs", ax=axes[1, 0])
axes[1, 0].set_title("3. Scatter Plot: Area vs Price")

# 4. Correlation Heatmap
corr_matrix = df_viz.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".2f", ax=axes[1, 1])
axes[1, 1].set_title("4. Correlation Heatmap")

plt.tight_layout()
# Save figure to file (or use plt.show() in Jupyter)
plt.savefig("eda_overview.png", dpi=120)
plt.close()
print("Saved 4-panel EDA chart to eda_overview.png successfully!")
print("Correlation Matrix:\n", corr_matrix.round(2))
```

### C. Expected Output
```text
Saved 4-panel EDA chart to eda_overview.png successfully!
Correlation Matrix:
             area_marla  bedrooms  price_lakhs
area_marla         1.00      0.98         0.79
bedrooms           0.98      1.00         0.77
price_lakhs        0.79      0.77         1.00
```

---

## 4. Practice Exercises & Solutions

### Exercises
1. **Beginner (NumPy):** Ek 1D array `arr = np.arange(1, 13)` banayein aur usay `(4, 3)` shape ke 2D matrix mein reshape karein. Phir har column ka mean (`axis=0`) calculate karein.
2. **Intermediate (Pandas):** Ek DataFrame banayein jismein 5 students ke `math_score` aur `english_score` hon (1 missing value ke sath). Missing value ko column ke mean se fill karein aur ek naya column `total_score` banayein.
3. **Challenge (Outlier Filtering):** Ek NumPy array `salaries = np.array([45000, 52000, 48000, 51000, 9500000, 49000])` mein se woh values filter kar ke nikal dein jo `mean + 2 * std` se bari hon.

### Solutions
```python
import numpy as np
import pandas as pd

# 1. Beginner Solution
arr = np.arange(1, 13).reshape(4, 3)
print("Column Means:", arr.mean(axis=0))  # [5.5, 6.5, 7.5]

# 2. Intermediate Solution
df_s = pd.DataFrame({"math": [80, np.nan, 70, 90, 85], "eng": [75, 80, 65, 88, 92]})
df_s["math"] = df_s["math"].fillna(df_s["math"].mean())
df_s["total"] = df_s["math"] + df_s["eng"]
print(df_s)

# 3. Challenge Solution
salaries = np.array([45000, 52000, 48000, 51000, 9500000, 49000])
cutoff = salaries.mean() + 2 * salaries.std()
clean_salaries = salaries[salaries <= cutoff]
print("Clean Salaries:", clean_salaries)  # [45000 52000 48000 51000 49000]
```

---
**Next Chapter:** [02_INTRODUCTION_TO_SCIKIT_LEARN.md](02_INTRODUCTION_TO_SCIKIT_LEARN.md)
