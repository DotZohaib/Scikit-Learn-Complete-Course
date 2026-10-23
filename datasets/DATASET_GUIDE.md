# Dataset Guide: Built-in, OpenML, and Synthetic Educational Datasets

Is course mein har code block aur project ko **100% self-contained aur offline-runnable** rakhne ke liye hum ne Scikit-learn ke [Dataset Loading Utilities (`sklearn.datasets`)](https://scikit-learn.org/stable/datasets.html) ka istemal kiya hai. Is guide mein bataya gaya hai ke:
1. Built-in Toy Datasets kaise load karein (bina internet ke).
2. Real-World External Datasets (`fetch_*` aur `OpenML`) internet ke sath kaise download karein.
3. Pakistani domain scenarios (Karachi/Lahore House Prices, Bilingual SMS Spam, Telecom Churn) ke liye reproducible CSV datasets apne folder mein kaise generate karein!

---

## 1. Built-in Offline Datasets (`load_*` — No Internet Needed)

Yeh 5 datasets Scikit-learn install hote hi aap ke computer par mojood hote hain:

| Loader Function | Samples ($N$) | Features ($P$) | Target ($y$) & Task Type | License / Origin |
| :--- | :--- | :--- | :--- | :--- |
| **`load_iris(as_frame=True)`** | 150 | 4 | 3 flower species (Multiclass Classification) | Fisher (1936), Public Domain |
| **`load_wine(as_frame=True)`** | 178 | 13 | 3 wine cultivars (Multiclass Classification) | UCI ML Repository, CC BY 4.0 |
| **`load_breast_cancer(as_frame=True)`** | 569 | 30 | `0=malignant`, `1=benign` (Binary Classification) | Wisconsin Diagnostic (UCI), CC BY 4.0 |
| **`load_digits(as_frame=True)`** | 1,797 | 64 | Handwritten digits `0`–`9` (8x8 image classification/clustering) | UCI Optical Recognition, CC BY 4.0 |
| **`load_diabetes(as_frame=True)`** | 442 | 10 | Disease progression after 1 year (Regression) | Efron et al. (2004), Public Educational |

---

## 2. Real-World Downloadable Datasets (`fetch_*` — Requires Internet on First Run)

Jab aap ke computer par internet active ho, to Scikit-learn automatically yeh datasets download kar ke `~/scikit_learn_data` mein cache kar leta hai:

```python
from sklearn.datasets import fetch_california_housing, fetch_openml, fetch_20newsgroups

# 1. California Housing (20,640 districts, 8 features -> Median House Value)
cal_housing = fetch_california_housing(as_frame=True)

# 2. Titanic Passenger Survival from OpenML.org (1,309 passengers, mixed numerical & categorical)
titanic = fetch_openml(name="titanic", version=1, as_frame=True, parser="auto")

# 3. 20 Newsgroups Text Corpus (for NLP Classification & TF-IDF)
newsgroups = fetch_20newsgroups(subset="train", categories=["sci.space", "rec.autos"], remove=("headers", "footers", "quotes"))
```

---

## 3. Reproducible CSV Generator Script for Pakistani Course Projects

Agar aap course ke projects (`pak_house_prices.csv`, `pak_sms_spam.csv`, `pak_telecom_churn.csv`) ko asaan `.csv` files ki shakal mein apne `datasets/` folder mein save karna chahte hain taake `pd.read_csv()` ki practice kar saken, to neeche diya gaya script run karein:

```python
import numpy as np
import pandas as pd
from pathlib import Path

out_dir = Path(".")
rng = np.random.RandomState(42)

# 1. Generate pak_house_prices.csv (SYNTHETIC EDUCATIONAL DATASET, n=500)
n_h = 500
city = rng.choice(["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad"], size=n_h)
area_marla = rng.uniform(3, 20, size=n_h).round(1)
bedrooms = np.clip(np.round(area_marla / 2.5), 1, 8).astype(int)
age_years = rng.randint(0, 25, size=n_h)
city_base = pd.Series(city).map({"Islamabad": 45, "Karachi": 35, "Lahore": 28, "Rawalpindi": 18, "Faisalabad": 14}).values
price_lakhs = (23.5 * area_marla + 13.0 * bedrooms - 1.5 * age_years + city_base + rng.normal(0, 12, size=n_h)).round(1)

df_houses = pd.DataFrame({
    "city": city,
    "area_marla": area_marla,
    "bedrooms": bedrooms,
    "age_years": age_years,
    "price_lakhs": price_lakhs
})
df_houses.to_csv(out_dir / "pak_house_prices.csv", index=False)
print("Saved pak_house_prices.csv:", df_houses.shape)
```

---
**Back to Main Index:** [../README.md](../README.md)
