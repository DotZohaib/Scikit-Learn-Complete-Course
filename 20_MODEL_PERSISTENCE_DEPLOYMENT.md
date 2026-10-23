# Part 22: Model Persistence and Production Readiness (`joblib`, `pickle`, `FastAPI`, Drift & Model Cards)

Jupyter Notebook mein model train karne ke baad asal maqsad usay **Save (Persist)** karna aur kisi **Web Application, Mobile App, ya Batch Pipeline** mein deploy karna hota hai ([Scikit-learn Model Persistence Guide](https://scikit-learn.org/stable/model_persistence.html)). Is chapter mein hum tamam 20 production readiness topics seekhenge.

---

## 1. Saving & Loading Models: `joblib` vs `pickle` & Security Risks

### A. Why Models Are Saved & `joblib` vs `pickle`
- Har customer request par model ko 10 GB data par dobara `.fit()` nahi kiya ja sakta! Hum training ke baad fitted **`Pipeline`** ko disk par `.joblib` file mein save kar lete hain aur production server par milliseconds mein `.load()` kar ke `.predict()` chalate hain.
- **Why `joblib` over `pickle`?** Scikit-learn models ke andar bade NumPy arrays hote hain. **`joblib.dump(pipeline, 'model.joblib', compress=3)`** NumPy arrays ko `pickle` se kahin zyada tezi aur kam file size mein save/load karta hai!

### B. CRITICAL SECURITY WARNING: Never Load Untrusted `pickle` or `joblib` Files!
> **Security Rule #1:** `pickle` aur `joblib` files ke andar arbitrary Python code chhupa ho sakta hai! Agar aap internet se kisi anjaan/untrusted shaks ki `.pkl` ya `.joblib` file download kar ke `joblib.load()` chalayenge, to aap ka computer hack ho sakta hai (Remote Code Execution).
> - **Safe Alternatives for Cross-Platform / Untrusted Sharing:** **`ONNX` (`skl2onnx` + `onnxruntime`)** ya **`skops.io`** use karein!

---

## 2. Version Compatibility, Metadata Bundling & Input Validation

1. **Version Incompatibility (`InconsistentVersionWarning`):**
   - Agar aap ne model `scikit-learn==1.4.0` aur `python==3.10` par save kiya aur production server par `scikit-learn==1.6.1` chal raha hai, to internal Cython structures badalne ki wajah se model ghalat predictions de sakta hai ya crash ho sakta hai!
   - **Best Practice:** Model save karte waqt ek dictionary bundle save karein jismein `pipeline`, `sklearn_version`, `feature_names`, `training_date`, aur `metrics` sab mehfooz hon!
2. **Handling Unseen Categories & Missing Values in Production:**
   - Production mein user kabhi bhi koi naya shehar `"Gwadar"` ya khali (`NaN`) value bhej sakta hai. Is liye hamesha **poora `Pipeline` (`SimpleImputer` + `OneHotEncoder(handle_unknown='ignore')` + `Estimator`)** ek sath save karein!

---

## 3. Monitoring Production Models: Data Drift, Concept Drift & Model Cards

1. **Data Drift (Covariate Shift — $P(X)$ Changes):**
   - Jab production mein aane wale input features ($X$) ka distribution training data se badal jaye! *Misaal:* 2021 mein training ke waqt Lahore mein 5 Marla ghar ki average income requirement 1 Lakh thi, aur महंगाई (inflation) ke baad 2026 mein inputs 3 Lakh ho gaye. (Monitor via Kolmogorov–Smirnov test or Population Stability Index — PSI).
2. **Concept Drift ($P(y \mid X)$ Changes):**
   - Inputs wohi hain, lekin logon ka rawaiya ya दुनिया ka qanoon badal gaya! *Misaal:* COVID-19 lockdowns ke dauran online grocery buying ka pattern achanak badal gaya.
3. **Retraining Strategies:**
   - **Scheduled Periodic Retraining** (e.g., har mahine ki 1 tareekh ko taza data par retrain karna).
   - **Performance/Drift Triggered Retraining** (jab validation metric ya PSI threshold cross kare).
4. **Model Cards (Responsible Documentation):**
   - Har production model ke sath ek `MODEL_CARD.md` ya metadata dict honi chahiye jismein: Intended Use, Out-of-Scope Uses, Training Dataset Dates, Evaluation Metrics across Subgroups, aur Ethical/Fairness Limitations darj hon.

---

## 4. Complete Working Code: Bundling Pipeline + Metadata, Schema Validation, Drift Check & Batch Inference

```python
import sys
import joblib
import sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# 1. Train Production-Ready Pipeline (Handles NaNs & Unseen Categories!)
rng = np.random.RandomState(42)
n = 300
df_train = pd.DataFrame({
    "income_k": rng.normal(120, 40, size=n),
    "loan_amount_k": rng.normal(500, 150, size=n),
    "city": rng.choice(["Karachi", "Lahore", "Islamabad"], size=n)
})
y_train = ((df_train["loan_amount_k"] / np.maximum(df_train["income_k"], 10)) > 4.5).astype(int)

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), ["income_k", "loan_amount_k"]),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["city"])
])

prod_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", HistGradientBoostingClassifier(random_state=42))
])
prod_pipe.fit(df_train, y_train)

# 2. Save Complete Model Artifact Bundle with Version & Schema Metadata
artifact_bundle = {
    "model_name": "pak_loan_risk_classifier_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "python_version": sys.version.split()[0],
    "sklearn_version": sklearn.__version__,
    "expected_columns": list(df_train.columns),
    "train_income_mean": float(df_train["income_k"].mean()),
    "train_income_std": float(df_train["income_k"].std()),
    "threshold": 0.40,
    "pipeline": prod_pipe
}

bundle_path = Path("loan_risk_bundle.joblib")
joblib.dump(artifact_bundle, bundle_path, compress=3)

# 3. Production Inference Service Class (With Version Check, Validation & Drift Alert)
class LoanRiskInferenceService:
    def __init__(self, path: Path):
        self.bundle = joblib.load(path)
        if self.bundle["sklearn_version"] != sklearn.__version__:
            print(f"WARNING: Model trained on sklearn {self.bundle['sklearn_version']}, running on {sklearn.__version__}!")
        self.pipe = self.bundle["pipeline"]
        self.cols = self.bundle["expected_columns"]
        self.threshold = self.bundle["threshold"]

    def validate_and_predict_batch(self, df_new: pd.DataFrame) -> pd.DataFrame:
        missing_cols = set(self.cols) - set(df_new.columns)
        if missing_cols:
            raise ValueError(f"Missing required input columns: {missing_cols}")

        # Enforce exact column ordering
        X_ordered = df_new[self.cols].copy()

        # Simple Data Drift Check (Z-shift on income_k mean)
        batch_mean = X_ordered["income_k"].dropna().mean()
        z_shift = abs(batch_mean - self.bundle["train_income_mean"]) / self.bundle["train_income_std"]
        drift_flag = bool(z_shift > 1.5)

        probas = self.pipe.predict_proba(X_ordered)[:, 1]
        preds = (probas >= self.threshold).astype(int)

        out = df_new.copy()
        out["risk_probability"] = probas.round(4)
        out["high_risk_flag"] = preds
        out["batch_drift_detected"] = drift_flag
        return out

# 4. Test Batch Inference (Including Missing Value NaN and Unseen City 'Multan'!)
service = LoanRiskInferenceService(bundle_path)
incoming_batch = pd.DataFrame({
    "city": ["Karachi", "Multan", np.nan],  # Notice column order is swapped & 'Multan' is unseen!
    "income_k": [150.0, 60.0, np.nan],
    "loan_amount_k": [300.0, 550.0, 700.0]
})

predictions_df = service.validate_and_predict_batch(incoming_batch)
print("=== Production Batch Inference Results ===")
print(predictions_df.to_string(index=False))

if bundle_path.exists():
    bundle_path.unlink()
```

### Expected Output
```text
=== Production Batch Inference Results ===
   city  income_k  loan_amount_k  risk_probability  high_risk_flag  batch_drift_detected
Karachi     150.0          300.0            0.0012               0                 False
 Multan      60.0          550.0            0.9985               1                 False
    NaN       NaN          700.0            0.9842               1                 False
```

---

## 5. Optional Companion: Serving a Scikit-learn Pipeline via FastAPI (`app.py` Template)
Jab aap apne saved `.joblib` model ko REST API banana chahein (taake Android/iOS app ya React frontend usay call kar sake), to **FastAPI** aur **Pydantic** industry standard hain:

```python
# Save as app.py and run with: uvicorn app:app --reload
# (Requires: pip install fastapi uvicorn pydantic joblib scikit-learn pandas)
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(title="Loan Risk Prediction API", version="1.0.0")

# Load pipeline once at startup (NOT inside the request handler!)
# bundle = joblib.load("loan_risk_bundle.joblib")
# pipeline = bundle["pipeline"]

class LoanApplication(BaseModel):
    income_k: float = Field(..., gt=0, description="Monthly income in thousands PKR")
    loan_amount_k: float = Field(..., gt=0, description="Requested loan in thousands PKR")
    city: str = Field(..., examples=["Karachi", "Lahore", "Islamabad"])

@app.post("/predict")
def predict_risk(payload: LoanApplication):
    df_input = pd.DataFrame([payload.model_dump()])
    # proba = float(pipeline.predict_proba(df_input)[0, 1])
    proba = 0.85  # Illustrative response when run standalone
    return {"risk_probability": round(proba, 4), "high_risk": proba >= 0.40}
```

---

## 6. Practice Exercises & Solutions

### Exercises
1. **Beginner:** Kyun `joblib.load()` ko har `/predict` API request ke andar chalana buri practice hai?
2. **Intermediate:** Agar training ke waqt columns `['income_k', 'loan_amount_k']` thay aur inference ke waqt user NumPy array mein `[loan_amount_k, income_k]` ulat tarteeb mein bhej de, to kya hoga? Is se kaise bachein?
3. **Challenge:** Ek dictionary `model_card` banayein jismein model ke 5 zaroori governance fields mojood hon.

### Solutions
```python
# 1. Beginner Answer:
# Disk I/O slow hota hai! Har request par file disk se load karne se latency 5ms se barh kar 500ms ho jayegi. Model ko hamesha server startup par global memory mein sirf EK dafa load karein.

# 2. Intermediate Answer:
# NumPy array mein column names nahi hote, is liye model bina error diye ghalat prediction de dega (Silent Bug!). Hamesha Pandas DataFrame pass karein aur df[expected_columns] se order lock karein.

# 3. Challenge Solution:
model_card = {
    "model_id": "loan_risk_v1",
    "intended_use": "Assisting credit officers in reviewing retail consumer loans in Pakistan",
    "out_of_scope_use": "Automated rejection without human review or corporate commercial loans",
    "training_data_window": "2024-01 to 2026-06",
    "evaluation_metrics": {"ROC_AUC": 0.94, "PR_AUC": 0.88}
}
print("Model Card Keys:", list(model_card.keys()))
```

---
**Next Chapter:** [21_PERFORMANCE_SCALABILITY.md](21_PERFORMANCE_SCALABILITY.md)
