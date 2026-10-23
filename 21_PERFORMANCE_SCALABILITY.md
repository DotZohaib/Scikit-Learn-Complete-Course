# Part 23: Performance, Scalability, and Advanced Concepts

Jab aap ka dataset 500 rows se barh kar **5 Million rows (50 Lakh samples)** ya **100,000 columns** ka ho jata hai, to aam algorithms (`SVC(kernel='rbf')` ya `KNN`) ghanton lena shuru kar dete hain ya `MemoryError` de kar ruk jate hain! Is chapter mein hum [Scikit-learn Computational Performance & Scaling Strategies](https://scikit-learn.org/stable/computing/scaling_strategies.html) ke tamam 18 topics seekhenge.

---

## 1. Intuitive Computational Complexity (Time & Memory Costs)

Har Machine Learning Engineer ko pata hona chahiye ke algorithm ka training waqt Samples ($N$) aur Features ($P$) barhne ke sath kaise barhta hai:

| Algorithm Family | Training Time Complexity | Prediction Speed | Large Dataset Verdict ($N > 100,000$) |
| :--- | :--- | :--- | :--- |
| **`SGDClassifier` / `SGDRegressor` / `LinearSVC`** | $O(N \cdot P)$ (Linear in $N$!) | **Instant ($O(P)$)** | **Excellent!** Millions of rows ke liye best. |
| **`HistGradientBoostingClassifier/Regressor`** | $O(N_{\text{bins}} \cdot P + N)$ | **Very Fast** | **Excellent!** Large tabular data ke liye #1 choice. |
| **`RandomForest` / `ExtraTrees`** | $O(T \cdot P \cdot N \log N)$ | Moderate ($O(T \cdot \text{depth})$) | Good with `n_jobs=-1`, uses more RAM. |
| **`KNeighborsClassifier` (KNN)** | $O(1)$ (Lazy fit) | **Very Slow ($O(N \cdot P)$!)** | **Avoid on large $N$** (Prediction time aur RAM bohot zyada lagti hai). |
| **`SVC(kernel='rbf')` / `SVR` / `KernelPCA`** | **$O(N^2 \cdot P)$ to $O(N^3 \cdot P)$!** | Slow ($O(N_{\text{sv}} \cdot P)$) | **Strictly Avoid for $N > 20,000$!** (Use `LinearSVC` or `Nystroem` + `SGDClassifier` instead). |

---

## 2. Out-of-Core (Incremental) Learning with `.partial_fit()`

### A. What Is Out-of-Core Learning?
Jab dataset itna bara ho (maslan 50 GB CSV file) ke aap ke computer ki RAM (8 GB ya 16 GB) mein ek sath load hi na ho sake!
- **Solution:** File ko chote tukron (**Chunks / Mini-Batches**, maslan 10,000 rows at a time) mein `pd.read_csv(..., chunksize=10000)` se parhein aur Scikit-learn ke incremental estimators par **`.partial_fit(X_chunk, y_chunk)`** chalayein!

### B. Complete List of Scikit-learn Estimators Supporting `.partial_fit()`
1. **Classification:** `SGDClassifier`, `Perceptron`, `PassiveAggressiveClassifier`, `MultinomialNB`, `ComplementNB`, `BernoulliNB`, `GaussianNB`, `MLPClassifier`. *(Note: Pehli dafa `.partial_fit()` chalate waqt `classes=np.array([0, 1])` pass karna LAZMI hai taake model ko shuru mein hi tamam classes ka pata ho!)*
2. **Regression:** `SGDRegressor`, `PassiveAggressiveRegressor`, `MLPRegressor`.
3. **Clustering & Decomposition:** `MiniBatchKMeans`, `Birch`, `IncrementalPCA`, `MiniBatchDictionaryLearning`, `MiniBatchNMF`.
4. **Incremental Preprocessing:** `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, aur stateless `HashingVectorizer` (text ke liye)!

---

## 3. Parallelism (`n_jobs`), `warm_start`, `Pipeline(memory=...)` & Numerical Stability

1. **`n_jobs=-1` (Multi-Core Parallelism):**
   - `RandomForest`, `ExtraTrees`, `GridSearchCV`, `cross_val_score`, aur `KNN` mein `n_jobs=-1` dene se tamam CPU cores kaam par lag jate hain.
   - **Nested Parallelism Trap:** Agar aap ne `RandomForestClassifier(n_jobs=-1)` ko `GridSearchCV(..., n_jobs=-1)` ke andar rakh diya, to dono CPU threads ke liye aapas mein larenge (oversubscription)! Behtar hai ke sirf outer `GridSearchCV(n_jobs=-1)` par lagayein.
2. **`warm_start=True` and Its Limitations:**
   - `RandomForest(n_estimators=50, warm_start=True)` fit karne ke baad agar aap `rf.n_estimators = 100` kar ke dobara `.fit()` chalayein, to yeh pehle 50 trees ko mitaane ke bajaye sirf 50 **naye** trees add karega!
   - **Limitation:** `warm_start` naye data chunks ke liye `.partial_fit()` ka mutabadil **nahi** hai—yeh **usi same dataset** par trees ya iterations barhane ke liye hota hai.
3. **Sparse vs Dense Data & `float32` Memory Saving:**
   - Default NumPy float `np.float64` (8 bytes per number) hota hai. `X.astype(np.float32)` karne se RAM ka kharcha **foran aadha (50% reduction)** ho jata hai aur `HistGradientBoosting` / `SGD` aur tez chalte hain!
4. **Approximate Kernel Methods (`RBFSampler` & `Nystroem`):**
   - Agar $N = 100,000$ samples par RBF SVM jaisa non-linear boundary chahiye, to `SVC(kernel='rbf')` crash ho jayega. Is ka professional ilaj `make_pipeline(StandardScaler(), Nystroem(n_components=300), SGDClassifier())` hai jo 50x tez chalta hai!

---

## 4. Complete Runnable Code: Out-of-Core Streaming `.partial_fit()`, `Nystroem` Approximation & `warm_start`

```python
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# =====================================================================
# 1. Out-of-Core Incremental Learning Demo (Streaming 5 Chunks with partial_fit)
# =====================================================================
X_stream, y_stream = make_classification(n_samples=2500, n_features=20, random_state=42)
X_test, y_test = X_stream[2000:], y_stream[2000:]

inc_scaler = StandardScaler()
inc_clf = SGDClassifier(loss="log_loss", random_state=42)
all_classes = np.array([0, 1])

batch_size = 400
print("=== 1. Out-of-Core Streaming Training via .partial_fit() ===")
for step, start_idx in enumerate(range(0, 2000, batch_size), start=1):
    X_batch = X_stream[start_idx : start_idx + batch_size].astype(np.float32)
    y_batch = y_stream[start_idx : start_idx + batch_size]

    # Step A: Incrementally update StandardScaler mean & variance
    inc_scaler.partial_fit(X_batch)
    X_batch_scaled = inc_scaler.transform(X_batch)

    # Step B: Incrementally update SGDClassifier weights
    inc_clf.partial_fit(X_batch_scaled, y_batch, classes=all_classes)

    current_acc = inc_clf.score(inc_scaler.transform(X_test.astype(np.float32)), y_test)
    print(f"  After Chunk {step} ({start_idx + batch_size:4d} samples seen) -> Test Accuracy: {current_acc:.4f}")

# =====================================================================
# 2. Exact RBF SVC vs Fast Nystroem Kernel Approximation + SGDClassifier
# =====================================================================
t0 = time.perf_counter()
exact_svc = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42))
exact_svc.fit(X_stream[:2000], y_stream[:2000])
svc_time = (time.perf_counter() - t0) * 1000

t1 = time.perf_counter()
approx_rbf = make_pipeline(
    StandardScaler(),
    Nystroem(kernel="rbf", n_components=150, random_state=42),
    SGDClassifier(loss="hinge", random_state=42)
)
approx_rbf.fit(X_stream[:2000], y_stream[:2000])
approx_time = (time.perf_counter() - t1) * 1000

print("\n=== 2. Exact SVC(rbf) vs Nystroem + SGDClassifier Benchmark ===")
print(f"Exact SVC(kernel='rbf')   -> Test Acc: {exact_svc.score(X_test, y_test):.4f} | Time: {svc_time:.1f} ms")
print(f"Nystroem + SGDClassifier  -> Test Acc: {approx_rbf.score(X_test, y_test):.4f} | Time: {approx_time:.1f} ms")

# =====================================================================
# 3. warm_start=True Demo (Growing a RandomForest from 20 to 60 trees)
# =====================================================================
rf_warm = RandomForestClassifier(n_estimators=20, warm_start=True, random_state=42)
rf_warm.fit(X_stream[:1000], y_stream[:1000])
print(f"\n=== 3. warm_start Demo ===")
print(f"Initial Trees: {len(rf_warm.estimators_)} -> Test Acc: {rf_warm.score(X_test, y_test):.4f}")

rf_warm.n_estimators = 60  # Add 40 more trees without retraining the first 20!
rf_warm.fit(X_stream[:1000], y_stream[:1000])
print(f"Expanded Trees: {len(rf_warm.estimators_)} -> Test Acc: {rf_warm.score(X_test, y_test):.4f}")
```

### Expected Output
```text
=== 1. Out-of-Core Streaming Training via .partial_fit() ===
  After Chunk 1 ( 400 samples seen) -> Test Accuracy: 0.7980
  After Chunk 2 ( 800 samples seen) -> Test Accuracy: 0.8220
  After Chunk 3 (1200 samples seen) -> Test Accuracy: 0.8380
  After Chunk 4 (1600 samples seen) -> Test Accuracy: 0.8440
  After Chunk 5 (2000 samples seen) -> Test Accuracy: 0.8520

=== 2. Exact SVC(rbf) vs Nystroem + SGDClassifier Benchmark ===
Exact SVC(kernel='rbf')   -> Test Acc: 0.8920 | Time: 145.2 ms
Nystroem + SGDClassifier  -> Test Acc: 0.8840 | Time: 38.4 ms

=== 3. warm_start Demo ===
Initial Trees: 20 -> Test Acc: 0.8860
Expanded Trees: 60 -> Test Acc: 0.9020
```

---

## 5. Practice Exercises & Solutions

### Exercises
1. **Beginner:** Ek NumPy array `X = np.ones((1000, 100), dtype=np.float64)` ka memory size (`X.nbytes`) `float64` aur `float32` mein compare karein.
2. **Intermediate:** `IncrementalPCA(n_components=3, batch_size=50)` ko `(200, 10)` data par `.partial_fit()` ke zariye 4 batches mein fit karein.
3. **Challenge:** `HashingVectorizer(n_features=2**10)` aur `MultinomialNB()` ko mila kar ek streaming text classifier banayein jo `.partial_fit()` se 2 text batches seekhe!

### Solutions
```python
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Beginner
X64 = np.ones((1000, 100), dtype=np.float64)
X32 = X64.astype(np.float32)
print(f"Memory -> float64: {X64.nbytes} bytes vs float32: {X32.nbytes} bytes (50% saved!)")

# 2. Intermediate
ipca = IncrementalPCA(n_components=3)
X_d = np.random.RandomState(42).normal(size=(200, 10))
for i in range(0, 200, 50):
    ipca.partial_fit(X_d[i:i+50])
print("IncrementalPCA Explained Variance Ratio:", ipca.explained_variance_ratio_.round(3))

# 3. Challenge: Streaming Text Classifier
hv = HashingVectorizer(n_features=2**10, alternate_sign=False)  # alternate_sign=False ensures non-negative for MultinomialNB!
nb = MultinomialNB()
nb.partial_fit(hv.transform(["free lottery cash", "meeting tomorrow"]), [1, 0], classes=[0, 1])
nb.partial_fit(hv.transform(["win prize money", "project report"]), [1, 0])
print("Streaming NB Prediction for 'free prize':", nb.predict(hv.transform(["free prize"])))
```

---
**Next Chapter:** [22_SCIKIT_LEARN_API_REFERENCE.md](22_SCIKIT_LEARN_API_REFERENCE.md)
