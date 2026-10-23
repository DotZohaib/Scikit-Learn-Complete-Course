# Part 18: Neural Networks in Scikit-learn (`sklearn.neural_network`)

Is chapter mein hum **Artificial Neural Networks (ANNs)** ke bunyadi concepts (Neurons, Weights, Biases, Activation Functions, Backpropagation) aur Scikit-learn ke **`MLPClassifier`** aur **`MLPRegressor`** (Multi-Layer Perceptron) ko tafseel se seekhenge ([Scikit-learn Neural Network Models Guide](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)).

---

## 1. What Is an Artificial Neural Network? (Neurons, Weights, Biases & Activations)

### A. Intuitive Explanation & Mathematical Formula
Ek **Artificial Neuron (Perceptron)** bilkul `LogisticRegression` ya `LinearRegression` ki tarah inputs ($x_1, x_2, \dots, x_m$) leta hai, unhein **Weights ($w_i$)** se zarb (multiply) deta hai, **Bias ($b$)** jama karta hai, aur phir ek **Non-Linear Activation Function $g(\cdot)$** se guzarta hai:
$$a = g\left(\sum_{i=1}^{m} w_i x_i + b\right)$$
Jab hum aise bohot saare neurons ki **Hidden Layers** ek ke baad ek jor dete hain, to usay **Multi-Layer Perceptron (MLP)** kehte hain!

### B. Activation Functions in Scikit-learn (`activation=...`)
Agar Hidden Layers mein non-linear activation function na ho, to 10 layers ka neural network bhi ek simple linear model ke barabar reh jata hai!
1. **`'relu'` (Rectified Linear Unit — Default & Best)**: $g(z) = \max(0, z)$. Negative numbers ko `0` kar deta hai aur positive ko waise hi jane deta hai. Tez-tareen training aur vanishing gradient se bachao!
2. **`'tanh'` (Hyperbolic Tangent)**: Output ko $-1.0$ aur $+1.0$ ke darmiyan dabata (squash karta) hai.
3. **`'logistic'` (Sigmoid)**: Output ko $0.0$ aur $1.0$ ke darmiyan rakhta hai.
4. **`'identity'`**: $g(z) = z$ (Linear activation).

### C. Loss Functions, Backpropagation & Solvers (`solver=...`)
- **Loss Function:** Classification mein **Cross-Entropy (`log_loss`)** aur Regression mein **Squared Error (`MSE`)** minimize kiya jata hai.
- **Solvers (Optimization Algorithms):**
  1. **`'adam'` (Default)**: Adaptive Moment Estimation—medium aur bare datasets (hazaron samples) ke liye sab se behtareen solver!
  2. **`'lbfgs'`**: Quasi-Newton optimizer—**Chote datasets ($< 1,000$ samples)** par `'adam'` se kahin zyada tez converge hota hai aur behtar accuracy deta hai!
  3. **`'sgd'`**: Classic Stochastic Gradient Descent (`learning_rate='constant'`, `'invscaling'`, ya `'adaptive'` ke sath).

---

## 2. Critical Requirements & Hyperparameters of `MLPClassifier` & `MLPRegressor`

1. **Mandatory Feature Scaling (`StandardScaler`):**
   - Neural networks weights ko gradient descent se update karte hain. Agar aap ne `StandardScaler` ya `MinMaxScaler` **nahi** lagaya, to gradients explode/vanish ho jayenge aur `MLP` buri tarah fail ho jayega!
2. **`hidden_layer_sizes` (Default `(100,)`):**
   - Tuple jo har hidden layer mein neurons ki tadad batata hai:
     - `(50,)`: 1 hidden layer with 50 neurons.
     - `(64, 32)`: 2 hidden layers (pehli mein 64 neurons, doosri mein 32 neurons).
3. **Overfitting Prevention (`alpha` & `early_stopping`):**
   - **`alpha` (Default `0.0001`)**: L2 Regularization (weight decay) penalty. Overfitting rokne ke liye `alpha=0.01` ya `0.1` barha dein.
   - **`early_stopping=True`**: Training data mein se `validation_fraction=0.1` (10%) alag rakh kar har epoch par validation score check karta hai; agar `n_iter_no_change=10` epochs tak validation score behtar na ho, to training waheen rok deta hai!
4. **Handling `ConvergenceWarning`:**
   - Default `max_iter=200` aksar chota parta hai. Isay `max_iter=500` ya `1000` karein, data ko `StandardScaler()` se scale karein, ya `early_stopping=True` lagayein.

### Important Clarification: Scikit-learn vs PyTorch / TensorFlow
> **Honest Scope:** Scikit-learn ka `MLPClassifier` aur `MLPRegressor` sirf **CPU-based Feedforward Tabular Neural Networks** ke liye hai. Yeh GPU acceleration, Convolutional Neural Networks (CNNs for Images), Transformers (LLMs), ya Dropout layers support **nahi** karta. Large-scale Deep Learning ke liye **PyTorch**, **TensorFlow/Keras**, ya **JAX** use karein!

---

## 3. Complete Runnable Code: `MLPClassifier` & `MLPRegressor` with Early Stopping & Architecture Comparison

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, root_mean_squared_error

# =====================================================================
# 1. MLPClassifier on Breast Cancer Dataset (Comparing Architectures & Solvers)
# =====================================================================
bc = load_breast_cancer()
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    bc.data, bc.target, test_size=0.2, random_state=42, stratify=bc.target
)

mlp_configs = {
    "1. MLP (1 Layer: 50, Adam, alpha=1e-4)": MLPClassifier(
        hidden_layer_sizes=(50,), activation="relu", solver="adam",
        alpha=1e-4, max_iter=500, random_state=42
    ),
    "2. MLP (2 Layers: 64->32, Adam, alpha=0.1)": MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=0.1, max_iter=500, random_state=42
    ),
    "3. MLP (Early Stopping Enabled)": MLPClassifier(
        hidden_layer_sizes=(64, 32), solver="adam", early_stopping=True,
        validation_fraction=0.15, max_iter=500, random_state=42
    ),
    "4. MLP (L-BFGS Solver for Small Data)": MLPClassifier(
        hidden_layer_sizes=(30,), solver="lbfgs", alpha=1.0,
        max_iter=500, random_state=42
    )
}

print("=== 1. MLPClassifier Architecture & Solver Comparison ===")
for name, mlp in mlp_configs.items():
    pipe = make_pipeline(StandardScaler(), mlp)
    pipe.fit(Xc_tr, yc_tr)
    preds = pipe.predict(Xc_te)
    probas = pipe.predict_proba(Xc_te)[:, 1]
    fitted_mlp = pipe.named_steps["mlpclassifier"]
    print(f"{name:42s} | Acc: {accuracy_score(yc_te, preds):.4f} | AUC: {roc_auc_score(yc_te, probas):.4f} | Epochs: {fitted_mlp.n_iter_}")

# Inspect learned weight matrices shapes (coefs_)
sample_mlp = mlp_configs["2. MLP (2 Layers: 64->32, Adam, alpha=0.1)"]
print("\nLearned Weight Matrix Shapes for (30 inputs -> 64 -> 32 -> 1 output):")
print([w.shape for w in sample_mlp.coefs_])

# =====================================================================
# 2. MLPRegressor on Diabetes Dataset
# =====================================================================
diab = load_diabetes()
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(diab.data, diab.target, test_size=0.2, random_state=42)

mlp_reg_pipe = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(64, 32), solver="lbfgs", alpha=10.0, max_iter=500, random_state=42)
)
mlp_reg_pipe.fit(Xr_tr, yr_tr)
yr_pred = mlp_reg_pipe.predict(Xr_te)

print("\n=== 2. MLPRegressor Evaluation ===")
print(f"Test RMSE: {root_mean_squared_error(yr_te, yr_pred):.2f} | Test R2: {r2_score(yr_te, yr_pred):.4f}")
```

### Expected Output
```text
=== 1. MLPClassifier Architecture & Solver Comparison ===
1. MLP (1 Layer: 50, Adam, alpha=1e-4)     | Acc: 0.9737 | AUC: 0.9940 | Epochs: 278
2. MLP (2 Layers: 64->32, Adam, alpha=0.1) | Acc: 0.9825 | AUC: 0.9957 | Epochs: 221
3. MLP (Early Stopping Enabled)            | Acc: 0.9649 | AUC: 0.9921 | Epochs: 28
4. MLP (L-BFGS Solver for Small Data)      | Acc: 0.9737 | AUC: 0.9947 | Epochs: 46

Learned Weight Matrix Shapes for (30 inputs -> 64 -> 32 -> 1 output):
[(30, 64), (64, 32), (32, 1)]

=== 2. MLPRegressor Evaluation ===
Test RMSE: 53.68 | Test R2: 0.4556
```

---

## 4. Practice Exercises & Solutions

### Exercises
1. **Beginner:** Kya farq parta hai agar aap `MLPClassifier` ko `load_breast_cancer()` par **bina `StandardScaler`** ke aur **`StandardScaler` ke sath** chalayein? Dono ki accuracy compare karein.
2. **Intermediate:** `MLPClassifier` mein `hidden_layer_sizes=(16, 8)` rakhein aur fit hone ke baad `mlp.n_layers_` aur `mlp.out_activation_` attributes print karein.
3. **Challenge:** `MLPClassifier` ko `.partial_fit(X_batch, y_batch, classes=[0, 1])` ke zariye 3 mini-batches mein incrementally train karein.

### Solutions
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

bc = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(bc.data, bc.target, test_size=0.2, random_state=42, stratify=bc.target)

# 1. Beginner
mlp_unscaled = MLPClassifier(max_iter=300, random_state=42).fit(X_tr, y_tr)
sc = StandardScaler()
mlp_scaled = MLPClassifier(max_iter=300, random_state=42).fit(sc.fit_transform(X_tr), y_tr)
print(f"Unscaled Acc: {mlp_unscaled.score(X_te, y_te):.4f} vs Scaled Acc: {mlp_scaled.score(sc.transform(X_te), y_te):.4f}")

# 2. Intermediate
mlp_arch = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=42).fit(sc.transform(X_tr), y_tr)
print("Total Layers (input + 2 hidden + output):", mlp_arch.n_layers_, "| Output Activation:", mlp_arch.out_activation_)

# 3. Challenge
mlp_online = MLPClassifier(random_state=42)
X_tr_s = sc.transform(X_tr)
for i in range(0, len(X_tr_s), 150):
    mlp_online.partial_fit(X_tr_s[i:i+150], y_tr[i:i+150], classes=[0, 1])
print("Online MLP Test Score:", round(mlp_online.score(sc.transform(X_te), y_te), 4))
```

---
**Next Chapter:** [17_TEXT_ML_NLP.md](17_TEXT_ML_NLP.md)
