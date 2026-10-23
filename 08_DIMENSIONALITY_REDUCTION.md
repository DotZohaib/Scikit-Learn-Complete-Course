# Part 10: Dimensionality Reduction and Feature Learning (`sklearn.decomposition`, `sklearn.manifold`, `sklearn.cluster`)

Jab hamare dataset mein darjanon ya hazaron columns (`n_features`) hon, to na hum uska 2D/3D graph bana sakte hain aur na hi distance-based models (`KNN`, `KMeans`) sahi kaam karte hain. Is chapter mein hum **Curse of Dimensionality** aur Scikit-learn ke **tamam 16 Dimensionality Reduction & Manifold Learning algorithms** ko seekhenge ([Decomposing Signals in Components](https://scikit-learn.org/stable/modules/decomposition.html) & [Manifold Learning](https://scikit-learn.org/stable/modules/manifold.html)).

---

## 1. What Dimensionality Means & The Curse of Dimensionality

- **Dimensionality Kya Hai?** Dataset mein input features (columns) ki tadad (`n_features`).
- **The Curse of Dimensionality (Zyada Columns Ki Museebat):**
  1. Jab dimensions barhte hain (maslan 2 columns se 500 columns), to space ka volume exponential raftaar se barhta hai aur aap ke samples ek doosre se bohot door-door bikhray hue (**Sparse**) ho jate hain.
  2. High dimensions mein sabse qareebi parosi aur sabse door wale point ka Euclidean distance taqreeban barabar hone lagta hai, jis se `KNN` aur `KMeans` andhe ho jate hain!
  3. Bohat saare columns aapas mein duplicate maloomat (**Multicollinearity**) aur noise rakhte hain, jis se model **Overfit** hota hai aur training slow ho jati hai.

### Feature Reduction (Extraction) vs Feature Selection
- **Feature Selection (Part 11):** Mojooda 100 columns mein se 10 sab se ahem asli columns (`['area_marla', 'city', ...]`) chunna aur baqi 90 ko drop kar dena. Asli column ke naam mehfooz rehte hain!
- **Dimensionality Reduction / Feature Extraction (This Chapter):** Mojooda 100 columns ko mathematical tareeqe se nichor (combine) kar 10 **naye synthetic columns (`PC1, PC2, ...`)** banana jin mein 100 columns ki 95% maloomat samayi hui ho!

---

## 2. Linear Decomposition Methods (`sklearn.decomposition`)

Linear methods data ki directions (axes) ko rotate ya factorize karte hain aur in mein `.transform(X_test)` mojood hota hai, is liye yeh **ML Pipelines mein Preprocessing** ke liye behtareen hain:

| Algorithm | How It Works & When to Use It | Key Parameters & Attributes |
| :--- | :--- | :--- |
| **1. `PCA` (Principal Component Analysis)** | Data mein sab se zyada phailao (**Maximum Variance**) wali orthogonal directions (**Principal Components**) dhoondta hai via SVD. **Scaling (`StandardScaler`) LAZMI hai!** | `n_components` (int `2` ya float `0.95` for 95% variance), `explained_variance_ratio_` |
| **2. `IncrementalPCA`** | Jab dataset itna bara ho ke RAM mein fit na aaye, to batches mein (`partial_fit`) PCA chalata hai. | `n_components`, `batch_size` |
| **3. `KernelPCA`** | **Kernel Trick** (`kernel='rbf'`, `'poly'`, `'cosine'`) ke zariye **Non-linear** data ko unroll karta hai. (`fit_inverse_transform=True` se denoising bhi kar sakta hai). | `n_components`, `kernel`, `gamma` |
| **4. `TruncatedSVD` (LSA)** | **Sparse Matrices (jaise `TfidfVectorizer` text matrices)** par bina data center kiye seedha SVD chalata hai! (`PCA` sparse matrix center nahi kar sakta, `TruncatedSVD` kar sakta hai). | `n_components`, `n_iter`, `explained_variance_ratio_` |
| **5. `SparsePCA` & 6. `MiniBatchSparsePCA`** | Aam PCA mein har component tamam original features ka mixture hota hai. `SparsePCA` L1 penalty laga kar aise components banata hai jin mein zyadatar weights `0` hon (zyada interpretable!). | `n_components`, `alpha` (sparsity control) |
| **7. `FactorAnalysis`** | Statistical model jo maanta hai ke observed features chand chhupay hue **Latent Factors** + har feature ke apne alag noise variance se paida hue hain. | `n_components`, `rotation` (`'varimax'`, `'quartimax'`) |
| **8. `FastICA` (Independent Component Analysis)** | **Cocktail Party Problem Solver**: Variance ke bajaye **Statistically Independent (Non-Gaussian)** sources alag karta hai (maslan ek kamray mein 3 maikrofon ki recording se 3 speakers ki alag aawazein nikalna, ya EEG brain signals saaf karna). | `n_components`, `whiten='unit-variance'` |
| **9. `NMF` (Non-Negative Matrix Factorization)** | Sirf **Non-Negative ($X \ge 0$)** data (Images, Audio spectrograms, Text TF-IDF) ko do non-negative matrices $W \times H$ mein torta hai. Topic modeling aur additive parts seekhne ke liye best! | `n_components`, `init='nndsvda'` |
| **10. `DictionaryLearning`** | Sparse coding ke liye overcomplete basis dictionary seekhta hai (Image denoising aur compression mein mufeed). | `n_components`, `alpha` |

---

## 3. Nonlinear Manifold Learning & Hierarchical Feature Merging (`sklearn.manifold`, `sklearn.cluster`)

Manifold algorithms yeh maante hain ke high-dimensional data asal mein ek mure hue kagaz (**Swiss Roll / Nonlinear Manifold**) par para hai.

| Algorithm | How It Works Intuitively | Has `.transform(X_test)`? | Best Use Case |
| :--- | :--- | :--- | :--- |
| **11. `TSNE` (t-Distributed Stochastic Neighbor Embedding)** | High-D mein parosiyon ki probabilities ko Low-D (2D/3D) Student-t distribution se match karta hai. Clusters ko bohot khoobsurti se alag dikhata hai! | **NO!** (Only `fit_transform` for visualization!) | **2D Visualization ONLY** (Never use as pipeline preprocessor for test data!) |
| **12. `MDS` (Multidimensional Scaling)** | Tamam points ke aapas ke faaslon (pairwise distances) ko 2D mein barqarar rakhne ki koshish karta hai. | **NO!** (Only `fit_transform`) | Visualizing distance/dissimilarity matrices |
| **13. `SpectralEmbedding`** | Graph Laplacian ke zariye qareebi points ko 2D mein qareeb rakhta hai. | **NO!** (Only `fit_transform`) | Graph/network visualization |
| **14. `Isomap` (Isometric Mapping)** | Seedhe Euclidean faasle ke bajaye **Geodesic (Neighborhood Graph Shortest Path)** faasla naap kar MDS lagata hai. | **Yes** | Unrolling curved manifolds (Swiss roll) |
| **15. `LocallyLinearEmbedding` (LLE)** | Har point ko uske `n_neighbors` ke linear weights se zahir kar ke 2D mein reconstruct karta hai. | **Yes** | Preserving local geometry |
| **16. `FeatureAgglomeration`** (`sklearn.cluster`) | Rows (samples) ke bajaye **Columns (features) par Agglomerative Clustering** chala kar milte-julte columns ko merge kar deta hai! | **Yes** | Merging correlated pixels/features |

> **Critical Warning on `TSNE`:**
> 1. `TSNE` mein `.transform(X_test)` **nahi hota**—isay sirf EDA/Visualization ke liye use karein, classification pipeline ke andar preprocessing ke taur par nahi!
> 2. `TSNE` plot mein clusters ke darmiyan khali jagah (distance between clusters) aur cluster ka size be-maani hota hai, kyunki `TSNE` local neighborhoods ko expand/compress karta hai. `perplexity` (default `30`, must be $< n_{\text{samples}}$) zaroor check karein.

---

## 4. Selecting the Number of Components in PCA (`explained_variance_ratio_`)
PCA mein aap ko andaze se `n_components=5` dene ki zaroorat nahi! Agar aap **`PCA(n_components=0.95)`** (ek float $0 < n < 1$) dein, to Scikit-learn khud-ba-khud utne Principal Components chun leta hai jo original data ki **95% Variance (maloomat)** ko mehfooz rakh saken!

---

## 5. Complete Working Code: PCA Variance Selection, TruncatedSVD, NMF, Isomap, t-SNE & FeatureAgglomeration

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, NMF, FastICA
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import FeatureAgglomeration

# 1. Load Handwritten Digits Dataset (1797 samples, 64 pixel features [8x8 image])
digits = load_digits()
X = digits.data      # Shape: (1797, 64), values 0..16 (non-negative!)
y = digits.target    # Classes: 0..9

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA: Automatic component selection to retain 90% variance
pca_90 = PCA(n_components=0.90, random_state=42)
X_pca90 = pca_90.fit_transform(X_scaled)

print("=== 1. PCA Variance Analysis ===")
print(f"Original Features: {X.shape[1]} -> Components needed for 90% variance: {pca_90.n_components_}")
print(f"Total Explained Variance Retained: {pca_90.explained_variance_ratio_.sum()*100:.2f}%")
print("Top 5 Components Variance Ratio  :", np.round(pca_90.explained_variance_ratio_[:5], 4))

# 3. 2D Reduction Comparison across 6 Techniques (on subset of 400 samples for speed)
X_sub_scaled = X_scaled[:400]
X_sub_nonneg = X[:400]  # NMF requires non-negative input >= 0

pca_2d = PCA(n_components=2, random_state=42).fit_transform(X_sub_scaled)
svd_2d = TruncatedSVD(n_components=2, random_state=42).fit_transform(X_sub_nonneg)
nmf_2d = NMF(n_components=2, init="nndsvda", random_state=42, max_iter=500).fit_transform(X_sub_nonneg)
ica_2d = FastICA(n_components=2, random_state=42).fit_transform(X_sub_scaled)
iso_2d = Isomap(n_neighbors=10, n_components=2).fit_transform(X_sub_scaled)
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_sub_scaled)
fagg_8d = FeatureAgglomeration(n_clusters=8).fit_transform(X_sub_scaled)

print("\n=== 2. Output Shapes After Dimensionality Reduction (400 samples, 64 features) ===")
print("PCA (2D)                 :", pca_2d.shape)
print("TruncatedSVD (2D)        :", svd_2d.shape)
print("NMF (2D, Non-Negative)   :", nmf_2d.shape)
print("FastICA (2D)             :", ica_2d.shape)
print("Isomap (2D)              :", iso_2d.shape)
print("t-SNE (2D Visualization) :", tsne_2d.shape)
print("FeatureAgglomeration (8D):", fagg_8d.shape)
```

### Expected Output
```text
=== 1. PCA Variance Analysis ===
Original Features: 64 -> Components needed for 90% variance: 31
Total Explained Variance Retained: 90.05%
Top 5 Components Variance Ratio  : [0.1203 0.0956 0.0844 0.065  0.0486]

=== 2. Output Shapes After Dimensionality Reduction (400 samples, 64 features) ===
PCA (2D)                 : (400, 2)
TruncatedSVD (2D)        : (400, 2)
NMF (2D, Non-Negative)   : (400, 2)
FastICA (2D)             : (400, 2)
Isomap (2D)              : (400, 2)
t-SNE (2D Visualization) : (400, 2)
FeatureAgglomeration (8D): (400, 8)
```

---

## 6. Practice Exercises & Solutions

### Exercises
1. **Beginner:** `load_wine()` dataset ko `StandardScaler` se scale karein, `PCA(n_components=2)` lagayein, aur batayein ke pehle 2 components mil kar kitne feesad (`%`) variance explain karte hain.
2. **Intermediate:** `PCA(n_components=5)` ko digits dataset par fit karein aur `.inverse_transform()` chala kar dekhein ke compressed data ko wapas `64` columns mein reconstruct karne par shape kya aati hai.
3. **Challenge:** Ek `Pipeline` banayein jismein `StandardScaler` $\rightarrow$ `PCA(n_components=0.95)` $\rightarrow$ `LogisticRegression()` ho, aur usay `load_breast_cancer()` ke train/test split par evaluate karein.

### Solutions
```python
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# 1 & 2. Beginner & Intermediate
X_w = StandardScaler().fit_transform(load_wine().data)
pca2 = PCA(n_components=2).fit(X_w)
print(f"Wine 2D PCA Explained Variance: {pca2.explained_variance_ratio_.sum()*100:.2f}%")

X_rec = pca2.inverse_transform(pca2.transform(X_w))
print("Reconstructed Shape:", X_rec.shape)

# 3. Challenge
bc = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(bc.data, bc.target, test_size=0.2, random_state=42, stratify=bc.target)
pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95), LogisticRegression(random_state=42)).fit(X_tr, y_tr)
print("Components kept:", pipe.named_steps["pca"].n_components_, "| Test Acc:", round(pipe.score(X_te, y_te), 4))
```

---
**Next Chapter:** [09_FEATURE_SELECTION_ENGINEERING.md](09_FEATURE_SELECTION_ENGINEERING.md)
