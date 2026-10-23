# Part 9: Unsupervised Learning and Clustering (`sklearn.cluster`, `sklearn.mixture`)

Jab hamare paas sirf **Features ($X$)** hon aur koi **Target Labels ($y$)** mojood na hon, to milte-julte samples ke groups banane ke amal ko **Clustering** kehte hain ([Scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)). Is chapter mein hum tamam 12 major clustering algorithms, cluster evaluation metrics, aur ek mukammal **Customer Segmentation Project** seekhenge.

---

## 1. Why Feature Scaling Is Mandatory for Clustering
Clustering algorithms (`KMeans`, `DBSCAN`, `AgglomerativeClustering`, `GaussianMixture`) do samples ke darmiyan **Geometric Distance (Euclidean Distance)** naapte hain:
$$d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots}$$
Agar ek column `Annual_Orders` (`1` se `50`) hai aur doosra `Annual_Spend_PKR` (`5,000` se `500,000`) hai, to bina **`StandardScaler`** lagaye algorithm sirf `Annual_Spend_PKR` ko dekhega! Is liye clustering se pehle scaling **100% lazmi** hai.

---

## 2. All 12 Scikit-learn Clustering & Mixture Algorithms Explained

| Algorithm | How It Works Intuitively | Cluster Shape | Requires `n_clusters`? | Handles Noise/Outliers? |
| :--- | :--- | :--- | :--- | :--- |
| **1. `KMeans`** | $K$ centroids rakhta hai, points ko qareebi centroid deta hai, aur centroids ko cluster ke center mein shift karta hai jab tak inertia ($\sum \|x_i - \mu_k\|^2$) minimum na ho jaye. | Round / Convex (Gol) | Yes (`n_clusters=k`) | No (Outliers drag centroids) |
| **2. `MiniBatchKMeans`** | Poore dataset ke bajaye chote random batches (`batch_size=1024`) par `KMeans` chalata hai—millions of rows ke liye bohat tez! | Round / Convex | Yes | No |
| **3. `DBSCAN`** | **Density-Based**: Jahan `eps` radius ke andar kam az kam `min_samples` parosi hon wahan cluster banata hai; akele door points ko **Noise (`label = -1`)** qarar deta hai! | Arbitrary (Har shakal: chand, daira) | **No!** (Auto-discovers) | **Yes (`-1` label)** |
| **4. `HDBSCAN`** *(v1.3+)* | Hierarchical DBSCAN jo alag-alag densities ke clusters khud dhoond leta hai bina `eps` tune kiye! | Arbitrary & Variable Density | **No!** (`min_cluster_size`) | **Yes (`-1` label)** |
| **5. `OPTICS`** | `DBSCAN` ka generalization jo reachability graph banata hai taake variable density clusters pakray ja saken. | Arbitrary & Variable Density | No | Yes (`-1` label) |
| **6. `AgglomerativeClustering`** | **Bottom-Up Hierarchical**: Shuru mein har point apna cluster hota hai; phir `linkage` (`'ward'`, `'complete'`, `'average'`) ke mutabiq qareebi clusters jurte jate hain (Dendrogram). | Depends on linkage | Yes (or `distance_threshold`) | No |
| **7. `Birch`** | Memory-efficient Clustering Feature (CF) Tree banata hai—bohat bare datasets ke liye mufeed. | Spherical | Yes | Optional |
| **8. `MeanShift`** | Density ke paharon ki choti (modes) ki taraf points ko khiska kar clusters dhoondta hai (`bandwidth`). | Smooth blobs | **No!** | No |
| **9. `SpectralClustering`** | Points ka similarity graph bana kar Graph Laplacian ke eigenvectors par KMeans lagata hai (ring/circle data ke liye best). | Non-convex / Rings | Yes | No |
| **10. `AffinityPropagation`** | Points ek doosre ko messages bhej kar apne numainday (**Exemplars**) chunte hain (slow on large $N$, $O(N^2)$). | Round / Exemplar | **No!** | No |
| **11. `GaussianMixture` (GMM)** | (`sklearn.mixture`) **Soft Clustering**: Expectation-Maximization (EM) ke zariye har cluster ko ek Elliptical Gaussian distribution manta hai aur `.predict_proba(X)` se har cluster ki probability deta hai! | Elliptical (Tirchay/Lambe) | Yes (`n_components`) | Via density threshold |
| **12. `BayesianGaussianMixture`** | Variational inference ke zariye ghair-zaroori clusters ka weight khud zero kar deta hai (auto-selects effective components). | Elliptical | Max upper bound (`n_components`) | Via density |

> **Important Note on `.predict()` in Clustering:** `KMeans`, `MiniBatchKMeans`, `Birch`, `MeanShift`, aur `GaussianMixture` mein naye test points ke liye `.predict(X_new)` mojood hai. Lekin **`DBSCAN`, `AgglomerativeClustering`, `SpectralClustering`, aur `OPTICS` mein `.predict()` method NAHI hota**—woh sirf `.fit_predict(X)` ke zariye mojooda dataset ko cluster karte hain!

---

## 3. Choosing the Number of Clusters ($K$) & Clustering Evaluation Metrics

### Why Clustering Evaluation Is Different from Classification Accuracy
- **Important Trap:** Agar kisi dataset mein classes `[0, 1]` hon aur `KMeans` pehle group ko Cluster `1` aur doosre ko Cluster `0` naam de de, to `accuracy_score` `0%` aayega halanke clustering 100% sahi hui hai! Kyunki cluster numbers (`0, 1, 2`) sirf arbitrary group IDs hote hain, class labels nahi!
- **Internal Metrics (Jab koi ground-truth label $y$ na ho — Real Unsupervised Case):**
  1. **Inertia (`kmeans.inertia_`) & The Elbow Method:** Cluster ke andar points ka apne center se squared distance. $K$ barhane se inertia hamesha kam hota hai; jahan graph mein kohni (**Elbow / موڑ**) bane woh $K$ chunein.
  2. **Silhouette Score (`silhouette_score(X, labels)`)**: Range **$-1.0$ se $+1.0$**. Har point ke apne cluster se qareeb hone ($a$) aur doosre qareebi cluster se door hone ($b$) ka tanasub: $\frac{b - a}{\max(a, b)}$. **$+1$ ke qareeb = Best**!
  3. **Calinski–Harabasz Index (`calinski_harabasz_score`)**: Between-cluster dispersion aur within-cluster dispersion ka ratio (**Higher is better**).
  4. **Davies–Bouldin Index (`davies_bouldin_score`)**: Clusters ki average similarity (**Lower is better, $0$ is best**).

---

## 4. Complete Runnable Code: Pakistani E-Commerce Customer Segmentation Project

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

# 1. Create Synthetic Pakistani E-Commerce Customer Dataset (n=300)
# Features: [Monthly_Orders, Avg_Cart_PKR, App_Minutes_Per_Week]
rng = np.random.RandomState(42)

# Segment A: Window Shoppers (Low orders, low spend, high app browsing)
seg_a = rng.normal(loc=[1.5, 1200, 180], scale=[0.5, 300, 25], size=(100, 3))
# Segment B: Regular Families (Medium orders, medium spend, medium browsing)
seg_b = rng.normal(loc=[6.0, 4500, 90], scale=[1.2, 600, 15], size=(100, 3))
# Segment C: VIP Bulk Buyers (High orders, high spend, quick checkout)
seg_c = rng.normal(loc=[14.0, 12000, 45], scale=[2.0, 1500, 10], size=(100, 3))

X_raw = np.vstack([seg_a, seg_b, seg_c])
df_cust = pd.DataFrame(X_raw, columns=["Monthly_Orders", "Avg_Cart_PKR", "App_Minutes"])

# 2. Scale Features (Mandatory!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cust)

# 3. Elbow Method + Silhouette Search for K in KMeans (K=2 to 6)
print("=== 1. Choosing K for KMeans (Elbow & Silhouette) ===")
for k in range(2, 6):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    sil = silhouette_score(X_scaled, km.labels_)
    db = davies_bouldin_score(X_scaled, km.labels_)
    ch = calinski_harabasz_score(X_scaled, km.labels_)
    print(f"K={k} | Inertia: {km.inertia_:6.1f} | Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f} | Calinski: {ch:6.1f}")

# 4. Fit Final KMeans (K=3) and Profile Customer Segments in Original PKR Units
best_km = KMeans(n_clusters=3, n_init=10, random_state=42)
df_cust["Cluster"] = best_km.fit_predict(X_scaled)

print("\n=== 2. Customer Segment Profiles (Original Units) ===")
profile = df_cust.groupby("Cluster").mean().round(1)
profile["Customer_Count"] = df_cust["Cluster"].value_counts()
print(profile)

# 5. Compare DBSCAN, HDBSCAN, Agglomerative, and GaussianMixture (Soft Clustering)
dbscan_labels = DBSCAN(eps=0.6, min_samples=5).fit_predict(X_scaled)
hdbscan_labels = HDBSCAN(min_cluster_size=15).fit_predict(X_scaled)
agg_labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X_scaled)
gmm = GaussianMixture(n_components=3, random_state=42).fit(X_scaled)

print("\n=== 3. Alternative Clustering Algorithms Summary ===")
print("DBSCAN Clusters found (excluding -1 noise):", set(dbscan_labels) - {-1}, "| Noise count:", list(dbscan_labels).count(-1))
print("HDBSCAN Clusters found (excluding -1 noise):", set(hdbscan_labels) - {-1}, "| Noise count:", list(hdbscan_labels).count(-1))
print("Agglomerative Silhouette Score:", round(silhouette_score(X_scaled, agg_labels), 4))
print("GMM Soft Probabilities for Customer 0:", gmm.predict_proba(X_scaled[:1]).round(4))
```

### Expected Output
```text
=== 1. Choosing K for KMeans (Elbow & Silhouette) ===
K=2 | Inertia:  325.4 | Silhouette: 0.5842 | Davies-Bouldin: 0.5912 | Calinski:  526.4
K=3 | Inertia:   62.8 | Silhouette: 0.7815 | Davies-Bouldin: 0.3105 | Calinski: 1982.3
K=4 | Inertia:   51.2 | Silhouette: 0.6310 | Davies-Bouldin: 0.7420 | Calinski: 1630.1
K=5 | Inertia:   42.9 | Silhouette: 0.4920 | Davies-Bouldin: 0.9810 | Calinski: 1475.8

=== 2. Customer Segment Profiles (Original Units) ===
         Monthly_Orders  Avg_Cart_PKR  App_Minutes  Customer_Count
Cluster                                                           
0                   6.0        4512.4         89.8             100
1                  14.1       11985.2         44.9             100
2                   1.5        1195.8        180.5             100

=== 3. Alternative Clustering Algorithms Summary ===
DBSCAN Clusters found (excluding -1 noise): {0, 1, 2} | Noise count: 4
HDBSCAN Clusters found (excluding -1 noise): {0, 1, 2} | Noise count: 2
Agglomerative Silhouette Score: 0.7815
GMM Soft Probabilities for Customer 0: [[0. 0. 1.]]
```
*(Interpretation: $K=3$ par Silhouette Score sab se zyada (`0.7815`) aur Davies–Bouldin sab se kam (`0.3105`) hai! Aur Cluster Profiles se saaf pata chal gaya ke Cluster 1 hamare VIP Bulk Buyers hain, Cluster 0 Regular Families hain, aur Cluster 2 Window Shoppers hain jinhein discount coupon bhej kar buyer banaya ja sakta hai!)*

---

## 5. Practice Exercises & Solutions

### Exercises
1. **Beginner:** `sklearn.datasets.make_moons(n_samples=200, noise=0.05, random_state=42)` generate karein aur us par `KMeans(n_clusters=2)` aur `DBSCAN(eps=0.3, min_samples=5)` dono chala kar dekhein ke kaunsa algorithm chand (crescent moon) ki shakal sahi pakarta hai.
2. **Intermediate:** `GaussianMixture(n_components=3, random_state=42)` ko Iris features (`load_iris().data`) par fit karein aur `.aic(X)` aur `.bic(X)` (Akaike/Bayesian Information Criteria—lower is better) print karein.
3. **Challenge:** `MiniBatchKMeans(n_clusters=3, batch_size=64, random_state=42)` ko scaled Iris dataset par `.partial_fit()` ke zariye 2 batches mein train karein.

### Solutions
```python
from sklearn.datasets import make_moons, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# 1. Beginner (make_moons: DBSCAN wins on non-convex shapes!)
X_m, y_m = make_moons(n_samples=200, noise=0.05, random_state=42)
X_ms = StandardScaler().fit_transform(X_m)
km_ari = adjusted_rand_score(y_m, KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X_ms))
db_ari = adjusted_rand_score(y_m, DBSCAN(eps=0.3, min_samples=5).fit_predict(X_ms))
print(f"Moons ARI -> KMeans: {km_ari:.3f} vs DBSCAN: {db_ari:.3f}")  # DBSCAN gets 1.000!

# 2. Intermediate
X_iris = StandardScaler().fit_transform(load_iris().data)
gmm = GaussianMixture(n_components=3, random_state=42).fit(X_iris)
print(f"GMM AIC: {gmm.aic(X_iris):.1f}, BIC: {gmm.bic(X_iris):.1f}")

# 3. Challenge
mbk = MiniBatchKMeans(n_clusters=3, batch_size=64, random_state=42)
mbk.partial_fit(X_iris[:75])
mbk.partial_fit(X_iris[75:])
print("MiniBatchKMeans Cluster Centers shape:", mbk.cluster_centers_.shape)
```

---
**Next Chapter:** [08_DIMENSIONALITY_REDUCTION.md](08_DIMENSIONALITY_REDUCTION.md)
