# Technical Glossary: Machine Learning & Scikit-learn (English to Roman Urdu)

Is Glossary mein Machine Learning aur Scikit-learn ki **100+ sab se ahem technical terms** ko alphabetical order mein asaan **Roman Urdu** mein samjhaya gaya hai.

---

| English Technical Term | Roman Urdu Explanation & Practical Meaning |
| :--- | :--- |
| **Accuracy (`accuracy_score`)** | Kul predictions mein se sahi predictions ka tanasub ($\frac{TP+TN}{\text{Total}}$). Balanced classes ke liye munasib, lekin imbalanced data par dhoka deta hai. |
| **Activation Function** | Neural network (`MLP`) mein neuron ke output par lagne wala non-linear function (jaise `relu`, `tanh`, `logistic`) jo complex patterns seekhne ki salahiyat deta hai. |
| **AdaBoost (`AdaBoostClassifier`)** | Adaptive Boosting ensemble jo pichle model ki ghaltiyon (misclassified samples) ka weight barha kar agla model train karta hai. |
| **AffinityPropagation** | Graph/message-passing clustering algorithm jo data ke andar se numainda points (exemplars) khud chunta hai. |
| **AgglomerativeClustering** | Bottom-up hierarchical clustering jismein shuru mein har point apna cluster hota hai aur qareebi clusters step-by-step merge hote jate hain. |
| **Anomaly Detection** | Normal data se hat kar ghair-mamooli ya mashkook points (jaise fraud ya machine fault, label `-1`) ko pehchanna. |
| **Area Under the Curve (AUC)** | ROC curve (`roc_auc_score`) ya Precision-Recall curve (`average_precision_score`) ke neeche ka raqba ($0.5 = \text{random}, 1.0 = \text{perfect}$). |
| **Attribute (Learned Attribute `_`)** | Woh variable jo estimator `.fit()` chalne ke baad data se seekhta hai; Scikit-learn mein iske aakhir mein `_` hota hai (maslan `coef_`, `classes_`, `mean_`). |
| **Bagging (Bootstrap Aggregating)** | Training data ke random samples (with replacement) par parallel models bana kar unki predictions ka vote/average lena (jaise `RandomForest`). |
| **Balanced Accuracy** | Sensitivity (TPR) aur Specificity (TNR) ka ausat; imbalanced classification ke liye behtareen metric. |
| **Baseline Model (`DummyClassifier`)** | Ek sada tareen benchmark model (jo sirf majority class ya mean predict kare) taake pata chale ke asli ML model ne waqai kuch seekha hai ya nahi. |
| **Batch Inference** | Production mein hazaron rows ke table (CSV/Database) par ek sath saved pipeline se `.predict()` chalana. |
| **Bias (Statistical / Model Bias)** | Jab model bohot sada hone ki wajah se data ka asli talluq na pakar sake (**Underfitting**). |
| **Binarizer** | Continuous numbers ko ek مقررہ threshold ke upar `1` aur neeche `0` mein badalne wala transformer. |
| **Binary Classification** | Aisi classification jismein target $y$ ki sirf 2 classes hon (`0` ya `1`, maslan `Spam` vs `Not Spam`). |
| **Boosting** | Models ko ek ke baad ek (sequentially) is tarah train karna ke har naya model pichle models ke errors (residuals) ko kam kare (`GradientBoosting`, `HistGradientBoosting`). |
| **Calibration (`CalibratedClassifierCV`)** | Model ki `.predict_proba()` probabilities ko sachi real-world frequencies ke mutabiq durust karna. |
| **Categorical Feature** | Aisa column jismein ginti ke bajaye groups ya text categories hon (jaise `City`: Karachi, Lahore; ya `Grade`: A, B, C). |
| **Centroid (`cluster_centers_`)** | `KMeans` mein kisi cluster ke tamam points ka darmiyani markaz (mean point). |
| **Class Imbalance** | Jab kisi classification dataset mein ek class ke samples bohot zyada (e.g., 98%) aur doosri class ke bohot kam (e.g., 2%) hon. |
| **Classification** | Supervised learning jismein target $y$ ek category ya class label ho. |
| **ClassifierChain** | Multi-label classification mein binary classifiers ko ek zanjeer mein jorna taake pichle labels ki predictions agle label ki madad kar saken. |
| **Clustering** | Unsupervised learning jismein bina kisi prior label $y$ ke milte-julte samples ke groups banaye jate hain. |
| **Coefficient (`coef_`)** | Linear model ($\hat{y} = w_1 x_1 + \dots + b$) mein har feature ka weight ($w_j$) jo batata hai ke us feature ke badalne se prediction kitni badlegi. |
| **Collinearity (Multicollinearity)** | Jab do ya zyada input features aapas mein bohot zyada correlated hon (ek hi baat dohra rahe hon), jis se OLS weights unstable ho jate hain. |
| **ColumnTransformer** | Scikit-learn ka tool jo numerical columns par alag pipeline aur categorical columns par alag pipeline ek sath chala kar unhein jor deta hai. |
| **Concept Drift** | Waqt ke sath production mein features ($X$) aur target ($y$) ke darmiyan asli talluq ka badal jana. |
| **Confusion Matrix** | Classification results ka $K \times K$ table jo True Positives ($TP$), False Positives ($FP$), True Negatives ($TN$), aur False Negatives ($FN$) dikhata hai. |
| **Contamination** | Anomaly detection (`IsolationForest`, `LOF`) mein data ke andar outliers ka mutawaqqa tanasub (maslan `0.05` = 5%). |
| **Convergence (`ConvergenceWarning`)** | Optimization algorithm (`lbfgs`, `adam`, `sgd`) ka loss minimize karte hue mustahkam (stable) solution tak pohanchna. |
| **Cross-Validation (CV)** | Data ko $K$ hisson (folds) mein baant kar bar-bar train aur validate karna taake model ki generalization ka qabil-e-bharosa andaza ho sake. |
| **Curse of Dimensionality** | Jab columns (`n_features`) bohot barh jayein to data points ek doosre se bohot door (sparse) ho jate hain aur distance metrics be-asar hone lagte hain. |
| **Data Drift (Covariate Shift)** | Production mein aane wale naye input features ($X$) ke distribution ka training data se badal jana. |
| **Data Leakage** | Test set ya future ki maloomat ka ghalti se training ya preprocessing ke dauran model ko mil jana. |
| **DBSCAN** | Density-based clustering algorithm jo bheer (dense) wale ilaqon ko cluster banata hai aur akele door points ko `-1` (Noise) qarar deta hai. |
| **Decision Boundary** | Feature space mein woh lakeer ya satah (surface) jahan model apna faisla ek class se doosri class mein badalta hai. |
| **Decision Function (`.decision_function`)** | Sample ka decision boundary se signed faasla batane wala method. |
| **Decision Tree** | `if/else` threshold splits ke zariye darakht ki shakal mein faisla karne wala algorithm. |
| **Dimensionality Reduction** | High-dimensional features ko kam columns mein compress karna (`PCA`, `TruncatedSVD`, `Isomap`, `TSNE`). |
| **Dummy Encoding / One-Hot Encoding** | Categorical column ki har unique value ke liye ek alag binary (`0`/`1`) column banana (`OneHotEncoder`). |
| **Early Stopping (`early_stopping=True`)** | Validation score behtar hona band hone par `MLP` ya `HistGradientBoosting` ki training waqt se pehle rok dena taake overfitting na ho. |
| **ElasticNet** | Linear regression jo L1 (`Lasso`) aur L2 (`Ridge`) dono penalties ko mila kar lagata hai. |
| **Elbow Method** | `KMeans` mein $K$ barhane ke sath `inertia_` ka graph banana aur jahan kohni (bend) bane woh $K$ chunna. |
| **Ensemble Learning** | Ek akele model ke bajaye kai models ki team bana kar faisla lena (`RandomForest`, `GradientBoosting`, `Voting`, `Stacking`). |
| **Epoch / Iteration** | Optimization solver ka poore training dataset se ek mukammal chakar guzarna. |
| **Estimator** | Scikit-learn ka koi bhi object jo `.fit()` ke zariye data se seekhta hai. |
| **Explained Variance (`explained_variance_ratio_`)** | `PCA` mein har Principal Component original data ki kitne feesad maloomat (variance) sambhalta hai. |
| **Extrapolation** | Training data ki دیکھی hui range se bahar (maslan bohot bari values ya future trend) predict karna; tree models extrapolate nahi kar sakte, linear models kar sakte hain. |
| **F1-Score (`f1_score`)** | Precision aur Recall ka harmonic mean ($2 \frac{P \cdot R}{P + R}$). |
| **False Negative ($FN$, Type II Error)** | Asal mein Positive (`1`, e.g., Bemar/Fraud) tha magar model ne ghalti se Negative (`0`) keh diya. |
| **False Positive ($FP$, Type I Error)** | Asal mein Negative (`0`, e.g., Be-gunah) tha magar model ne ghalti se Positive (`1`) keh diya (False Alarm). |
| **Feature ($X$)** | Input column jiski madad se prediction ki jati hai (independent variable). |
| **Feature Engineering** | Domain knowledge ya math (`PolynomialFeatures`, ratios, lags) se naye mufeed columns banana. |
| **Feature Importance** | Har feature ne model ki prediction mein kitna hissa dala (`permutation_importance`, `feature_importances_`). |
| **Feature Selection** | Faltu ya noisy columns hata kar sirf sab se ahem columns rakhna (`SelectKBest`, `RFECV`, `SelectFromModel`). |
| **Fit (`.fit(X, y)`)** | Training data se model ke internal weights ya transformer ke statistics seekhne ka amal. |
| **Generalization** | Trained model ka bilkul naye (unseen) data par sahi prediction dene ki salahiyat. |
| **Gini Impurity** | Decision Tree mein kisi node ke andar classes ki milawat (impurity) naapne ka paimana ($1 - \sum p_i^2$). |
| **Gradient Boosting** | Sequential tree ensemble jismein har naya tree loss function ke negative gradient (residuals) par fit hota hai. |
| **Gradient Descent (`SGD`)** | Pahari se neeche utarne ki tarah step-by-step weights badal kar loss function ko minimum par lane ka tareeqa. |
| **GridSearchCV** | Diye gaye parameter grid ke har mumkina combination ko cross-validation par azma kar best model chunne wala tool. |
| **GroupKFold** | Cross-validation splitter jo ek hi group (patient/user) ke tamam records ko ek hi fold mein rakhta hai. |
| **HalvingGridSearchCV / HalvingRandomSearchCV** | Successive halving tournament ke zariye kam waqt mein hyperparameters tune karne wale experimental tools. |
| **HashingVectorizer** | Bina vocabulary save kiye hash function se text ko fixed columns mein badalne wala fast, stateless vectorizer. |
| **Heteroscedasticity** | Regression mein jab target ki value barhne ke sath prediction errors (residuals) ka phailao bhi barhne lage. |
| **HistGradientBoosting** | Histogram-based fast gradient boosting jo natively `NaN` aur categorical columns support karta hai. |
| **Hyperparameter** | Woh setting jo user model train karne se pehle khud deta hai (maslan `n_neighbors=5`, `C=1.0`, `max_depth=4`). |
| **Imputation (`SimpleImputer`, `KNNImputer`)** | Dataset mein khali jagahon (`NaN` / missing values) ko mean, median, most_frequent, ya neighbors se bharna. |
| **Inference / Prediction (`.predict(X)`)** | Trained model se naye data par jawab hasil karna. |
| **Inlier (`+1`)** | Normal data ke andar mojood sahi point (outlier ka ulat). |
| **Intercept (`intercept_`)** | Linear model mein constant base value ($b$) jab tamam features zero hon. |
| **Interpretability** | Model ke faislon ki wajah insano ko asani se samjha sakne ki salahiyat. |
| **IQR (Interquartile Range)** | 75th percentile aur 25th percentile ka farq ($Q_3 - Q_1$); `RobustScaler` aur outlier clipping mein use hota hai. |
| **IsolationForest** | Random trees ke zariye chote path length wale anomalies (`-1`) ko alag karne wala fast ensemble. |
| **Joblib (`joblib.dump`, `joblib.load`)** | NumPy arrays wale Scikit-learn pipelines ko disk par save aur load karne ki recommended library. |
| **Kernel Trick** | Data ko high-dimensional space mein explicitly le jaye baghair non-linear similarities calculate karna (`SVC`, `SVR`, `KernelPCA`). |
| **K-Fold Cross-Validation** | Data ko $K$ barabar folds mein baant kar $K$ martaba train aur validate karna. |
| **K-Means (`KMeans`)** | $K$ centroids ki madad se data ko $K$ gol (convex) clusters mein baantne wala unsupervised algorithm. |
| **K-Nearest Neighbors (`KNN`)** | Naye point ke sab se qareebi $k$ parosiyon ko dekh kar vote/average lene wala instance-based algorithm. |
| **Label / Target ($y$)** | Woh output column jise model predict karna seekhta hai. |
| **LabelPropagation / LabelSpreading** | Graph-based semi-supervised algorithms jo labeled points se unlabeled (`-1`) points tak labels phailate hain. |
| **Lasso (`Lasso`)** | L1-regularized linear regression jo automatic feature selection (sparse zero weights) karta hai. |
| **Learning Rate (`learning_rate`)** | Gradient boosting ya neural networks mein har step par weights ya trees ka kitna hissa shamil karna hai. |
| **Log Loss (`log_loss`)** | Cross-entropy loss jo predicted probabilities ki durusti naapta hai. |
| **Logistic Regression** | Sigmoid function ke zariye probabilities aur class labels predict karne wala linear classifier. |
| **MAE (`mean_absolute_error`)** | Asli aur predicted values ke farq (absolute error) ka seedha ausat ($\frac{1}{N}\sum |y - \hat{y}|$). |
| **Manifold Learning** | High-dimensional space mein mure hue non-linear surfaces ko low dimensions mein kholna (`Isomap`, `TSNE`, `LLE`). |
| **Margin** | SVM mein decision boundary aur sab se qareebi points (Support Vectors) ke darmiyan faasla. |
| **Matthews Correlation Coefficient (`matthews_corrcoef`)** | Binary classification confusion matrix ke charon khano ($TP, TN, FP, FN$) kochamil karne wala balanced correlation score ($-1$ to $+1$). |
| **Multiclass Classification** | 3 ya zyada mutually exclusive classes wali classification (`Setosa`, `Versicolor`, `Virginica`). |
| **Multilabel Classification** | Jab ek hi sample ke ek waqt mein multiple sahi tags/labels ho saken (`[Sports, Economy]`). |
| **Multi-Output (`MultiOutputRegressor`)** | Ek hi model se 2 ya zyada targets ek sath predict karwana. |
| **Naive Bayes (`GaussianNB`, `MultinomialNB`)** | Bayes' Theorem par مبنی fast classifiers jo features ko conditionally independent farz karte hain. |
| **Nested Cross-Validation** | Outer CV loop ke andar Inner `GridSearchCV` chalana taake hyperparameter tuning ki unbiased generalization evaluate ho sake. |
| **N-Grams (`ngram_range`)** | Text mein lagatar $N$ alfaz ya characters ke joray (maslan bigram `"acha nahi"`). |
| **Normalizer** | Columns ke bajaye **har Row (sample)** ko unit norm ($\|x\|=1$) par divide karne wala transformer. |
| **Novelty Detection** | Clean (100% normal) data par model fit kar ke future mein aane wale naye/ajeeb samples (`-1`) ko pakarna. |
| **Out-of-Core Learning (`.partial_fit`)** | RAM se bare datasets ko disk se chote chunks mein parh kar incrementally train karna. |
| **Outlier** | Baqi data se bohot door ya ghair-mamooli point. |
| **Overfitting (High Variance)** | Jab model training data aur uske shor (noise) ko ratta laga le—training score bohot high magar unseen test score kharab! |
| **Partial Dependence Plot (`PartialDependenceDisplay`)** | Kisi ek ya do features ke badalne se model ki average prediction par parne wala asar dikhane wala graph. |
| **PCA (`Principal Component Analysis`)** | Maximum variance wali orthogonal directions dhoond kar features compress karne wala linear method. |
| **Permutation Importance** | Test set par ek feature ko shuffle kar ke model ke score mein kami naapne wala unbiased interpretability method. |
| **Pipeline (`Pipeline`)** | Transformers aur final predictor ko ek single object mein jorne wala container jo data leakage rokta hai. |
| **PolynomialFeatures** | Features ke squares, cubes, aur interactions ($x_1^2, x_1 x_2$) paida karne wala transformer. |
| **Precision (`precision_score`)** | Model ne jitne samples ko Positive kaha, un mein se kitne waqai Positive nikle ($\frac{TP}{TP+FP}$). |
| **Pseudo-Labeling (`SelfTrainingClassifier`)** | Semi-supervised learning mein high-confidence unlabeled samples ko model ke apne predicted labels de kar training mein shamil karna. |
| **QuantileRegressor** | Mean ke bajaye Median (`quantile=0.5`) ya kisi bhi percentile (e.g. 90th percentile interval) ko predict karne wala linear regressor. |
| **R² Score (`r2_score`)** | Coefficient of Determination—regression model ne target ki kitne feesad variance explain ki ($1.0$ best). |
| **RandomForest** | Sainkron independent Decision Trees (trained on bootstrap samples + random feature subsets) ka ensemble. |
| **RandomizedSearchCV** | Parameter distributions mein se fixed `n_iter` random combinations azma kar tez hyperparameter tuning karne wala tool. |
| **RANSACRegressor** | Iterative taur par outliers ko alag kar ke sirf inliers par linear model fit karne wala robust regressor. |
| **Recall / Sensitivity (`recall_score`)** | Asal Positive samples mein se model ne kitne feesad sahi pakray ($\frac{TP}{TP+FN}$). |
| **Regression** | Supervised learning jismein target $y$ ek continuous number (qeemat, wazan, darja hararat) ho. |
| **Regularization** | Overfitting rokne ke liye model ke bare weights ya complexity par loss function mein jurmana (penalty) lagana. |
| **Residual** | Regression mein asli value aur predicted value ka farq ($y_i - \hat{y}_i$). |
| **RFECV (`Recursive Feature Elimination with CV`)** | Kamzor features ko bari-bari hata kar cross-validation se behtareen feature subset chunne wala tool. |
| **Ridge (`Ridge`)** | L2-regularized linear regression jo weights ko chota kar ke multicollinearity aur overfitting sambhalta hai. |
| **RMSE (`root_mean_squared_error`)** | Squared errors ke ausat ka jazar ($\sqrt{\frac{1}{N}\sum(y-\hat{y})^2}$)—target ke asli units mein sab se mashhoor regression metric. |
| **RobustScaler** | Median aur Interquartile Range (IQR) se scale karne wala transformer jo outliers se kharab nahi hota. |
| **Silhouette Score (`silhouette_score`)** | Clustering ki quality naapne ka internal metric ($-1$ se $+1$; $+1$ ke qareeb matlab clusters ghane aur alag-alag hain). |
| **Sparse Matrix (CSR)** | Aisi matrix jismein 95%+ values zero hon; memory bachane ke liye sirf non-zero values aur unke indices store kiye jate hain. |
| **Specificity (True Negative Rate)** | Asal Negative samples mein se kitne feesad ko model ne sahi Negative pehchana ($\frac{TN}{TN+FP}$). |
| **SplineTransformer** | Smooth piecewise polynomial B-spline curves banane wala feature engineering transformer. |
| **Stacking (`StackingClassifier`, `StackingRegressor`)** | Mukhtalif models ki cross-validated predictions ko ek final meta-model (`final_estimator`) ke zariye combine karna. |
| **StandardScaler** | Har feature ka mean $0$ aur standard deviation $1$ banane wala z-score scaler ($z = \frac{x-\mu}{\sigma}$). |
| **Stratified Splitting (`stratify=y`, `StratifiedKFold`)** | Train aur test folds mein har class ka percentage tanasub bilkul barabar rakhna. |
| **Support Vector Machine (`SVC`, `SVR`)** | Maximum margin hyperplane aur kernel trick use karne wala supervised algorithm. |
| **TF-IDF (`TfidfVectorizer`)** | Lafz ki document mein ahmiyat (TF) aur poore corpus mein nayabi (IDF) ko zarb de kar text features banana. |
| **TimeSeriesSplit** | Waqt ki tarteeb barqarar rakhte hue hamesha mazi (past) par train aur mustaqbil (future) par validate karne wala CV splitter. |
| **Transformer** | Scikit-learn estimator jo `.transform(X)` ke zariye features ko saaf, scale, ya encode karta hai. |
| **TransformedTargetRegressor** | Regression target $y$ par training ke waqt function (e.g. `np.log1p`) aur prediction ke waqt inverse (`np.expm1`) automatically lagane wala wrapper. |
| **TunedThresholdClassifierCV** | Binary classifier ke decision threshold (`0.50`) ko cross-validation ke zariye kisi metric (e.g. `f1`) ke liye optimize karne wala estimator. |
| **Underfitting (High Bias)** | Jab model itna sada ho ke training data ka pattern bhi na seekh sake (train aur test dono par kharab score). |
| **Variance (Model Variance)** | Training data ke badalne se model ki predictions mein aane wali tabdeeli; high variance ka matlab **Overfitting** hai. |
| **VarianceThreshold** | Zero ya bohot kam variance wale (constant/quasi-constant) columns ko hatane wala feature selector. |
| **Voting (`VotingClassifier`, `VotingRegressor`)** | Multiple models ki predictions ka majority vote (`'hard'`) ya probability average (`'soft'`) lena. |
