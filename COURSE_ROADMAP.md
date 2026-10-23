# Course Roadmap & Study Guide — Zero to Professional

## 1. Learning Philosophy (Seekhne Ka Usool)
Machine Learning ratta lagane (memorization) ka naam nahi hai. Ek kamyab Machine Learning Engineer woh hota hai jo samajhta hai ke:
1. Problem kis qism ka hai (Classification, Regression, Clustering, ya Anomaly Detection)?
2. Data mein kya masail hain (Missing values, outliers, categorical text, ya class imbalance)?
3. Kaunsa algorithm kab chuna jaye aur usay bina **Data Leakage** ke kaise evaluate kiya jaye?

Is course ko parhte waqt har code block ko khud apne Jupyter Notebook ya Python script mein type kar ke run karein. Copy-paste karne se hath ki practice (muscle memory) nahi banti.

## 2. Prerequisites (Pehle Se Kya Aana Chahiye?)
- **Computer Basics:** File folders banana, browser use karna, aur basic terminal/command prompt kholna.
- **Mathematics:** Sirf matric/high-school level ka basic math (جمع، تفریق، ضرب، تقسیم، اوسط / average, aur graph par X-Y coordinates samajhna). Baqi tamam statistical aur mathematical intuition hum course ke andar Roman Urdu mein sikhayenge.
- **Prior Coding Experience:** Koi zaroorat nahi! Hum [PYTHON_PREREQUISITES.md](PYTHON_PREREQUISITES.md) mein bilkul zero se Python sikhayenge.

## 3. Estimated Study Time & 12-Week Structured Plan
Agar aap rozana **1.5 se 2 ghante (hours)** dete hain (hafte mein 10–12 ghante), to aap **12 Weeks (3 Months)** mein zero se professional level tak pohanch sakte hain:

| Week | Focus Area | Modules to Complete | Weekly Milestone / Goal |
| :--- | :--- | :--- | :--- |
| **Week 1** | AI/ML Foundations & Setup | [INSTALLATION.md](INSTALLATION.md), [01_AI_AND_MACHINE_LEARNING.md](01_AI_AND_MACHINE_LEARNING.md) | Understand ML types, setup Python & Jupyter environment |
| **Week 2** | Python for Data Science | [PYTHON_PREREQUISITES.md](PYTHON_PREREQUISITES.md) | Write functions, loops, dicts, and read CSV files independently |
| **Week 3** | Data Manipulation & EDA | [NUMPY_PANDAS_FOUNDATIONS.md](NUMPY_PANDAS_FOUNDATIONS.md) | Clean a raw DataFrame and plot distributions with Seaborn |
| **Week 4** | Scikit-learn Core & Preprocessing | [02_INTRODUCTION_TO_SCIKIT_LEARN.md](02_INTRODUCTION_TO_SCIKIT_LEARN.md), [03_DATA_PREPROCESSING.md](03_DATA_PREPROCESSING.md), [04_FIRST_MACHINE_LEARNING_MODEL.md](04_FIRST_MACHINE_LEARNING_MODEL.md) | Build first Iris model and master `ColumnTransformer` |
| **Week 5** | Supervised Classification | [05_CLASSIFICATION.md](05_CLASSIFICATION.md) | Train and compare Logistic Regression, Trees, Random Forests, SVM |
| **Week 6** | Supervised Regression | [06_REGRESSION.md](06_REGRESSION.md) | Build house price predictors with Linear, Ridge, Lasso, Boosting |
| **Week 7** | Unsupervised ML: Clustering & Dim Reduction | [07_CLUSTERING.md](07_CLUSTERING.md), [08_DIMENSIONALITY_REDUCTION.md](08_DIMENSIONALITY_REDUCTION.md) | Customer segmentation with KMeans/DBSCAN and PCA visualization |
| **Week 8** | Features, Metrics & Cross-Validation | [09_FEATURE_SELECTION_ENGINEERING.md](09_FEATURE_SELECTION_ENGINEERING.md), [10_MODEL_EVALUATION.md](10_MODEL_EVALUATION.md), [11_CROSS_VALIDATION_TUNING.md](11_CROSS_VALIDATION_TUNING.md) | Master ROC-AUC, Precision-Recall, and `GridSearchCV` |
| **Week 9** | Production Pipelines & Imbalanced Data | [12_PIPELINES.md](12_PIPELINES.md), [13_IMBALANCED_LEARNING.md](13_IMBALANCED_LEARNING.md) | Build zero-leakage pipelines and tune probability thresholds |
| **Week 10** | Specialized ML (Anomaly, Semi-Supervised, NN, NLP) | [14_ANOMALY_DETECTION.md](14_ANOMALY_DETECTION.md), [15_SEMI_SUPERVISED_LEARNING.md](15_SEMI_SUPERVISED_LEARNING.md), [16_NEURAL_NETWORKS.md](16_NEURAL_NETWORKS.md), [17_TEXT_ML_NLP.md](17_TEXT_ML_NLP.md) | Fraud detection and TF-IDF spam/sentiment classification |
| **Week 11** | Time Series, Interpretability, Deployment & Scale | [18_TIME_SERIES_SPECIALIZED_TASKS.md](18_TIME_SERIES_SPECIALIZED_TASKS.md), [19_MODEL_INTERPRETABILITY.md](19_MODEL_INTERPRETABILITY.md), [20_MODEL_PERSISTENCE_DEPLOYMENT.md](20_MODEL_PERSISTENCE_DEPLOYMENT.md), [21_PERFORMANCE_SCALABILITY.md](21_PERFORMANCE_SCALABILITY.md) | Explain models with Permutation Importance & wrap in FastAPI |
| **Week 12** | API Mastery, Projects & Interview Prep | [22_SCIKIT_LEARN_API_REFERENCE.md](22_SCIKIT_LEARN_API_REFERENCE.md), [23_COMMON_ERRORS.md](23_COMMON_ERRORS.md), [24_PRACTICAL_PROJECTS.md](24_PRACTICAL_PROJECTS.md), [25_QUIZZES_ASSIGNMENTS.md](25_QUIZZES_ASSIGNMENTS.md), [26_INTERVIEW_PREPARATION.md](26_INTERVIEW_PREPARATION.md) | Complete 15 projects, pass capstone assessment & mock interview |

## 4. How Students Should Practice (Practice Kaise Karein?)
1. **Read & Understand (Parhein aur Samjhein):** Har topic ki Roman Urdu explanation aur real-world analogy dhyan se parhein.
2. **Run & Modify Code (Code Chala Kar Tabdeel Karein):** Diye gaye code ko run karein, phir uske parameters badal kar dekhein (maslan `n_neighbors=3` ki jagah `15` kar ke dekhein ke accuracy par kya farq parta hai).
3. **Break It On Purpose (Jaan Boojh Kar Ghalti Karein):** Har chapter ke "Common Errors" aur "Common Mistakes" section ko zaroor test karein taake jab asli project mein `ValueError` aaye to aap ghabrayein nahi.
4. **Solve Practice Exercises First:** Har chapter ke aakhir mein Beginner, Intermediate, aur Challenge exercises di gayi hain. Pehle khud code likhne ki koshish karein, aur phir Solution section se match karein.

---
**Next Step:** Chalein ab [INSTALLATION.md](INSTALLATION.md) par chalte hain aur apna coding environment set up karte hain!
