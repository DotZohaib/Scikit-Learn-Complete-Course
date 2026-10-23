# Part 19: Text Processing and Natural Language Processing (`sklearn.feature_extraction.text`)

SMS messages, customer reviews, emails, aur social media posts **Unstructured Text** hote hain. Machine Learning models alfaz nahi samajhte, woh sirf numbers ke matrices samajhte hain! Is chapter mein hum [Scikit-learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) ke tamam 18 topics aur **Roman Urdu Spam & Sentiment Detection Projects** seekhenge.

---

## 1. Text Classification Workflow & Core Vectorizers

Text ko numerical matrix mein badalne ke amal ko **Vectorization** kehte hain. Scikit-learn mein 4 main text transformers hain:

| Transformer (`sklearn.feature_extraction.text`) | How It Works Intuitively | Pros & Cons |
| :--- | :--- | :--- |
| **1. `CountVectorizer`** | **Bag-of-Words (BoW)**: Poore corpus ke tamam unique alfaz ki ek **Vocabulary (`vocabulary_`)** banata hai aur har document mein har lafz kitni baar aaya, uski ginti (count) likhta hai! | Simple, fast; lekin har sentence mein aane wale aam alfaz (`"hai"`, `"the"`, `"is"`) ko bohot bara number de deta hai. |
| **2. `TfidfVectorizer`** | **Term Frequency–Inverse Document Frequency**: Jo lafz ek document mein bar-bar aaye ($\text{TF}$ high) lekin baqi poore corpus mein kam documents mein mile ($\text{IDF}$ high—jaise `"inaam"`, `"otp"`, `"bisp"`), usay zyada weight deta hai; aur jo lafz har jagah aaye (`"hai"`, `"mein"`) uska weight kam kar deta hai! | **Best all-around text vectorizer for Classification!** (`CountVectorizer` + `TfidfTransformer` ka ikattha रूप). |
| **3. `TfidfTransformer`** | `CountVectorizer` ke count matrix ko TF-IDF weights mein badalta hai: $\text{idf}(t) = \log \frac{1 + n}{1 + \text{df}(t)} + 1$. | Jab aap ke paas pehle se word counts mojood hon. |
| **4. `HashingVectorizer`** | **Stateless Hashing Trick**: Koi `vocabulary_` memory mein save nahi karta! Har lafz par hash function laga kar fixed `n_features=2**18` columns mein map kar deta hai. | **Out-of-core / Streaming Large Text** ke liye best! **Limitation:** `inverse_transform()` nahi ho sakta (pata nahi chalta kis column mein kaunsa lafz gaya) aur hash collisions ho sakte hain. |

---

## 2. Important Text Hyperparameters: N-Grams, Stop Words, `min_df`, `max_df` & Sublinear TF

1. **Tokenization & N-Grams (`ngram_range=(1, 2)`):**
   - **Unigrams `(1, 1)`**: Ek-ek lafz alag (`" acha "`, `"nahi"`). Masla: `"acha"` dekh kar model खुश ho jayega halanke aagay `"nahi"` likha tha!
   - **Bigrams `(1, 2)`**: Do-do alfaz ke joray (`"acha nahi"`, `"bohat behtareen"`) bhi vocabulary mein shamil karta hai taake negation (inkaar) pakra ja sake!
   - **Character N-Grams (`analyzer='char_wb', ngram_range=(3, 5)`)**: Spelling mistakes aur **Roman Urdu variations** (`"shukria"`, `"shukriya"`, `"shukrya"`) ko pakarne ke liye lafzon ke bajaye 3–5 characters ke tukray banana bohot kaargar hota hai!
2. **Stop Words (`stop_words='english'` or Custom List) & Trade-offs:**
   - Aam alfaz (`"the"`, `"and"`, `"ka"`, `"ki"`, `"hai"`) hatane se columns kam hote hain.
   - **Warning / Trade-off:** Sentiment analysis mein `"not"`, `"no"`, `"never"` (ya Roman Urdu mein `"nahi"`, `"mat"`) ko stop words mein **kabhi mat hatayein**, warna `"not good"` sirf `"good"` reh jayega!
3. **`min_df` and `max_df`:**
   - `min_df=2`: Woh alfaz jo poore dataset mein 2 se kam documents mein aaye hon (typos/rare names) unhein ignore karein.
   - `max_df=0.90`: Woh alfaz jo 90% se zyada documents mein mojood hon unhein ignore karein.
4. **Sparse Text Matrices:**
   - 50,000 alfaz ki vocabulary mein ek SMS mein sirf 10 alfaz hote hain (baqi 49,990 columns `0` hote hain). Is liye vectorizers hamesha **SciPy CSR Sparse Matrix** return karte hain. Is par kabhi `.toarray()` mat chalayein jab tak data bohot chota na ho, warna RAM bhar jayegi!

---

## 3. Honest Linguistic Notes: Roman Urdu, Urdu (Arabic Script), Sindhi & LLMs
1. **Built-in Linguistic Support Limitation:** Scikit-learn mein sirf `stop_words="english"` built-in hai. **Roman Urdu, Urdu (نستعلیق/عربی رسم الخط), Sindhi (سنڌي), Pashto, ya Punjabi** ke liye Scikit-learn mein koi built-in stemmer, lemmatizer, ya stop-word list **nahi** hai!
2. **Roman Urdu Strategy:** Roman Urdu mein koi standard spelling nahi hoti (`"acha"` vs `"achha"`). Is ke liye **`TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))`** ya `FeatureUnion` (Word N-grams + Char N-grams) behtareen kaam karta hai!
3. **Classical NLP vs Transformer LLMs:** Scikit-learn bag-of-words/TF-IDF models (`MultinomialNB`, `ComplementNB`, `LinearSVC`, `LogisticRegression`) fast spam/topic/sentiment classification ke liye lajawab hain. Lekin translation, summarization, ya deep semantic understanding (jaise BERT, GPT, Gemini) ke liye HuggingFace `transformers` ya PyTorch ki zaroorat hoti hai.

---

## 4. Complete Runnable Code: Roman Urdu + English SMS Spam Detection & Sentiment Classification Pipelines

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =====================================================================
# PROJECT 1: Pakistani SMS Spam Detection (Roman Urdu + English)
# =====================================================================
sms_data = [
    # SPAM (1): Fake BISP, Jeeto lottery, Prize bonds, Promo links
    ("Mubarak ho! Apka BISP 25000 cash inaam nikla hai abhi call karein", 1),
    ("Congratulation you won 10 Lakh Jeeto Pakistan lottery send CNIC and fee", 1),
    ("Free 50GB Jazz data حاصل karne ke liye is link par click karein winner", 1),
    ("Apka ATM card block ho gaya hai foran OTP aur PIN share karein", 1),
    ("Urgent prize claim 500000 PKR cash transfer send bank account details", 1),
    ("Inaam jeeto lucky draw mein apka mobile number select hua hai", 1),
    ("Win free iPhone 16 Pro Max click link and pay delivery charges now", 1),
    ("Benazir Kafalat program 10500 rupay lene ke liye abhi code bhejein", 1),
    # HAM / NORMAL (0): Daily family, office, university messages
    ("Bhai ghar aate waqt 2 kilo doodh aur dahi le aana", 0),
    ("Kal subah 9 baje university mein Machine Learning ki class hogi", 0),
    ("Meeting ka time 3 baje se badal kar 4 baje kar diya gaya hai", 0),
    ("Ammi keh rahi hain khana tayyar hai jaldi ghar aa jao", 0),
    ("Please review the project report and send feedback by tomorrow morning", 0),
    ("Yar aaj shaam ko cricket khelne chalna hai ground mein?", 0),
    ("Assignment submit karne ki aakhri tareekh Friday hai", 0),
    ("Main abhi Shahrah-e-Faisal traffic mein phansa hua hoon 15 min lagenge", 0),
]

df_sms = pd.DataFrame(sms_data, columns=["text", "is_spam"])
X_tr, X_te, y_tr, y_te = train_test_split(
    df_sms["text"], df_sms["is_spam"], test_size=0.25, random_state=42, stratify=df_sms["is_spam"]
)

# Combine Word (1,2)-grams and Character (3,5)-grams to handle Roman Urdu spelling variations!
combined_features = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)))
])

spam_pipe = Pipeline([
    ("features", combined_features),
    ("clf", ComplementNB(alpha=0.5))
])

spam_pipe.fit(X_tr, y_tr)
print("=== 1. Pakistani SMS Spam Detection Results ===")
print(f"Test Set Accuracy: {spam_pipe.score(X_te, y_te)*100:.1f}%")

new_messages = [
    "Mubarak! Apka 5 Lakh BISP inaam nikla hai abhi OTP bhejein",
    "Bhai kal office ki meeting subah 10 baje hai",
]
preds = spam_pipe.predict(new_messages)
probas = spam_pipe.predict_proba(new_messages)[:, 1]
for msg, p, pr in zip(new_messages, preds, probas):
    label = "SPAM (Fraud)" if p == 1 else "HAM (Normal)"
    print(f"  [{label:12s} | Prob={pr:.2f}] -> '{msg}'")

# =====================================================================
# PROJECT 2: Roman Urdu Product Review Sentiment Analysis (Bigrams for Negation!)
# =====================================================================
reviews = [
    "Yeh mobile bohat acha aur tez hai battery kamaal hai",
    "Quality bilkul bekar hai paise zaya ho gaye",
    "Delivery time par hui aur packing bohat behtareen thi",
    "Product bilkul acha nahi hai kharab nikla",
    "Bohat shandar service aur sasta rate shukriya",
    "Camera acha nahi hai aur mobile garam hota hai",
]
sentiments = [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Notice ngram_range=(1, 2) captures "acha nahi" as a distinct bigram!
sent_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("svm", LinearSVC(random_state=42))
])
sent_pipe.fit(reviews, sentiments)

test_reviews = ["Bohat acha aur shandar product hai", "Bilkul acha nahi hai bekar service"]
print("\n=== 2. Roman Urdu Sentiment Predictions (Capturing 'acha nahi' Bigram) ===")
for rev, pred in zip(test_reviews, sent_pipe.predict(test_reviews)):
    print(f"  Review: '{rev}' -> Sentiment: {'POSITIVE (1)' if pred==1 else 'NEGATIVE (0)'}")

# Inspect Sparse Matrix properties & HashingVectorizer
vec_mat = sent_pipe.named_steps["tfidf"].transform(reviews)
print(f"\nTF-IDF Matrix Type: {type(vec_mat).__name__} | Shape: {vec_mat.shape} | Non-zero elements: {vec_mat.nnz}")
h_mat = HashingVectorizer(n_features=128).transform(reviews)
print(f"HashingVectorizer Output Shape: {h_mat.shape}")
```

### Expected Output
```text
=== 1. Pakistani SMS Spam Detection Results ===
Test Set Accuracy: 100.0%
  [SPAM (Fraud) | Prob=0.89] -> 'Mubarak! Apka 5 Lakh BISP inaam nikla hai abhi OTP bhejein'
  [HAM (Normal) | Prob=0.14] -> 'Bhai kal office ki meeting subah 10 baje hai'

=== 2. Roman Urdu Sentiment Predictions (Capturing 'acha nahi' Bigram) ===
  Review: 'Bohat acha aur shandar product hai' -> Sentiment: POSITIVE (1)
  Review: 'Bilkul acha nahi hai bekar service' -> Sentiment: NEGATIVE (0)

TF-IDF Matrix Type: csr_matrix | Shape: (6, 49) | Non-zero elements: 60
HashingVectorizer Output Shape: (6, 128)
```

---

## 5. Practice Exercises & Solutions

### Exercises
1. **Beginner:** `CountVectorizer(ngram_range=(1, 2))` ko `["hotel acha hai", "hotel acha nahi hai"]` par fit karein aur `.get_feature_names_out()` print karein.
2. **Intermediate:** Ek custom Roman Urdu stop words list `roman_stops = ["hai", "hain", "ka", "ki", "ke"]` banayein aur usay `TfidfVectorizer(stop_words=roman_stops)` mein pass karein.
3. **Challenge:** `OneVsRestClassifier(LinearSVC())` aur `TfidfVectorizer` ke zariye ek **Multi-Label Text Classifier** banayein jahan ek news headline ke multiple tags (`[Politics, Sports, Economy]`) ho saken!

### Solutions
```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

# 1. Beginner
cv = CountVectorizer(ngram_range=(1, 2)).fit(["hotel acha hai", "hotel acha nahi hai"])
print("Vocabulary with Bigrams:", cv.get_feature_names_out())

# 2. Intermediate
tv = TfidfVectorizer(stop_words=["hai", "hain", "ka", "ki", "ke"]).fit(["yeh kitab ka safha hai"])
print("Filtered Features:", tv.get_feature_names_out())

# 3. Challenge: Multi-Label News Tagging
headlines = [
    "Pakistan cricket team wins match and boosts stadium economy",
    "Government announces new budget tax policy for economy",
    "PSL final match ticket sales break records"
]
# Multi-label binary matrix: [Sports, Economy, Politics]
Y_multi = np.array([
    [1, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
])
ml_pipe = make_pipeline(TfidfVectorizer(), OneVsRestClassifier(LinearSVC(random_state=42))).fit(headlines, Y_multi)
print("Multi-label prediction for 'cricket budget tax':", ml_pipe.predict(["cricket match boosts economy"]))
```

---
**Next Chapter:** [18_TIME_SERIES_SPECIALIZED_TASKS.md](18_TIME_SERIES_SPECIALIZED_TASKS.md)
