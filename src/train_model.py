import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 🔹 Load Data
df = pd.read_csv("Data-folder/cleaned_reviews.csv")

# 🔹 Remove duplicates & missing values
df = df.drop_duplicates(subset=['cleaned_text']).dropna(subset=['Rating', 'cleaned_text'])

# 🔹 Convert Ratings to binary labels
df['label'] = df['Rating'].apply(lambda x: 1 if x >= 3 else 0)

# 🔹 Balance Dataset (Prevents bias)
df_majority = df[df.label == 1]
df_minority = df[df.label == 0]

df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])

# 🔹 Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df_balanced['cleaned_text'] = df_balanced['cleaned_text'].apply(clean_text)

# 🔹 TF-IDF Vectorizer with n-grams
vectorizer = TfidfVectorizer(ngram_range=(1,2))  
X = vectorizer.fit_transform(df_balanced['cleaned_text'])
y = df_balanced['label']

# 🔹 Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train XGBoost Model with Optimized Parameters
model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 🔹 Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

# 🔹 Save Model & Vectorizer
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))






