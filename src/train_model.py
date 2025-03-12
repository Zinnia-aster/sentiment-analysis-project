import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

df = pd.read_csv("Data-folder/cleaned_reviews.csv")
df = df.drop_duplicates(subset=['cleaned_text']).dropna(subset=['Rating', 'cleaned_text'])
df['label'] = df['Rating'].apply(lambda x: 1 if x >= 3 else 0)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['cleaned_text'] = df['cleaned_text'].apply(clean_text)

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=300,  
    learning_rate=0.08,  
    max_depth=6,  
    colsample_bytree=0.8,  
    scale_pos_weight=1,  
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Improved Model Accuracy: {accuracy * 100:.2f}%")

pickle.dump(model, open("Models/sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("Models/tfidf_vectorizer.pkl", "wb"))








