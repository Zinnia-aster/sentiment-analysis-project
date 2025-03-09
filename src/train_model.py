import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Data
df = pd.read_csv("Data-folder/cleaned_reviews.csv")

# Remove duplicates
df = df.drop_duplicates(subset=['cleaned_text'])
df = df.dropna(subset=['Rating', 'cleaned_text'])

# Convert Ratings to binary labels
df['label'] = df['Rating'].apply(lambda x: 1 if x >= 3 else 0)  

# modelling
X = df['cleaned_text']
y = df['label']

# TF-IDF Vectorizer with better preprocessing
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words='english')
X_transformed = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Logistic Regression with Hyperparameter Tuning
param_grid = {'C': [0.1, 1, 10, 50]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

model = grid.best_estimator_

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save Model & Vectorizer
with open("Models/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("Models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print(f"Model trained with accuracy: {accuracy:.2f}")
