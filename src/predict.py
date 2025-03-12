import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 🔹 Load Model & Vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# 🔹 Strong Negative Words Override
negative_words = {"worst", "shit", "terrible", "horrible", "awful", "sucks", "disgusting", "pathetic", "garbage","no"}

# 🔹 Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# 🔹 Prediction Function
def predict_sentiment(review):
    review_cleaned = clean_text(review)
    
    # 🔸 Rule-based override for strong negatives
    if any(word in review_cleaned.split() for word in negative_words):
        print("\nPredicted Sentiment: Negative (Rule-based override)")
        return
    
    review_vectorized = vectorizer.transform([review_cleaned])
    prediction_proba = model.predict_proba(review_vectorized)[0]
    
    # 🔸 Adjusted threshold (fixes false positives)
    prediction = "Positive" if prediction_proba[1] > 0.6 else "Negative"
    
    print(f"\nRaw Prediction Probabilities: {prediction_proba}")
    print(f"Predicted Sentiment: {prediction}")

# 🔹 Run Predictions
while True:
    user_input = input("\nEnter a review (or type 'exit' to stop): ").strip()
    if user_input.lower() == "exit":
        break
    predict_sentiment(user_input)





