import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model = pickle.load(open("Models/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("Models/tfidf_vectorizer.pkl", "rb"))

# strong negative words override
negative_words = {"worst", "shit","bad", "terrible", "horrible", "awful", "sucks", "disgusting", "pathetic", "garbage","no"}

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Prediction Function
def predict_sentiment(review):
    review_cleaned = clean_text(review)
    
    if any(word in review_cleaned.split() for word in negative_words):
        print("\nPredicted Sentiment: Negative (Rule-based override)")
        return
    
    review_vectorized = vectorizer.transform([review_cleaned])
    prediction_proba = model.predict_proba(review_vectorized)[0]
    
    prediction = "Positive" if prediction_proba[1] > 0.6 else "Negative"
    
    print(f"\nRaw Prediction Probabilities: {prediction_proba}")
    print(f"Predicted Sentiment: {prediction}")

# Run Predictions
while True:
    user_input = input("\nEnter a review (or type 'exit' to stop): ").strip()
    if user_input.lower() == "exit":
        break
    predict_sentiment(user_input)





