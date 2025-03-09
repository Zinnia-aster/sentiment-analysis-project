import pickle
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model
with open("Models/sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the saved vectorizer
with open("Models/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Function to clean text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text

# Predict sentiment function
def predict_sentiment(review):
    cleaned_review = preprocess_text(review)
    review_vectorized = vectorizer.transform([cleaned_review])  # Transform the input using the saved vectorizer
    prediction = model.predict(review_vectorized)  # Get prediction (0 = Negative, 1 = Positive)
    return "Positive" if prediction[0] == 1 else "Negative"

# Example usage
if __name__ == "__main__":
    review_text = "good!"  # User input
    sentiment = predict_sentiment(review_text)
    print(f"Predicted Sentiment: {sentiment}")
