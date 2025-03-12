import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
model = pickle.load(open("../Models/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("../Models/tfidf_vectorizer.pkl", "rb"))

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  
    text = " ".join([word for word in text.split() if word not in stop_words])  
    return text

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", page_icon="📊", layout="centered")

# Title Section
st.markdown(
    """
    <h1 style='text-align: center; color: #1E88E5;'>
        📊 Sentiment Analysis Dashboard
    </h1>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align: center;'>Analyze the sentiment of your review instantly! 📝</h4>", 
    unsafe_allow_html=True
)

st.markdown("---")  

st.subheader("🧐 Enter a review to analyze its sentiment")
user_input = st.text_area("Type your review here...", height=150)

# Button
if st.button("🔍 Analyze Sentiment", key="analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Process input
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.success("😊 **Positive Sentiment**")
        else:
            st.error("😢 **Negative Sentiment**")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>Made with ❤️ using Streamlit</p>", 
    unsafe_allow_html=True
)

