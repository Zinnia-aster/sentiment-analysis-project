import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Function to clean text by removing special characters, converting to lowercase, and removing stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

def preprocess_data(input_file, output_file):
    """Function to preprocess review data and extract features using TF-IDF."""
    df = pd.read_csv(input_file)
    print("Columns in input file:", df.columns)  # Debugging step
    
    if 'Review' not in df.columns:
        raise ValueError("Error: 'Review' column not found in the input file.")
    
    # Clean the review text
    df['cleaned_text'] = df['Review'].apply(clean_text)
    
    # Convert text into numerical features using TF-IDF
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(df['cleaned_text'])
    
    # Save cleaned data
    df.to_csv(output_file, index=False)
    
    # Reload cleaned file to access sentiment labels (assuming 'Rating' represents sentiment)
    df_cleaned = pd.read_csv(output_file)
    print("Columns in cleaned file:", df_cleaned.columns)  # Debugging step
    
    if 'Rating' in df_cleaned.columns:
        return x, df_cleaned['Rating']  # Use 'Rating' as sentiment label
    else:
        print("Warning: 'Rating' column not found in the cleaned file.")
        return x, None

# Example usage
x, y = preprocess_data(
    r"C:\Users\priyadharshini\Desktop\github\sentiment-analysis-project\Data-folder\raw_reviews.csv",
    r"C:\Users\priyadharshini\Desktop\github\sentiment-analysis-project\Data-folder\cleaned_reviews.csv"
)

