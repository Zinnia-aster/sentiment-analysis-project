Here's your **README** with proper formatting using hashtags for easy copy-pasting! 🚀  

---

# **📊 Sentiment Analysis on Reviews using XGBoost**

## **📌 Project Overview**
This project focuses on sentiment analysis of user reviews using **TF-IDF vectorization** and an **XGBoost classifier** to classify sentiments as positive or negative. The model achieves an impressive **97.89% accuracy** in predicting sentiment.

## **📂 Dataset**
- The dataset consists of user reviews along with ratings.
- Reviews were preprocessed by removing duplicates and null values.
- A new label was created:  
  - **Positive sentiment** (Rating ≥ 3) → **1**  
  - **Negative sentiment** (Rating < 3) → **0**  
- To balance the dataset, **downsampling** was applied.

## **🔍 Text Preprocessing**
- **Lowercasing** the text  
- **Removing special characters & punctuation**  
- **Removing stopwords** using the NLTK library  

## **📈 Feature Extraction: TF-IDF**
- **TF-IDF (Term Frequency - Inverse Document Frequency)** was used to convert text into numerical features.  
- We used **bigram representation (n-grams: 1,2)** to capture word relationships.  

## **🛠️ Model: XGBoost Classifier**
- The model was trained using **XGBoost**, a highly efficient gradient boosting algorithm.  
- **Hyperparameters used:**
  - **n_estimators** = 200  
  - **learning_rate** = 0.1  
  - **max_depth** = 5  
- Achieved an **accuracy of 97.89%** on test data. ✅  

## **💻 Deployment with Streamlit**
- A **dashboard** was created using **Streamlit** for real-time sentiment prediction.
- Users can input a review, and the model predicts whether it's **positive or negative**.

## **🚀 How to Run**
1️⃣ Clone the repository  
2️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```  
3️⃣ Run the **Streamlit app**:  
```bash
streamlit run dashboard/app.py
```

## **📢 Conclusion**
This project successfully demonstrates **sentiment analysis using XGBoost and TF-IDF** with a high accuracy of **97.89%**. 🚀  

---

Hope this looks neat! Let me know if you want any changes. 🔥🔥
