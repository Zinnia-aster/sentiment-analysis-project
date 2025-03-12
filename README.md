# **📊 Sentiment Analysis on Reviews using XGBoost**

## **📌 Project Overview**
This project focuses on sentiment analysis of user reviews using **TF-IDF vectorization** and an **XGBoost classifier** to classify sentiments as positive or negative. The model achieves an impressive **97.89% accuracy** in predicting sentiment.

### **📂 Dataset**  
- The dataset contains **880 user reviews** about the anime *One Piece*.  
- Each review is accompanied by a **rating** indicating the user’s sentiment.  
- The dataset was **cleaned** by removing duplicates and null values.  
- Sentiments were categorized as:  
  - **Positive (1)** → Ratings **≥ 3**  
  - **Negative (0)** → Ratings **< 3**  
- Since the dataset had **more positive reviews than negative ones**, **downsampling** was applied to balance the classes.  

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
- Achieved an **accuracy of 97.89%** on test data. 

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

