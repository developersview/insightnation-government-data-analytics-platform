import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os



# Function to classify new text input
def classify_text(text):
    # Load models and vectorizer
    # log_model = joblib.load("D:/insightnation-government-data-analytics-platform/models/logistic_model.pkl")
    # svm_model = joblib.load("D:/insightnation-government-data-analytics-platform/models/svm_model.pkl")
    # vectorizer = joblib.load("D:/insightnation-government-data-analytics-platform/models/tfidf_vectorizer.pkl")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    #print(f"Model Directory: {MODEL_DIR}")
    log_model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.pkl"))
    svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

    # Preprocess and vectorize the text
    text_vectorized = vectorizer.transform([text])
    
    # Predict using both models
    log_pred = log_model.predict(text_vectorized)
    svm_pred = svm_model.predict(text_vectorized)
    
    # Print results
    log_sentiment = "Positive" if log_pred == 1 else "Negative"
    svm_sentiment = "Positive" if svm_pred == 1 else "Negative"
    
    # print(log_pred)
    # print(svm_pred)
    
    print(f"Logistic Regression Prediction: {log_sentiment}")
    print(f"SVM Prediction: {svm_sentiment}")
    
    return log_sentiment, svm_sentiment


if __name__ == "__main__":
    text_input = input("Enter the text for classification: ")
    print(f"Provided Feedback: {text_input}\n")
    classify_text(text_input)