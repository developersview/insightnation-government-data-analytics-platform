from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum
from airflow.decorators import task
from datetime import datetime
from include.tasks.data_cleaning import clean_data
from include.tasks.nlp_pipeline import run_nlp_pipeline
from include.tasks.model_trainer import train_model

RAW_DATA_PATH = "./data/raw/citizen_feedback.csv"
CLEANED_DATA_PATH = "./data/preprocessed/cleaned_citizen_feedback.csv"
CLEANED_DATA_WITH_TEXT_PATH = "./data/processed/cleaned_citizen_feedback_with_text.csv"
OUTPUT_MODEL_LOGISTIC_REGRESSION = "./models/logistic_model.pkl"
OUTPUT_MODEL_SVM = "./models/svm_model.pkl"
OUTPUT_VECTORIZER = "./models/tfidf_vectorizer.pkl"

default_args = {
    "owner": "pranoy",
    "start_date": pendulum.datetime(2025, 7, 5, tz="UTC"),
    #"retries": 1,
}

#create the DAG
with DAG(
    dag_id="insightnation_ml_pipeline",
    default_args=default_args,
    description="Automated NLP and ML training pipeline for InsightNation",
    schedule="*/30 * * * *",
    catchup=False,
) as dag:
    
    # Task 1: Data Cleaning
    @task
    def clean_data_task():
        """
        Task to clean raw citizen feedback data and save to intermediate CSV.
        """
        clean_data(RAW_DATA_PATH, CLEANED_DATA_PATH)
        print("Data cleaning completed and saved to:", CLEANED_DATA_PATH)
        return CLEANED_DATA_PATH


    # Task 2: NLP Pipeline
    @task
    def nlp_pipeline_task(CLEANED_DATA_PATH):
        """
        Task to run NLP preprocessing on cleaned data and save to processed CSV.
        """
        run_nlp_pipeline(CLEANED_DATA_PATH, CLEANED_DATA_WITH_TEXT_PATH)
        print("NLP preprocessing completed and saved to:", CLEANED_DATA_WITH_TEXT_PATH)
        return CLEANED_DATA_WITH_TEXT_PATH


    # Task 3: Model Training
    @task
    def model_trainer_task(CLEANED_DATA_WITH_TEXT_PATH):
        """
        Task to train ML models on processed data and save them.
        """
        train_model(CLEANED_DATA_WITH_TEXT_PATH, 
                    OUTPUT_MODEL_LOGISTIC_REGRESSION, 
                    OUTPUT_MODEL_SVM, 
                    OUTPUT_VECTORIZER)
        print("Model training completed and models saved.")

    # Chain the tasks with data passed via XCom
    cleaned_path = clean_data_task()
    processed_path = nlp_pipeline_task(cleaned_path)
    model_trainer_task(processed_path)