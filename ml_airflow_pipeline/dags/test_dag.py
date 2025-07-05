from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum
from airflow.decorators import task

def hello():
    print("Hello from Airflow!")

with DAG(
    dag_id="test_dag",
    start_date=pendulum.datetime(2024, 7, 1, tz="UTC"),
    schedule="@daily",
    catchup=False
) as dag:
    
    @task
    def say_hello():
        """
        Task to print a hello message.
        """
    
    hello()
